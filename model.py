from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import Flatten
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional.classification import multiclass_accuracy


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand: int, stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_residual = stride == 1 and in_channels == out_channels
        expand_channels = expand * in_channels
        layers = []
        if expand != 1:
            layers.extend([
                nn.Conv2d(in_channels, expand_channels, 1),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6()
            ])
        layers.extend([
            nn.Conv2d(expand_channels, expand_channels, 3, padding=1, stride=stride, groups=expand_channels),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU6(),
            nn.Conv2d(expand_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        if self.use_residual:
            y = x + y
        return y


class InvertedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand, stride=1, repeat=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [InvertedResidual(in_channels, out_channels, expand, stride)]
        for i in range(repeat - 1):
            layers.append(InvertedResidual(out_channels, out_channels, expand))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvNet(L.LightningModule):
    def __init__(self, ssl_stride: bool = False, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32 // 2, 3, 2, 1),
            nn.ReLU6(),
            InvertedBlock(32 // 2, 16 // 2, 1),
            InvertedBlock(16 // 2, 24 // 2, 3, 2, 2),
            InvertedBlock(24 // 2, 32 // 2, 3, 2, 3),
            InvertedBlock(32 // 2, 64 // 2, 3, 2, 4),
            InvertedBlock(64 // 2, 96 // 2, 3, 1, 3),
            InvertedBlock(96 // 2, 160 // 2, 3, 2, 3),
            InvertedBlock(160 // 2, 320 // 2, 3),
            nn.Conv2d(320 // 2, 1280 // 4, 1),
            nn.AvgPool2d(7, 2, 3 if ssl_stride else 0),
            nn.Conv2d(1280 // 4, 1024, 1)
        )

    def forward(self, x) -> torch.Tensor:
        return self.layers(x)


class FoodCNN(L.LightningModule):
    def __init__(self, n_classes, conv_net: ConvNet | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.conv_net = conv_net if conv_net else ConvNet()
        self.linear = nn.Sequential(
            Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(1024, self.n_classes),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.conv_net(x)
        return self.linear(y)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label)
        y_proba = nn.functional.softmax(y, dim=1)
        acc = multiclass_accuracy(y_proba, label, num_classes=self.n_classes)
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Training accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label)
        y_proba = nn.functional.softmax(y, dim=1)
        acc = multiclass_accuracy(y_proba, label, num_classes=self.n_classes)
        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=0.045, weight_decay=0.00004, momentum=0.9)
        return {'optimizer': opt, 'lr_scheduler': StepLR(opt, 1, 0.98)}


class FoodSSL(L.LightningModule):
    def __init__(self, num_perm: int, grid_size: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_perm = num_perm
        self.num_tiles = grid_size * grid_size
        self.conv_net = ConvNet(ssl_stride=True)
        self.shared = nn.Sequential(
            self.conv_net,
            Flatten(1),
            nn.Linear(1024, 64),
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(64 * grid_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_perm)
        )

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=0.045, weight_decay=0.00004, momentum=0.9)
        return {'optimizer': opt, 'lr_scheduler': StepLR(opt, 1, 0.98)}

    def forward(self, x):
        shared_outs = []
        for t in x.tensor_split(self.num_tiles, dim=1):
            shared_outs.append(self.shared(t))
        y = torch.cat(shared_outs, dim=1)
        return self.linear(y)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        tiles, labels = batch
        y = self.forward(tiles)
        loss = nn.functional.cross_entropy(y, labels)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label)
        y_proba = nn.functional.softmax(y, dim=1)
        acc = multiclass_accuracy(y_proba, label, num_classes=self.num_perm)
        self.log("Validation loss", loss, on_step=False, on_epoch=True)
        self.log("Validation accuracy", acc, on_step=False, on_epoch=True)
        return loss
