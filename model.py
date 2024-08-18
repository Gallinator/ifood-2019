from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import Flatten
from torch.optim import AdamW
from torchmetrics.functional.classification import multiclass_accuracy


class ConvNet(L.LightningModule):
    def __init__(self, ssl_stride: bool = False, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.conv = nn.Sequential(
            EncoderBlock(3, 32, 4, 1 if ssl_stride else 3, 1, dropout=0.01),
            EncoderBlock(32, 64, 4, 1 if ssl_stride else 2, 1, dropout=0.01),
            EncoderBlock(64, 64, 4, 2, 1, dropout=0.01),
            EncoderBlock(64, 128, 4, 2, 1, dropout=0.01),
            EncoderBlock(128, 256, 2, 2, dropout=0.01),
            EncoderBlock(256, 256, 2, 2, dropout=0.01),
            EncoderBlock(256, 256, 2, 2, dropout=0.01))

    def forward(self, x) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True, dropout=0.005,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.use_batch_norm = batch_norm
        self.f = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        return self.f(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True, dropout=0.005,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample = nn.Upsample(scale_factor=2)
        self.padding = nn.ReplicationPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.use_batch_norm = batch_norm
        self.f = nn.LeakyReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.padding(x)
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        return self.f(x)


class FoodCNN(L.LightningModule):
    def __init__(self, n_classes, conv_net: ConvNet | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.conv_net = conv_net if conv_net else ConvNet()
        self.linear = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x):
        y = self.conv_net(x)
        y = torch.flatten(y, start_dim=1)
        return self.linear(y)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label)
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
        return AdamW(self.parameters(), lr=0.001)


class FoodSSL(L.LightningModule):
    def __init__(self, num_perm: int, grid_size: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_perm = num_perm
        self.num_tiles = grid_size * grid_size
        self.conv_net = ConvNet(ssl_stride=True)
        self.shared = nn.Sequential(
            self.conv_net,
            Flatten(start_dim=1),
            nn.Linear(256, 64),
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(64 * grid_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_perm)
        )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)

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
