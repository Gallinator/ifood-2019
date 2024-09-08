import os.path
from typing import Any

import lightning as L
import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn, Tensor
from torch.nn import Flatten
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy
import skops.io as sio

from food_dataset import FoodDataset
from visualization import plot_boosting_losses, plot_pca_variance


class InvertedResidual(nn.Module):
    """
    Implementation of an inverted residual block. The residual is added only if in_channels==out_channels and stride==1.\n
    The first pointwise convolution is skipped if no expand==1
    """

    def __init__(self, in_channels, out_channels, expand: int, stride=1, *args, **kwargs):
        """
        :param in_channels: the number of input channels
        :param out_channels: the number of output channels
        :param expand: the expansion coefficient. The internal number of channel is calculated as in_channels*expand
        :param stride: the stride
        :param args:
        :param kwargs:
        """
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
            nn.Conv2d(1280 // 4, 1024, 1),
            Flatten(1)
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
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, self.n_classes)
        )

    def forward(self, x):
        y = self.conv_net(x)
        return self.linear(y)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label, label_smoothing=0.1)
        acc = multiclass_accuracy(y, label, self.n_classes, average="micro")
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Training accuracy", acc * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label, label_smoothing=0.1)
        acc = multiclass_accuracy(y, label, self.n_classes, average="micro")
        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation accuracy", acc * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        """
        :param batch: input of size (batch_size,c,w,h)
        :return: a tuple containing the predicted classes and predicted probabilities
        """
        x, _ = batch
        y = self.forward(x)
        y = nn.functional.softmax(y, dim=1)
        return torch.argmax(y, dim=1), y

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=0.18, weight_decay=0.00004, momentum=0.9)
        return {'optimizer': opt, 'lr_scheduler': StepLR(opt, 1, 0.985)}


class FoodSSL(L.LightningModule):
    def __init__(self, num_perm: int, grid_size: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_perm = num_perm
        self.num_tiles = grid_size * grid_size
        self.conv_net = ConvNet(ssl_stride=True)
        self.shared = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(64 * grid_size ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_perm)
        )

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=0.025, weight_decay=0.00004, momentum=0.9)
        return {'optimizer': opt, 'lr_scheduler': StepLR(opt, 1, 0.995)}

    def forward(self, x):
        shared_outs = []
        for t in x.tensor_split(self.num_tiles, dim=1):
            o = self.conv_net(torch.squeeze(t))
            shared_outs.append(self.shared(o))
        y = torch.cat(shared_outs, dim=1)
        return self.linear(y)

    def training_step(self, batch, *args: Any, **kwargs: Any):
        tiles, labels = batch
        y = self.forward(tiles)
        loss = nn.functional.cross_entropy(y, labels)
        acc = multiclass_accuracy(y, labels, self.num_perm, average="micro")
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Training accuracy", acc * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any):
        img, label = batch
        y = self.forward(img)
        loss = nn.functional.cross_entropy(y, label)
        acc = multiclass_accuracy(y, label, self.num_perm, average="micro")
        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Validation accuracy", acc * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


class TraditionalFoodClassifier:
    def __init__(self, conv_net: ConvNet, device, repr_scaler, classifier=None, pca=None):
        self.conv_net = conv_net
        self.conv_net.to(device)
        self.classifier = classifier if classifier else HistGradientBoostingClassifier(
            verbose=2,
            learning_rate=0.02,
            max_iter=100,
            max_features=0.5,
            early_stopping=True,
            l2_regularization=0.1,
            random_state=8421
        )
        self.device = device
        self.repr_scaler = repr_scaler
        self.pca = pca if pca else PCA(n_components=7840, svd_solver='full', random_state=8421)
        self.n_pca_comps = 728

    def extract_representations(self, dataset: FoodDataset):
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=14, persistent_workers=True)
        representations = []
        labels = []

        # Extract features from last inverted residual block
        for batch in tqdm.tqdm(dataloader, desc='Extracting data representation'):
            x, label = batch
            y = x.to(self.device)
            for i, l in enumerate(self.conv_net.layers):
                y = l.forward(y)
                if i == 8:
                    break
            y = torch.flatten(y, start_dim=1)
            y = y.to('cpu').detach()
            representations.append(y)
            labels.append(label)
            del x
            torch.cuda.empty_cache()
        return torch.cat(representations, dim=0).numpy(force=True), torch.cat(labels, dim=0).numpy(force=True)

    def save(self, path: str):
        sio.dump(self.classifier, os.path.join(path, 'traditional_classifier.skops'))
        sio.dump(self.repr_scaler, os.path.join(path, 'repr_scaler.skops'))
        sio.dump(self.pca, os.path.join(path, 'pca.skops'))

    @staticmethod
    def load(root: str, device: torch.device):
        classifier_path = os.path.join(root, 'traditional_classifier.skops')
        scaler_path = os.path.join(root, 'repr_scaler.skops')
        pca_path = os.path.join(root, 'pca.skops')

        conv_net = ConvNet()
        conv_net.load_state_dict(torch.load(os.path.join(root, 'ssl_conv_net.pt')))

        classifier = sio.load(classifier_path, sio.get_untrusted_types(file=classifier_path))
        scaler = sio.load(scaler_path, sio.get_untrusted_types(file=scaler_path))
        pca = sio.load(pca_path, sio.get_untrusted_types(file=pca_path))

        return TraditionalFoodClassifier(conv_net, device, scaler, classifier, pca)

    def fit_transform_pca(self, representations) -> np.ndarray:
        print('PCA...')
        self.pca.fit(representations)
        plot_pca_variance(self.pca)
        return self.pca.transform(representations)[:, :self.n_pca_comps]

    def fit(self, dataset: FoodDataset):
        self.conv_net.eval()
        representations, labels = self.extract_representations(dataset)
        self.repr_scaler = self.repr_scaler.fit(representations)
        representations = self.repr_scaler.transform(representations)
        representations = self.fit_transform_pca(representations)

        print('Training classifier...')
        self.classifier.fit(representations, labels)

        plot_boosting_losses(-self.classifier.train_score_,
                             -self.classifier.validation_score_)
        return self

    def predict(self, img: Tensor | FoodDataset):
        self.conv_net.eval()
        representation = None
        if isinstance(img, Tensor):
            representation = self.conv_net(img.to(self.device))
        elif isinstance(img, FoodDataset):
            representation, _ = self.extract_representations(img)

        representation = self.repr_scaler.transform(representation)
        representation = self.pca.transform(representation)[:, :self.n_pca_comps]

        return self.classifier.predict(representation), self.classifier.predict_proba(representation)
