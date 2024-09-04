import argparse
import os.path

import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torchinfo import torchinfo

from food_dataset import FoodDataset, SSLFoodDataset
import lightning as L

from model import FoodCNN, FoodSSL, ConvNet, TraditionalFoodClassifier
from transforms import SUP_TRAIN_TRANSFORM, SUP_VAL_TRANSFORM, SSL_DATA_TRANSFORM, MixCollate


class ModuleSummaryCallback(L.Callback):
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print('\n')
        torchinfo.summary(pl_module)
        print('\n')


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-dir', type=str, default='data/train_set',
                            help='directory containing the training data')
    arg_parser.add_argument('--val-dir', type=str, default='data/val_set',
                            help='directory containing the training data')
    arg_parser.add_argument('--weights-dir', type=str, default='weights',
                            help='directory to store the trained model weights into')
    arg_parser.add_argument('--type', type=str, default='sup',
                            help='type of training. Supported values are (full, sup, selfsup)')
    arg_parser.add_argument('--use-ssl-pretrained', action='store_true', default=False,
                            help='whether or not to use self supervised pretrained weights.Affects both sup and full training types')
    arg_parser.add_argument('--ssl-permutations', type=str, default='data/ssl_permutations.npy',
                            help='path to the file containing the jigsaw permutations')
    return arg_parser


def train(train_dir: str, val_dir: str, weights_dir: str, use_pretrained_conv_net: bool = False):
    train_data = FoodDataset(train_dir, transform=SUP_TRAIN_TRANSFORM)
    train_loader = DataLoader(train_data,
                              batch_size=128,
                              num_workers=14,
                              shuffle=True,
                              persistent_workers=True,
                              collate_fn=MixCollate(train_data.num_classes))

    val_data = FoodDataset(val_dir, transform=SUP_VAL_TRANSFORM)
    val_loader = DataLoader(val_data, batch_size=256, num_workers=14, persistent_workers=True)
    trainer = L.Trainer(devices='auto',
                        enable_progress_bar=True,
                        max_epochs=100,
                        enable_model_summary=False,
                        callbacks=[ModuleSummaryCallback(), LearningRateMonitor('epoch')])
    conv_net = None
    if use_pretrained_conv_net:
        conv_net = ConvNet(False)
        conv_net.load_state_dict(torch.load(os.path.join(weights_dir, 'ssl_conv_net.pt')))
    model = FoodCNN(train_data.num_classes, conv_net)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join(weights_dir, 'cnn.ckpt'))


def ssl_train(train_dir: str, val_dir: str, weights_dir: str, perms_path: str):
    permset = torch.tensor(np.load(perms_path))
    train_data = SSLFoodDataset(train_dir, SSL_DATA_TRANSFORM, permset)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=14, persistent_workers=True)
    val_data = SSLFoodDataset(val_dir, SSL_DATA_TRANSFORM, permset)
    val_loader = DataLoader(val_data, batch_size=256, num_workers=14, persistent_workers=True)
    trainer = L.Trainer(devices='auto',
                        enable_progress_bar=True,
                        max_epochs=100,
                        enable_model_summary=False,
                        callbacks=[ModuleSummaryCallback(), LearningRateMonitor('epoch')])

    model = FoodSSL(2, 3)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join(weights_dir, 'ssl_cnn.ckpt'))
    torch.save(model.conv_net.state_dict(), os.path.join(weights_dir, 'ssl_conv_net.pt'))


def train_classifier(weights_dir: str, train_dir: str, val_dir: str):
    conv_net = ConvNet()
    conv_net.load_state_dict(torch.load(os.path.join(weights_dir, 'ssl_conv_net.pt')))
    train_data = FoodDataset(train_dir, transform=SUP_TRAIN_TRANSFORM)
    val_data = FoodDataset(val_dir, transform=SUP_VAL_TRANSFORM)

    classifier = TraditionalFoodClassifier(conv_net, torch.device('cuda'),
                                           MinMaxScaler(),
                                           DecisionTreeClassifier(max_depth=1, max_features=10))

    classifier.fit(train_data)
    classifier.save(weights_dir)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    match args.type:
        case 'full':
            ssl_train(args.train_dir, args.val_dir, args.weights_dir, args.ssl_permutations)
            train(args.train_dir, args.val_dir, args.weights_dir, args.use_ssl_pretrained)
        case 'sup':
            train(args.train_dir, args.val_dir, args.weights_dir, args.use_ssl_pretrained)
        case 'selfsup':
            ssl_train(args.train_dir, args.val_dir, args.weights_dir, args.ssl_permutations)
        case 'classifier':
            train_classifier(args.weights_dir, args.val_dir, args.val_dir)
