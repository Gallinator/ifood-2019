import argparse
import os.path

import torch
from torch.utils.data import DataLoader
from torchinfo import torchinfo

from data_preprocessing import DATA_TRANSFORM, SSL_PER_TILE_TRANSFORM
from food_dataset import FoodDataset, SSLFoodDataset
import lightning as L

from model import FoodCNN, FoodSSL, ConvNet
from transforms import SUP_TRAIN_TRANSFORM, SUP_VAL_TRANSFORM


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
    return arg_parser


def train(train_dir: str, val_dir: str, weights_dir: str, use_pretrained_conv_net: bool = False):
    train_data = FoodDataset(train_dir, transform=SUP_TRAIN_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=14, persistent_workers=True)
    val_data = FoodDataset(val_dir, transform=SUP_VAL_TRANSFORM)
    val_loader = DataLoader(val_data, batch_size=256, num_workers=14, persistent_workers=True)
    trainer = L.Trainer(devices='auto',
                        enable_progress_bar=True,
                        enable_checkpointing=False,
                        limit_train_batches=0.5,
                        limit_val_batches=0.5,
                        max_epochs=1,
                        enable_model_summary=False,
                        callbacks=[ModuleSummaryCallback()])

    conv_net = None
    if use_pretrained_conv_net:
        conv_net = ConvNet(False)
        conv_net.load_state_dict(torch.load(os.path.join(weights_dir, 'ssl_conv_net.pt')))
    model = FoodCNN(len(train_data.classes), conv_net)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join(weights_dir, 'cnn.ckpt'))


def ssl_train(train_dir: str, val_dir: str, weights_dir: str):
    train_data = SSLFoodDataset(train_dir, 3, SSL_PER_TILE_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True)
    val_data = SSLFoodDataset(val_dir, 3, SSL_PER_TILE_TRANSFORM)
    val_loader = DataLoader(val_data, batch_size=4, num_workers=8, persistent_workers=True)
    trainer = L.Trainer(devices='auto',
                        enable_progress_bar=True,
                        enable_checkpointing=False,
                        limit_train_batches=0.1,
                        limit_val_batches=0.1,
                        max_epochs=1,
                        enable_model_summary=False,
                        callbacks=[ModuleSummaryCallback()])

    model = FoodSSL(2, 3)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join(weights_dir, 'ssl_cnn.ckpt'))
    torch.save(model.conv_net.state_dict(), os.path.join(weights_dir, 'ssl_conv_net.pt'))


def get_ssl_data_dir(data_dir: str) -> str:
    """
    :param data_dir: source data path
    :return: the directory path of the self supervised version of the data
    """
    dest_dir = os.path.basename(data_dir)
    return os.path.join(os.path.split(data_dir)[0], 'ssl', dest_dir)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    match args.type:
        case 'full':
            ssl_train(get_ssl_data_dir(args.train_dir), get_ssl_data_dir(args.val_dir), args.weights_dir)
            train(args.train_dir, args.val_dir, args.weights_dir, args.use_ssl_pretrained)
        case 'sup':
            train(args.train_dir, args.val_dir, args.weights_dir, args.use_ssl_pretrained)
        case 'selfsup':
            ssl_train(get_ssl_data_dir(args.train_dir), get_ssl_data_dir(args.val_dir), args.weights_dir)
