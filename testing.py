import argparse

import numpy as np
import torch
from sklearn.metrics import top_k_accuracy_score, confusion_matrix, f1_score, recall_score, \
    precision_score
from torch.utils.data import DataLoader

from food_dataset import FoodDataset
from model import FoodCNN, TraditionalFoodClassifier
import lightning as L

from transforms import SUP_VAL_TRANSFORM
from visualization import plot_metrics


def calc_metrics(pred, pred_proba, target):
    acc1 = top_k_accuracy_score(target, pred_proba, k=1)
    acc3 = top_k_accuracy_score(target, pred_proba, k=3)
    cmatrix = confusion_matrix(target, pred)
    f1 = f1_score(target, pred, average='macro')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    plot_metrics(cmatrix, Accuracy=acc1, Top3=acc3, F1Score=f1, Recall=recall, Precision=precision)


def test_cnn(test_data: str, model_checkpoint: str):
    data = FoodDataset(test_data, SUP_VAL_TRANSFORM)
    model = FoodCNN.load_from_checkpoint(model_checkpoint)
    loader = DataLoader(data, shuffle=False, num_workers=14, persistent_workers=True)
def evaluate_torch_model(test_data: ImageFolder, model: L.LightningModule):
    loader = DataLoader(test_data, shuffle=False, num_workers=14, persistent_workers=True)
    trainer = L.Trainer()
    pred = trainer.predict(model, loader)
    pred, pred_proba = list(zip(*pred))
    pred = np.array([p.numpy(force=True) for p in pred]).flatten()
    pred_proba = np.array([p.numpy(force=True)[0] for p in pred_proba])
    targets = list(zip(*test_data.samples))[1]
    calc_metrics(pred, pred_proba, targets)


def test_classifier(test_data: ImageFolder, weights_dir: str):
    model = TraditionalFoodClassifier.load(weights_dir, torch.device('cuda'))
    pred, pred_proba = model.predict(test_data)
    targets = list(zip(*test_data.samples))[1]

    calc_metrics(pred, pred_proba, targets)


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', type=str, default='data/test_set',
                            help='directory containing the evaluation data')
    arg_parser.add_argument('--classifier-dir', type=str, default='weights',
                            help='directory containing the traditional classifier')
    arg_parser.add_argument('--cnn-checkpoint', type=str, default='weights/cnn.ckpt',
                            help='path to the Pytorch Lightning model checkpoint')
    return arg_parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    data = FoodDataset(args.data_dir, SUP_VAL_TRANSFORM)

    if args.cnn_checkpoint:
        model = FoodCNN.load_from_checkpoint(args.cnn_checkpoint)
        evaluate_torch_model(data, model)

    if args.classifier_dir:
        test_classifier(data, args.classifier_dir)
