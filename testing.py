import numpy as np
import torch
import torchinfo
from sklearn.metrics import top_k_accuracy_score, roc_auc_score, confusion_matrix, f1_score, recall_score, \
    average_precision_score
from torch.utils.data import DataLoader

from food_dataset import FoodDataset
from model import FoodCNN, TraditionalFoodClassifier
import lightning as L

from transforms import SUP_VAL_TRANSFORM
from visualization import plot_metrics


def calc_metrics(pred, pred_proba, target):
    acc1 = top_k_accuracy_score(target, pred_proba, k=1)
    acc3 = top_k_accuracy_score(target, pred_proba, k=3)
    auc = roc_auc_score(target, pred_proba, multi_class='ovo')
    cmatrix = confusion_matrix(target, pred)
    f1 = f1_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    m_ap = average_precision_score(target, pred_proba)
    plot_metrics(cmatrix, Accuracy=acc1, Top3=acc3, AUC=auc, F1Score=f1, Recall=recall, MAP=m_ap)


def test_cnn(test_data: str, model_checkpoint: str):
    data = FoodDataset(test_data, SUP_VAL_TRANSFORM)
    model = FoodCNN.load_from_checkpoint(model_checkpoint)
    loader = DataLoader(data, shuffle=False, num_workers=14, persistent_workers=True)
    trainer = L.Trainer()
    pred = trainer.predict(model, loader)
    pred, pred_proba = list(zip(*pred))
    pred = np.array([p.numpy(force=True) for p in pred]).flatten()
    pred_proba = np.array([p.numpy(force=True)[0] for p in pred_proba])
    targets = list(zip(*data.samples))[1]
    calc_metrics(pred, pred_proba, targets)


def test_classifier(test_data: str, weights_dir: str):
    data = FoodDataset(test_data, SUP_VAL_TRANSFORM)
    model = TraditionalFoodClassifier.load(weights_dir, torch.device('cuda'))

    pred, pred_proba = model.predict(data)
    targets = list(zip(*data.samples))[1]

    calc_metrics(pred, pred_proba, targets)
