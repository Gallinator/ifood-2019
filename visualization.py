import math
import random

import PIL.Image
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import Conv2d
from torchvision.utils import make_grid


def plot_filters(conv2d_layer: Conv2d):
    grid_size = math.ceil(math.sqrt(conv2d_layer.weight.shape[0]))
    grid = make_grid(conv2d_layer.weight,
                     nrow=grid_size,
                     normalize=True,
                     scale_each=True,
                     padding=1,
                     pad_value=1)
    grid = grid.numpy(force=True)
    plt.figure()
    plt.imshow(grid.transpose((1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_image(image: PIL.Image.Image, label: str):
    plt.imshow(image)
    plt.title(label.replace('_', ' ').capitalize(), fontsize='xx-large')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_counts(counts, classes):
    idx = np.flip(np.argsort(counts))
    counts = np.array(counts)[idx]
    classes = np.array(classes)[idx]
    x = np.arange(len(counts))
    plt.bar(x, counts, color='salmon')
    plt.title('Samples count')
    plt.xticks(x, classes, rotation=90)
    plt.show()


def plot_metrics(cmatrix, **scores):
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(18, 9))

    scores, values = list(scores.keys()), list(scores.values())
    cmap = matplotlib.colormaps['tab10']
    rng = random.Random(8421)
    colors = [cmap(rng.random()) for _ in scores]
    y = np.arange(len(scores))
    bars = axs[0].barh(y, width=values, color=colors)

    high_labels = [f'{v:.3f}' if v >= 0.3 else '' for v in values]
    low_labels = [f'{v:.3f}' if v < 0.3 else '' for v in values]
    axs[0].bar_label(bars, low_labels, padding=8, color='black')
    axs[0].bar_label(bars, high_labels, padding=-48, color='white')

    axs[0].set_yticks(y, scores)
    axs[0].set_xlim(left=0.0, right=1.0)

    axs[1].imshow(cmatrix)
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')
    axs[1].set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
