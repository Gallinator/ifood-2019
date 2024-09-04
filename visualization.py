import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


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
    fig, axs = plt.subplots(1, 2)

    scores, values = list(scores.keys()), list(scores.values())
    y = np.arange(len(scores))
    axs[0].barh(y, width=values)
    axs[0].set_yticks(y, scores)
    axs[0].set_xlim(left=0.0, right=1.0)

    axs[1].imshow(cmatrix)
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')
    axs[1].set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
