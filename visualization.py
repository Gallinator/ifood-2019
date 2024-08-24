import PIL.Image
import numpy as np
from matplotlib import pyplot as plt


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
