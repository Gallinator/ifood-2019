import PIL.Image
from matplotlib import pyplot as plt


def plot_image(image: PIL.Image.Image, label: str):
    plt.imshow(image)
    plt.title(label.replace('_', ' ').capitalize(), fontsize='xx-large')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
