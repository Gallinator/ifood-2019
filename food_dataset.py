from typing import Optional, Callable
from torchvision.datasets import ImageFolder


class FoodDataset(ImageFolder):
    """
    Simple ImageFolder dataloader which provides an additional dict to convert from class index to label
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}


class SSLFoodDataset(ImageFolder):
    def __init__(self, root: str, transform):
        super().__init__(root, transform=transform)

    def __getitem__(self, item):
        """
        Discards the original label and uses the permutation instead.
        :param item: a tuple of ((tensor, perm_label), class_label)
        :return: a tuple of (tensor, perm_label)
        """
        img, label = super().__getitem__(item)
        return img
