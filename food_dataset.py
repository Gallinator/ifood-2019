import random
from typing import Optional, Callable
from torchvision.datasets import ImageFolder


class FoodDataset(ImageFolder):
    """
    Simple ImageFolder dataloader which provides an additional dict to convert from class index to label
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    @property
    def num_classes(self):
        return len(self.classes)


class SSLFoodDataset(ImageFolder):
    def __init__(self, root: str, transform, permset):
        super().__init__(root, transform=transform)
        self.permset = permset
        self.perm_labels = random.choices(range(len(permset)), k=len(self))

    def shuffle_jigsaw(self, inpt):
        label = random.randint(0, len(self.permset) - 1)
        inpt = inpt[self.permset[label]]
        return inpt, label

    def __getitem__(self, item):
        """
        Discards the original label and uses the permutation instead.
        :param item: a tuple of ((tensor, perm_label), class_label)
        :return: a tuple of (tensor, perm_label)
        """
        img, _ = super().__getitem__(item)
        return self.shuffle_jigsaw(img)
