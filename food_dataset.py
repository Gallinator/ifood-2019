from typing import Optional, Callable

from torchvision.datasets import ImageFolder


class FoodDataset(ImageFolder):
    """
    Simple ImageFolder dataloader which provides an additional dict to convert from class index to label
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
