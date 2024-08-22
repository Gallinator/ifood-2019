from typing import Optional, Callable

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from transforms import cut_tiles


class FoodDataset(ImageFolder):
    """
    Simple ImageFolder dataloader which provides an additional dict to convert from class index to label
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}


class SSLFoodDataset(ImageFolder):
    def __init__(self, root: str, grid_size: int, per_tile_transform=None):
        super().__init__(root, transform=v2.Compose([v2.ToTensor()]))
        self.per_tile_transform = per_tile_transform
        self.grid_size = grid_size

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        tiles = cut_tiles(self.grid_size, img)
        if self.per_tile_transform:
            tiles = torch.vstack([self.per_tile_transform(t) for t in tiles])
        return tiles, label
