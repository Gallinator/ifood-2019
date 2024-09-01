from typing import Callable

import torch
from torch import nn
from torch.utils.data import default_collate
from torchvision.transforms import v2


class JigSaw(nn.Module):
    def __init__(self, grid_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size

    def forward(self, inpt):
        tiles = torch.stack(list(inpt.tensor_split(self.grid_size, dim=2)), 0)
        tiles = torch.stack(list(tiles.tensor_split(self.grid_size, dim=2)), 0)
        return tiles.flatten(start_dim=0, end_dim=1)


class MultiImageTransform(nn.Module):
    def __init__(self, final_size, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        self.final_size = final_size

    def forward(self, inpt):
        out = torch.zeros((len(inpt), 3, self.final_size, self.final_size))
        for i, img in enumerate(inpt):
            out[i] = self.transform(img)
        return out


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

SUP_TRAIN_TRANSFORM = v2.Compose([v2.Resize(256),
                                  v2.CenterCrop(224),
                                  v2.RandomHorizontalFlip(0.5),
                                  v2.RandomChoice([
                                      v2.ColorJitter(hue=0.4, saturation=0.4, brightness=0.4),
                                      v2.RandomRotation(20),
                                      v2.RandomHorizontalFlip(0)]),
                                  v2.PILToTensor(),
                                  v2.ToDtype(torch.float, scale=True),
                                  v2.Normalize(NORM_MEAN, NORM_STD),
                                  v2.ToPureTensor()])
SUP_VAL_TRANSFORM = v2.Compose([v2.Resize(256),
                                v2.CenterCrop(224),
                                v2.PILToTensor(),
                                v2.ToDtype(torch.float, scale=True),
                                v2.Normalize(NORM_MEAN, NORM_STD),
                                v2.ToPureTensor()])

SSL_PER_TILE_TRANSFORM = v2.Compose([v2.RandomCrop(64),
                                     v2.ToDtype(torch.float, scale=True),
                                     v2.Normalize(NORM_MEAN, NORM_STD),
                                     v2.ToPureTensor()])

SSL_DATA_TRANSFORM = v2.Compose([v2.Resize(256),
                                 v2.CenterCrop(225),
                                 v2.RandomHorizontalFlip(0.5),
                                 v2.RandomChoice([
                                     v2.ColorJitter(hue=0.4, saturation=0.4, brightness=0.4),
                                     v2.RandomHorizontalFlip(0)]),
                                 v2.PILToTensor(),
                                 JigSaw(3),
                                 MultiImageTransform(64, SSL_PER_TILE_TRANSFORM)])


class MixCollate(Callable):
    def __init__(self, num_classes: int):
        self.transform = v2.RandomChoice([v2.MixUp(num_classes=num_classes), v2.CutMix(num_classes=num_classes)])

    def __call__(self, batch):
        if torch.rand(1) > 0.5:
            return self.transform(*default_collate(batch))
        else:
            return default_collate(batch)
