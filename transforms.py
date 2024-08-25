import torch
from torch import nn
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
        imgs, label = inpt
        out = torch.zeros((len(imgs), 3, self.final_size, self.final_size))
        for i, img in enumerate(imgs):
            out[i] = self.transform(img)
        return out, label


class ShuffleJigSaw(nn.Module):
    def __init__(self, perms: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perms = perms
        self.perms_size = len(perms)

    def forward(self, inpt: torch.Tensor):
        label = torch.randint(high=self.perms_size, size=(1,))
        perm = self.perms[label]
        return inpt[perm], label[0]


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

SUP_TRAIN_TRANSFORM = v2.Compose([v2.Resize(256),
                                  v2.CenterCrop(224),
                                  v2.RandomHorizontalFlip(),
                                  v2.ColorJitter(),
                                  v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize(NORM_MEAN, NORM_STD)])
SUP_VAL_TRANSFORM = v2.Compose([v2.Resize(256),
                                v2.CenterCrop(224),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(NORM_MEAN, NORM_STD)])

SSL_PER_TILE_TRANSFORM = v2.Compose([v2.RandomCrop(64),
                                     v2.Normalize(NORM_MEAN, NORM_STD)])

SSL_DATA_TRANSFORM = v2.Compose([v2.Resize(256),
                                 v2.CenterCrop(225),
                                 v2.ToTensor(),
                                 JigSaw(3),
                                 ShuffleJigSaw([[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 5, 6, 7, 8, 4, 3],
                                                [0, 3, 2, 8, 6, 7, 1, 4, 5]]),
                                 MultiImageTransform(64, SSL_PER_TILE_TRANSFORM)])
