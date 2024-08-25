import torch
from torchvision.transforms import v2

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

SSL_DATA_TRANSFORM = v2.Compose([v2.Resize(256),
                                 v2.CenterCrop(225),
                                 v2.ToTensor()])

SSL_PER_TILE_TRANSFORM = v2.Compose([v2.RandomCrop(64),
                                     v2.Normalize(NORM_MEAN, NORM_STD)])


def cut_tiles(grid_size: int, inpt: torch.Tensor) -> torch.Tensor:
    """
    Given an input tensor cuts tiles.
    If the image is made of the tiles:
    >>>[[0,1],
    >>> [2,3]]
    Returns the tiles in the order
    >>>[0,1,2,3]
    :param grid_size: size of the cut grid. grid_size**2 tiles will be cut.
    :param inpt: source tensor
    :return: a list of tiles ordered row wise from left to right
    """
    tiles = torch.stack(list(inpt.tensor_split(grid_size, dim=2)), 0)
    tiles = torch.stack(list(tiles.tensor_split(grid_size, dim=2)), 0)
    tiles = tiles.flatten(start_dim=0, end_dim=1)
    return tiles
