import argparse
import csv
import itertools
import multiprocessing
import os
import shutil
import tarfile
import random

import numpy as np
import pandas as pd
import requests
import torch
import tqdm
from PIL import Image
from scipy.spatial.distance import hamming
from simple_file_checksum import get_checksum
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid

from data_cleaning import clean_data
from transforms import NORM_MEAN, NORM_STD, cut_tiles
from visualization import plot_counts

TRAIN_URL = 'https://food-x.s3.amazonaws.com/train.tar'
TRAIN_CHECKSUM = '8e56440e365ee852dcb0953a9307e27f'
VAL_URL = 'https://food-x.s3.amazonaws.com/val.tar'
VAL_CHECKSUM = 'fa9a4c1eb929835a0fe68734f4868d3b'
ANNOTATIONS_URL = 'https://food-x.s3.amazonaws.com/annot.tar'
ANNOTATIONS_CHECKSUM = '0c632c543ceed0e70f0eb2db58eda3ab'

DATA_TRANSFORM = v2.Compose([v2.Resize(256),
                             v2.CenterCrop(224),
                             v2.RandomHorizontalFlip(),
                             v2.ColorJitter(),
                             v2.ToImage(),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(NORM_MEAN, NORM_STD)])

SSL_DATA_TRANSFORM = v2.Compose([v2.Resize(256),
                                 v2.CenterCrop(225),
                                 v2.ToTensor()])

SSL_PER_TILE_TRANSFORM = v2.Compose([v2.RandomCrop(64),
                                     v2.Normalize(NORM_MEAN, NORM_STD)])


def get_max_permutation_set(length: int, set_size: int) -> list[tuple]:
    """
    Generates a set of permutations which maximizes the hamming distance between samples
    :param length: size of the permutation
    :param set_size: the size of the permutation set
    :return: the permutation set
    """
    permutations = list(itertools.permutations(range(length)))
    perm_set = []
    j = random.randint(0, length - 1)
    dist = np.zeros(len(permutations))

    for _ in tqdm.tqdm(range(set_size), desc='Creating permutation set'):
        perm_set.append(permutations[j])
        del permutations[j]
        ham = pairwise_distances([perm_set[-1]], permutations, metric=hamming, n_jobs=-1)
        dist = np.delete(dist, j)
        dist = dist + ham
        j = np.argmax(dist)
    return perm_set


def generate_tiles_task(path, files, grid_size: int, classes_dirs: list, permset):
    """
    This function is used by multiprocessing to parallelize the tile generation task.
    :param permset: the permutation set
    :param path: passed from os.walk()
    :param files: passed from os.walk()
    :param grid_size: size of the tiles grid
    :param classes_dirs: list containing the class subfolder
    """
    for f in files:
        fpath = os.path.join(path, f)
        img = SSL_DATA_TRANSFORM(Image.open(fpath))
        tiles = cut_tiles(grid_size, img)
        for i, p in enumerate(permset):
            t = tiles[list(p)]
            grid = make_grid(t, padding=0, nrow=grid_size)
            grid = to_pil_image(grid)
            grid.save(os.path.join(classes_dirs[i], f))


def create_ssl_set(src_dir: str, permset, grid_size: int):
    """
    Generates the unnormalized tiles for the Jigsaw pretext task from the source dataset.
    The generated data is stored a tensors for convenience.
    :param src_dir: path of the original data
    :param permset: permutations
    :param grid_size: size of the grid edge. Each permutation will have length grid_size**2
    """
    # Create dirs
    dest_dir = os.path.join(os.path.split(src_dir)[0], 'ssl', os.path.basename(src_dir))
    os.makedirs(dest_dir, exist_ok=True)
    classes_dirs = []
    for i, p in enumerate(permset):
        classes_dirs.append(os.path.join(dest_dir, str(p)))
        os.makedirs(classes_dirs[-1], exist_ok=True)

    with multiprocessing.Pool() as pool:
        args = [(pth, files, grid_size, classes_dirs, permset) for pth, folders, files in os.walk(src_dir)]
        pool.starmap(generate_tiles_task, args)


def parse_classes_dict(directory: str, int_to_label=False) -> dict[str, int] | dict[int, str]:
    with open(os.path.join(directory, 'class_list.txt')) as f:
        reader = csv.reader(f, delimiter=' ')
        if int_to_label:
            return {int(row[0]): row[1] for row in reader}
        else:
            return {row[1]: int(row[0]) for row in reader}


def check_md5(file_path: str, checksum: str):
    print(f'Checking {os.path.basename(file_path)} checksum...')
    correct = get_checksum(file_path, algorithm='MD5') == checksum
    if correct:
        print('Checksum verified!')
    else:
        print('Checksum mismatch! The downloaded file might not be safe')
    return correct


def download_data(url: str, checksum: str, dest_dir: str) -> str:
    """
    :param checksum: md5 checksum of the file
    :param url: the data url
    :param dest_dir: the download directory path
    :return: the path of the downloaded file
    """
    url_resource_name = os.path.basename(url)
    file_path = os.path.join(dest_dir, url_resource_name)
    if os.path.exists(file_path) and check_md5(file_path, checksum):
        print(f'The file {file_path} already exists, skipping download')
        return file_path
    with open(file_path, 'wb') as file:
        with requests.get(url, stream=True) as res:
            res.raise_for_status()
            total_b = int(res.headers.get('content-length', 0))
            with tqdm.tqdm(desc=url_resource_name, total=total_b, unit='b', unit_divisor=1024,
                           unit_scale=1) as progress:
                for chunk in res.iter_content(chunk_size=8192):
                    progress.update(len(chunk))
                    file.write(chunk)
    check_md5(file_path, checksum)
    return file_path


def extract_tar(file_path: str, dest_path: None | str):
    """
    :param file_path: tar file to extract
    :param dest_path: parent directory to extract the file to
    """
    with tarfile.open(file_path) as src:
        for m in tqdm.tqdm(src.getmembers(), desc=f'Extracting {os.path.basename(file_path)}'):
            src.extract(member=m, path=dest_path)


def clean_data_dir(data_path: str):
    """
    Removes all files but the source tars
    :param data_path: data directory
    """
    for f in tqdm.tqdm(os.listdir(data_path), desc='Cleaning old files'):
        f_path = os.path.join(data_path, f)
        if os.path.isdir(f_path):
            shutil.rmtree(f_path)
        elif f not in ['annot.tar', 'train.tar', 'val.tar', 'train_set_embeddings.csv']:
            os.remove(f_path)


def create_val_set(data_dir: str, val_size: float):
    """
    Creates the validation set from the training set. Images are moved from the training to the validation directory.
    To manage the train and evaluation info and images at the same time, a set of indices for each split is created randomly and used for all operations.
    :param val_size: the size of the validation set in [0,1]
    :param data_dir: directory containing the training data
    """
    train_dir = os.path.join(data_dir, 'train_set')
    imgs = []
    classes = []
    for r, d, files in os.walk(train_dir):
        for f in files:
            imgs.append(os.path.join(r, f))
            classes.append(os.path.split(r)[1])

    _, val_imgs = train_test_split(imgs, test_size=val_size, shuffle=True, stratify=classes)

    # Create validation directories
    val_dir = os.path.join(data_dir, 'val_set')
    for c in os.scandir(train_dir):
        _, class_name = os.path.split(c)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move
    for img in tqdm.tqdm(val_imgs, desc='Creating validation set', unit=' image'):
        val_path = str(img).replace('train_set', 'val_set')
        shutil.move(img, val_path)


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', type=str, default='data',
                            help='directory to store the preprocessed data into')
    arg_parser.add_argument('--download-dir', type=str, default='data',
                            help='directory to store the downloaded data into. The download size is bout 3 Gb')
    arg_parser.add_argument('--remove-src', action='store_true', default=False,
                            help='remove the source tar files')
    arg_parser.add_argument('--train-size', type=float, default=0.8,
                            help='size of the train set. Must be in [0,1]')
    arg_parser.add_argument('--generate-ssl', type=bool, default=False,
                            help='generate the self supervised learning datasets')
    arg_parser.add_argument('--ssl-perms', type=int, default=2,
                            help='number of permutations of the generated self supervised learning datasets')
    arg_parser.add_argument('--clean-data', action='store_true', default=False,
                            help='whether or not to clean the data')
    return arg_parser


def create_split_directory_structure(data_dir: str, split: str):
    split_dir = os.path.join(data_dir, f'{split}_set')
    split_info = pd.read_csv(os.path.join(data_dir, f'{split}_info.csv'), sep=',', names=['Image', 'Class'])
    classes_dict = parse_classes_dict(data_dir, True)

    # Create directories
    for c in split_info['Class'].unique():
        os.mkdir(os.path.join(split_dir, str(classes_dict[c])))

    # Move images
    for _, row in tqdm.tqdm(split_info.iterrows(), f'Creating {split}_set structure', total=len(split_info)):
        class_name = classes_dict[row['Class']]
        dest_dir = os.path.join(split_dir, str(class_name))
        shutil.move(os.path.join(split_dir, row['Image']), os.path.join(dest_dir, row['Image']))


def plot_labels_dist(info_path: str):
    counts = []
    classes = []
    for d in os.listdir(info_path):
        classes.append(d)
        counts.append(len(os.listdir(os.path.join(info_path, d))))
    plot_counts(counts, classes)


def main():
    args = build_arg_parser().parse_args()
    download_dir = args.download_dir
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train_set')
    val_dir = os.path.join(data_dir, 'val_set')
    test_dir = os.path.join(data_dir, 'test_set')

    annotations_tar = download_data(ANNOTATIONS_URL, ANNOTATIONS_CHECKSUM, download_dir)
    train_tar = download_data(TRAIN_URL, TRAIN_CHECKSUM, download_dir)
    val_tar = download_data(VAL_URL, VAL_CHECKSUM, download_dir)

    clean_data_dir(data_dir)

    extract_tar(annotations_tar, data_dir)
    extract_tar(train_tar, data_dir)
    extract_tar(val_tar, data_dir)

    if args.remove_src:
        print('Removing tar files...')
        os.remove(annotations_tar)
        os.remove(train_tar)
        os.remove(val_tar)

    os.replace(os.path.join(data_dir, 'val_info.csv'), os.path.join(data_dir, 'test_info.csv'))
    os.rename(val_dir, test_dir)

    create_split_directory_structure(data_dir, 'train')
    create_split_directory_structure(data_dir, 'test')

    if args.clean:
        clean_data(os.path.join(data_dir, 'train_set'))

    create_val_set(data_dir, 1 - args.train_size)

    plot_labels_dist(train_dir)
    plot_labels_dist(val_dir)

    if args.generate_ssl:
        perms = get_max_permutation_set(9, args.ssl_perms)
        create_ssl_set(train_dir, perms, 3)
        create_ssl_set(val_dir, perms, 3)


if __name__ == '__main__':
    main()
