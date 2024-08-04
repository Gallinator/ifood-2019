import argparse
import csv
import os
import shutil
import tarfile

import numpy as np
import pandas as pd
import requests
import tqdm
from simple_file_checksum import get_checksum

TRAIN_URL = 'https://food-x.s3.amazonaws.com/train.tar'
TRAIN_CHECKSUM = '8e56440e365ee852dcb0953a9307e27f'
VAL_URL = 'https://food-x.s3.amazonaws.com/val.tar'
VAL_CHECKSUM = 'fa9a4c1eb929835a0fe68734f4868d3b'
ANNOTATIONS_URL = 'https://food-x.s3.amazonaws.com/annot.tar'
ANNOTATIONS_CHECKSUM = '0c632c543ceed0e70f0eb2db58eda3ab'


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
    print(f'Extracting {os.path.basename(file_path)}...')
    with tarfile.open(file_path) as src:
        src.extractall(dest_path)


def clean_data_dir(data_path: str):
    """
    Removes all files but the source tars
    :param data_path: data directory
    """
    for f in os.listdir(data_path):
        f_path = os.path.join(data_path, f)
        if os.path.isdir(f_path):
            shutil.rmtree(f_path)
        elif f not in ['annot.tar', 'train.tar', 'val.tar']:
            os.remove(f_path)


def create_val_set(data_dir: str, val_size: float):
    """
    Creates the validation set from the training set. Images are moved from the training to the validation directory.
    To manage the train and evaluation info and images at the same time, a set of indices for each split is created randomly and used for all operations.
    :param val_size: the size of the validation set in [0,1]
    :param data_dir: directory containing the training data
    """
    train_info = pd.read_csv(os.path.join(data_dir, 'train_info.csv'), sep=',', names=['Image', 'Class'])
    data_size = len(train_info)
    rand_idx = np.random.permutation(range(data_size))
    num_train = int(data_size * (1 - val_size))
    train_idx = rand_idx[:num_train]
    val_idx = rand_idx[num_train:]

    val_info = train_info.iloc[val_idx, :]
    train_info = train_info.iloc[train_idx, :]
    train_info = train_info.reset_index(drop=True)

    val_info.to_csv(os.path.join(data_dir, 'val_info.csv'), sep=',', index=False, header=False)
    train_info.to_csv(os.path.join(data_dir, 'train_info.csv'), sep=',', index=False, header=False)

    train_dir = os.path.join(data_dir, 'train_set')
    val_dir = os.path.join(data_dir, 'val_set')
    os.makedirs(val_dir, exist_ok=True)

    for _, row in val_info.iterrows():
        image_name = row['Image']
        shutil.move(os.path.join(train_dir, image_name), os.path.join(val_dir, image_name))


def build_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-dir', '-d', type=str, default='data',
                            help='directory to store the preprocessed data into')
    arg_parser.add_argument('--download-dir', '-dl', type=str, default='data',
                            help='directory to store the downloaded data into. The download size is bout 3 Gb')
    arg_parser.add_argument('--remove-src', '-rs', type=bool, default=True,
                            help='remove the source tar files')
    arg_parser.add_argument('--train-size', '-ts', type=float, default=0.8,
                            help='size of the train set. Must be in [0,1]')
    return arg_parser


def main():
    args = build_arg_parser().parse_args()
    download_dir = args.download_dir
    data_dir = args.data_dir

    annotations_tar = download_data(ANNOTATIONS_URL, ANNOTATIONS_CHECKSUM, download_dir)
    train_tar = download_data(TRAIN_URL, TRAIN_CHECKSUM, download_dir)
    val_tar = download_data(VAL_URL, VAL_CHECKSUM, download_dir)

    print('Cleaning old files...')
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
    os.rename(os.path.join(data_dir, 'val_set'), os.path.join(data_dir, 'test_set'))

    create_val_set(data_dir, 1 - args.train_size)


if __name__ == '__main__':
    main()
