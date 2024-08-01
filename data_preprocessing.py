import argparse
import os
import shutil
import tarfile

import requests
import tqdm
from simple_file_checksum import get_checksum

TRAIN_URL = 'https://food-x.s3.amazonaws.com/train.tar'
TRAIN_CHECKSUM = '8e56440e365ee852dcb0953a9307e27f'
VAL_URL = 'https://food-x.s3.amazonaws.com/val.tar'
VAL_CHECKSUM = 'fa9a4c1eb929835a0fe68734f4868d3b'
ANNOTATIONS_URL = 'https://food-x.s3.amazonaws.com/annot.tar'
ANNOTATIONS_CHECKSUM = '0c632c543ceed0e70f0eb2db58eda3ab'


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


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    download_dir = args.download_dir
    data_dir = args.data_dir

    annotations_tar = download_data(ANNOTATIONS_URL, ANNOTATIONS_CHECKSUM, download_dir)
    train_tar = download_data(TRAIN_URL, TRAIN_CHECKSUM, download_dir)
    val_tar = download_data(VAL_URL, VAL_CHECKSUM, download_dir)

    extract_tar(annotations_tar, data_dir)
    extract_tar(train_tar, data_dir)
    extract_tar(val_tar, data_dir)
    os.remove(os.path.join(data_dir, 'test_info.csv'))

    if args.remove_src:
        print('Removing tar files...')
        os.remove(annotations_tar)
        os.remove(train_tar)
        os.remove(val_tar)
