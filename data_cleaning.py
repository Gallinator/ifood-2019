import os.path
import random

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchinfo
import tqdm
from pyod.models.iforest import IForest
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from transformers import AutoModelForImageClassification

from food_dataset import FoodDataset
from transforms import NORM_MEAN, NORM_STD
from visualization import plot_image

SWIN_TRANSFORMS = v2.Compose([v2.Resize(224),
                              v2.CenterCrop(224),
                              v2.ToTensor(),
                              v2.Normalize(NORM_MEAN, NORM_STD)])


def get_embeddings_path(data_path: str) -> str:
    """
    Extract the embeddings file path from the data split name
    :param data_path: data folder
    :return: the embeddings file path
    """
    root, split = os.path.split(data_path)
    return os.path.join(root, f'{split}_embeddings.csv')


def get_embeddings(data_path: str) -> str:
    """
    Extract the embeddings using a pretrained Swin transformer and saves them to a csv.
    :param data_path: path to the images to extract the embeddings of
    :return: the path to the file containing the embeddings
    """
    embeddings_path = get_embeddings_path(data_path)
    if os.path.exists(embeddings_path):
        print('Using existing embeddings...')
        return embeddings_path

    model = AutoModelForImageClassification.from_pretrained("aspis/swin-finetuned-food101")
    model.eval()
    model.classifier = nn.Sequential(nn.Identity())
    torchinfo.summary(model, input_size=(1, 3, 224, 224))
    model.to('cuda')

    data = FoodDataset(data_path, transform=SWIN_TRANSFORMS)
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=8, persistent_workers=True)

    preds = []

    for batch in tqdm.tqdm(loader):
        x, label = batch
        y = model.forward(x.to('cuda')).logits
        y = y.to('cpu').detach()
        preds.append(y)
        del x
        torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=0).numpy(force=True)
    out_df = pd.DataFrame.from_records(preds, columns=[f'Dim{i}' for i in range(preds.shape[1])])
    meta_df = pd.DataFrame(data.samples, columns=['Path', 'Label']).astype({'Path': str})
    out_df = pd.concat([out_df, meta_df], axis=1)
    out_df.to_csv(embeddings_path)
    return embeddings_path


def find_anomalies(reprs_path: str, image_delete_path: str) -> list[str]:
    """
    Remove the anomalies from the data. This process is destructive as it deletes images. Also saves a file containing the deleted images.
    :param reprs_path: path to the extracted embeddings
    :param image_delete_path: csv path to save the list of deleted images
    :return: a list with the paths of deleted images
    """
    df = pd.read_csv(reprs_path, index_col=0)
    data = df.iloc[:, :1024].to_numpy()

    data = MinMaxScaler().fit_transform(data)

    iforest = IForest(1000, n_jobs=-1, max_samples=1.0, random_state=8421, verbose=1).fit(data)
    thr = np.mean(iforest.decision_scores_) - 3 * np.std(iforest.decision_scores_)

    plt.scatter(np.arange(len(data)), iforest.decision_scores_, s=7)
    plt.axhline(thr, color='red', linestyle='--')
    plt.show()

    outliers = iforest.decision_scores_ < thr
    outliers = df['Path'][outliers]

    print(f'Found {len(outliers)} outliers!')

    # Show 20 examples
    show_idx = random.sample(range(len(outliers)), 20)
    for i, path in enumerate(outliers.tolist()):
        if i in show_idx:
            img = PIL.Image.open(path)
            img_label = os.path.split(os.path.split(path)[0])[1]
            plot_image(img, img_label)

    outliers.to_csv(image_delete_path)
    return outliers


def clean_data(data_dir: str):
    """
    Execute the data cleaning process
    :param data_dir: data to clean
    """
    embeddings_path = get_embeddings(data_dir)
    for img in find_anomalies(embeddings_path, 'data/deleted_images.csv'):
        os.remove(img)
