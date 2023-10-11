import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle


from data.dataset import CustomImageDataset
from data.transforms import data_transforms
from src.utils.train_utils import load_object


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_ohe_tags(dataframes: dict[str, pd.DataFrame], config):
    encoder = load_object(config.encoder)(**config.encoder_kwargs)

    ohe_tags_subset_train = encoder.fit_transform(dataframes['train'].list_tags.values)
    ohe_tags_subset_val = encoder.transform(dataframes['val'].list_tags.values)

    # save pickle to be able to access the encoder during inference
    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    return {'train': ohe_tags_subset_train, 'val': ohe_tags_subset_val}


def load_data(config) -> dict[str, torch.utils.data.DataLoader]:
    train_classes_path = config.data_dir + '/planet/planet/train_classes.csv'
    data_dir = config.data_dir + '/planet/planet/train-jpg'
    df_class = pd.read_csv(train_classes_path)
    df_class["list_tags"] = df_class.tags.str.split(" ")

    df_train, df_val = train_test_split(df_class, test_size=config.test_size)
    dataframes = {"train": df_train, "val": df_val}
    ohe_tags = build_ohe_tags(dataframes, config)

    g = torch.Generator()
    g.manual_seed(config.seed)

    dataloaders = {}
    for subset in dataframes:
        dataset = CustomImageDataset(
            dataframes[subset]['image_name'].to_numpy(),
            path=data_dir,
            ohe_encoder=ohe_tags[subset],
            transform=data_transforms[subset],
        )
        dataloaders[subset] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=subset == "train",
            num_workers=2,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True,
            generator=g,
        )
    return dataloaders
