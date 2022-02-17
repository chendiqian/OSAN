import os
import copy
from typing import Tuple, Optional, List, Dict, Mapping
from tqdm import tqdm

import torch
from torch_geometric.datasets import TUDataset

from data.custom_dataloader import MYDataLoader
from data.subgraph_policy import policy2transform, Sampler
from data.custom_datasets import CustomTUDataset


def get_data(dataset: str,
             data_path: str,
             batch_size: int,
             policy: str,
             sample_mode: str,
             fraction: float,
             k: int,
             debug: bool)\
        -> Tuple[MYDataLoader, MYDataLoader, Optional[MYDataLoader]]:
    """

    :param dataset:
    :param data_path:
    :param batch_size:
    :param policy:
    :param sample_mode: only for baselines
    :param fraction: only for baselines
    :param k: only for baselines
    :param debug:
    :return:
    """

    if dataset.lower() == 'zinc':
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        pre_transform = policy2transform(policy)

        if pre_transform is None:
            dataset = TUDataset(data_path, name="ZINC_full")
        else:
            transform = Sampler(sample_mode, fraction, k)
            dataset = CustomTUDataset(data_path + '/deck', name="ZINC_full",
                                      transform=transform, pre_transform=pre_transform)

        # infile = open("./datasets/indices/test.index.txt", "r")
        # for line in infile:
        #     test_indices = line.split(",")
        #     if debug:
        #         test_indices = test_indices[:16]
        #     test_indices = [int(i) for i in test_indices]

        infile = open("./datasets/indices/val.index.txt", "r")
        for line in infile:
            val_indices = line.split(",")
            if debug:
                val_indices = val_indices[:16]
            val_indices = [int(i) for i in val_indices]

        infile = open("./datasets/indices/train.index.txt", "r")
        for line in infile:
            train_indices = line.split(",")
            if debug:
                train_indices = train_indices[:16]
            train_indices = [int(i) for i in train_indices]

        train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=batch_size, shuffle=True,
                                    subgraph_loader=pre_transform is not None)
        # test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=batch_size, shuffle=False,
        #                            subgraph_loader=pre_transform is not None)
        val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=batch_size, shuffle=False,
                                  subgraph_loader=pre_transform is not None)
    else:
        raise NotImplementedError

    return train_loader, val_loader, None
