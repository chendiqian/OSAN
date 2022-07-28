import os
from typing import Tuple, Union, List
from argparse import Namespace
from ml_collections import ConfigDict

import torch
import numpy as np
from torch import device as torchdevice
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.transforms import Compose, Distance
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold

from data.const import MAX_NUM_NODE_DICT
from data.planarsatpairsdataset import PlanarSATPairsDataset
from data.custom_dataloader import MYDataLoader
from data.data_utils import AttributedDataLoader
from data.data_preprocess import GraphToUndirected, GraphCoalesce, AugmentwithLineGraph, AugmentwithKhopList, AugmentwithAdj
from subgraph.subgraph_policy import policy2transform, RawNodeSampler, RawEdgeSampler, RawKhopSampler, \
    RawGreedyExpand, RawMSTSampler, RawKhopDualSampler

DATASET = (PlanarSATPairsDataset, TUDataset, QM9, PygGraphPropPredDataset)

TRANSFORM_DICT = {'node': RawNodeSampler,
                  'edge': RawEdgeSampler,
                  'khop_subgraph': RawKhopSampler,
                  'khop_dual': RawKhopDualSampler,
                  'greedy_exp': RawGreedyExpand,
                  'mst': RawMSTSampler, }

NAME_DICT = {'zinc': "ZINC_full",
             'mutag': "MUTAG",
             'alchemy': "alchemy_full"}


def get_pretransform(args: Union[Namespace, ConfigDict]):
    postfix = None
    if args.imle_configs is None and args.sample_configs.sample_with_esan:
        if args.sample_configs.sample_policy in ['node', 'edge']:
            assert args.sample_configs.sample_k == -1, "ESAN supports remove one substance only"
        pre_transform = policy2transform(args.sample_configs.sample_policy, relabel=args.sample_configs.remove_node)
    elif args.imle_configs is not None and args.sample_configs.sample_policy == 'edge_linegraph':
        pre_transform = AugmentwithLineGraph()
        postfix = 'linegraph'
    elif args.sample_configs.sample_policy is not None and 'khop' in args.sample_configs.sample_policy:
        pre_transform = AugmentwithKhopList(MAX_NUM_NODE_DICT[args.dataset.lower()], args.sample_configs.sample_k)
        postfix = f'khop{args.sample_configs.sample_k}'
    elif args.imle_configs is not None and args.sample_configs.sample_policy == 'node_ordered':
        pre_transform = AugmentwithAdj()
        postfix = 'withAdj'
    else:
        pre_transform = lambda x: x  # no deck
    return pre_transform, postfix


def get_transform(args: Union[Namespace, ConfigDict]):
    transform = None
    sample_collator = False

    if args.imle_configs is None:
        if args.sample_configs.sample_with_esan:
            # sample_collator = True
            # transform = DeckSampler(args.esan_configs, add_full_graph=args.add_full_graph)
            # dataset_func = CustomTUDataset
            # data_path += f'/deck/{args.esan_configs.esan_policy}'
            raise NotImplementedError
        else:
            if args.sample_configs.num_subgraphs > 0:  # sample-on-the-fly
                sample_collator = True
                transform = TRANSFORM_DICT[args.sample_configs.sample_policy](args.sample_configs.num_subgraphs,
                                                                              args.sample_configs.sample_k,
                                                                              args.sample_configs.remove_node,
                                                                              args.sample_configs.add_full_graph)

    return transform, sample_collator


def get_data(args: Union[Namespace, ConfigDict], device: torchdevice) -> Tuple[List[AttributedDataLoader],
                                                                               List[AttributedDataLoader],
                                                                               List[AttributedDataLoader]]:
    """
    Distributor function

    :param args:
    :param device:
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    if 'ogb' in args.dataset.lower():
        train_set, val_set, test_set, mean, std, sample_collator = get_ogb_data(args)
    elif args.dataset.lower() == 'qm9':
        train_set, val_set, test_set, mean, std, sample_collator = get_qm9(args, device)
    elif args.dataset.lower() in ['exp', 'cexp']:
        train_set, val_set, test_set, mean, std, sample_collator = get_synthdata(args)
    elif args.dataset.lower() in ['zinc', 'alchemy']:
        train_set, val_set, test_set, mean, std, sample_collator = get_TUdata(args, device)
    else:
        raise ValueError

    if isinstance(train_set, list):
        train_loaders = [AttributedDataLoader(
            loader=MYDataLoader(t,
                                batch_size=args.batch_size,
                                shuffle=not args.debug,
                                subgraph_loader=sample_collator),
            mean=mean,
            std=std) for t in train_set]
    elif isinstance(train_set, DATASET):
        train_loaders = [AttributedDataLoader(
            loader=MYDataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=not args.debug,
                                subgraph_loader=sample_collator),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(val_set, list):
        val_loaders = [AttributedDataLoader(
            loader=MYDataLoader(t,
                                batch_size=args.batch_size,
                                shuffle=False,
                                subgraph_loader=sample_collator),
            mean=mean,
            std=std) for t in val_set]
    elif isinstance(val_set, DATASET):
        val_loaders = [AttributedDataLoader(
            loader=MYDataLoader(val_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                subgraph_loader=sample_collator),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    if isinstance(test_set, list):
        test_loaders = [AttributedDataLoader(
            loader=MYDataLoader(t,
                                batch_size=args.batch_size,
                                shuffle=False,
                                subgraph_loader=sample_collator),
            mean=mean,
            std=std) for t in test_set]
    elif isinstance(test_set, DATASET):
        test_loaders = [AttributedDataLoader(
            loader=MYDataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                subgraph_loader=sample_collator),
            mean=mean,
            std=std)]
    else:
        raise TypeError

    return train_loaders, val_loaders, test_loaders


def get_TUdata(args: Union[Namespace, ConfigDict], device: torchdevice):
    """

    :param args
    :param device
    :return:
    """
    infile = open(f"./datasets/indices/{args.dataset.lower()}/test.index.txt", "r")
    line = next(iter(infile))
    test_indices = line.split(",")
    if args.debug:
        test_indices = test_indices[:16]
    test_indices = [int(i) for i in test_indices]

    infile = open(f"./datasets/indices/{args.dataset.lower()}/val.index.txt", "r")
    line = next(iter(infile))
    val_indices = line.split(",")
    if args.debug:
        val_indices = val_indices[:16]
    val_indices = [int(i) for i in val_indices]

    infile = open(f"./datasets/indices/{args.dataset.lower()}/train.index.txt", "r")
    line = next(iter(infile))
    train_indices = line.split(",")
    if args.debug:
        train_indices = train_indices[:16]
    train_indices = [int(i) for i in train_indices]

    pre_transform, postfix = get_pretransform(args)

    if args.dataset.lower() == 'zinc':
        # for my version of PyG, ZINC is directed
        pre_transform = Compose([GraphToUndirected(), pre_transform])
    elif args.dataset.lower() in ['alchemy']:
        pass
    else:
        raise NotImplementedError

    dataset_func = TUDataset
    data_path = args.data_path
    if postfix is not None:
        data_path = os.path.join(data_path, postfix)

    transform, sample_collator = get_transform(args)

    dataset = dataset_func(data_path,
                           name=NAME_DICT[args.dataset.lower()],
                           transform=transform,
                           pre_transform=pre_transform)

    if dataset.data.y.ndim == 1:
        dataset.data.y = dataset.data.y[:, None]

    if args.normalize_label:
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean, std = None, None

    if args.dataset.lower() == 'zinc':
        train_set = dataset[:220011][train_indices]
        val_set = dataset[225011:][val_indices]
        test_set = dataset[220011:225011][test_indices]
    else:
        train_set = dataset[train_indices]
        val_set = dataset[val_indices]
        test_set = dataset[test_indices]

    return train_set, val_set, test_set, mean, std, sample_collator


def get_ogb_data(args: Union[Namespace, ConfigDict]):
    pre_transform, postfix = get_pretransform(args)
    pre_transform = Compose([GraphCoalesce(), pre_transform])

    data_path = args.data_path
    if postfix is not None:
        data_path = os.path.join(data_path, postfix)
    transform, sample_collator = get_transform(args)

    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      root=data_path,
                                      transform=transform,
                                      pre_transform=pre_transform)
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"] if not args.debug else split_idx["train"][:16]
    val_idx = split_idx["valid"] if not args.debug else split_idx["valid"][:16]
    test_idx = split_idx["test"] if not args.debug else split_idx["test"][:16]

    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]

    return train_set, val_set, test_set, None, None, sample_collator


def get_qm9(args, device):
    np.random.seed(42)
    idx = np.random.permutation(130831)
    train_indices = idx[:10000]
    val_indices = idx[10000:11000]
    test_indices = idx[11000:12000]

    if args.debug:
        train_indices = train_indices[:16]
        val_indices = val_indices[:16]
        test_indices = test_indices[:16]

    pre_transform, postfix = get_pretransform(args)
    pre_transform = Compose([Distance(norm=False), pre_transform])

    data_path = os.path.join(args.data_path, 'QM9')
    if postfix is not None:
        data_path = os.path.join(data_path, postfix)
    transform, sample_collator = get_transform(args)

    dataset = QM9(data_path,
                  transform=transform,
                  pre_transform=pre_transform)
    dataset.data.y = dataset.data.y[:, 0:12]

    train_set = dataset[train_indices]
    val_set = dataset[val_indices]
    test_set = dataset[test_indices]

    if args.normalize_label:
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean, std = None, None

    return train_set, val_set, test_set, mean, std, sample_collator


def get_synthdata(args):
    pre_transform, postfix = get_pretransform(args)
    pre_transform = Compose([GraphToUndirected(), GraphCoalesce(), pre_transform])

    data_path = os.path.join(args.data_path, args.dataset.upper())
    if postfix is not None:
        data_path = os.path.join(data_path, postfix)
    transform, sample_collator = get_transform(args)

    dataset = PlanarSATPairsDataset(data_path,
                                    transform=transform,
                                    pre_transform=pre_transform)
    dataset.data.y = dataset.data.y[:, None]

    def separate_data(fold_idx):
        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test
        assert 0 <= fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        labels = dataset.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return torch.tensor(train_idx), torch.tensor(test_idx), torch.tensor(test_idx)

    train_sets, val_sets, test_sets = [], [], []
    for idx in range(10):
        train, val, test = separate_data(idx)
        train_set = dataset[train]
        val_set = dataset[val]
        test_set = dataset[test]

        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)

    return train_sets, val_sets, test_sets, None, None, sample_collator
