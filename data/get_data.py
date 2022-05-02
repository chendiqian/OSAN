import os
from typing import Tuple, Union
from argparse import Namespace
from ml_collections import ConfigDict

import numpy as np
from torch import device as torchdevice
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.transforms import Compose, Distance
from ogb.graphproppred import PygGraphPropPredDataset

from data.custom_dataloader import MYDataLoader
from data.data_utils import GraphToUndirected, GraphCoalesce, AttributedDataLoader
from subgraph.subgraph_policy import policy2transform, RawNodeSampler, RawEdgeSampler, RawKhopSampler, \
    RawGreedyExpand, RawMSTSampler

TRANSFORM_DICT = {'node': RawNodeSampler,
                  'edge': RawEdgeSampler,
                  'khop_subgraph': RawKhopSampler,
                  'greedy_exp': RawGreedyExpand,
                  'mst': RawMSTSampler, }

NAME_DICT = {'zinc': "ZINC_full",
             'mutag': "MUTAG",
             'alchemy': "alchemy_full"}


def get_data(args: Union[Namespace, ConfigDict], device: torchdevice) -> Tuple[AttributedDataLoader,
                                                                               AttributedDataLoader,
                                                                               AttributedDataLoader]:
    """

    :param args
    :param device
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    # ============================================================================
    # Get reference indices in each split
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

    # =============================================================================
    # get pre_transform: to_directed + create ESAN deck
    if args.imle_configs is None and args.sample_configs.sample_with_esan:
        if args.sample_configs.sample_policy in ['node', 'edge']:
            assert args.sample_configs.sample_k == -1, "ESAN supports remove one substance only"
        pre_transform = policy2transform(args.sample_configs.sample_policy, relabel=args.sample_configs.remove_node)
    else:
        pre_transform = lambda x: x  # no deck

    if args.dataset.lower() == 'zinc':
        # for my version of PyG, ZINC is directed
        pre_transform = Compose([GraphToUndirected(), pre_transform])
    elif args.dataset.lower() in ['alchemy']:
        pass
    else:
        raise NotImplementedError

    # ==============================================================================
    # get transform: ESAN -> sample from deck; IMLE or normal -> None; On the fly -> customed function
    transform = None
    dataset_func = TUDataset
    data_path = args.data_path
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

    # ==============================================================================
    # get dataset
    dataset = dataset_func(data_path,
                           name=NAME_DICT[args.dataset.lower()],
                           transform=transform,
                           pre_transform=pre_transform)

    # ==============================================================================
    # get split
    if args.dataset.lower() == 'zinc':
        train_set = dataset[:220011][train_indices]
        val_set = dataset[225011:][val_indices]
        test_set = dataset[220011:225011][test_indices]
    else:
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

    train_loader = AttributedDataLoader(
        loader=MYDataLoader(train_set,
                            batch_size=args.batch_size,
                            shuffle=not args.debug,
                            subgraph_loader=sample_collator),
        mean=mean,
        std=std)
    test_loader = AttributedDataLoader(
        loader=MYDataLoader(test_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            subgraph_loader=sample_collator),
        mean=mean,
        std=std)
    val_loader = AttributedDataLoader(
        loader=MYDataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            subgraph_loader=sample_collator),
        mean=mean,
        std=std)

    return train_loader, val_loader, test_loader


def get_ogb_data(args: Union[Namespace, ConfigDict]) -> Tuple[AttributedDataLoader,
                                                              AttributedDataLoader,
                                                              AttributedDataLoader,
                                                              int]:
    if args.imle_configs is None and args.sample_configs.sample_with_esan:
        if args.sample_configs.sample_policy in ['node', 'edge']:
            assert args.sample_configs.sample_k == -1, "ESAN supports remove one substance only"
        pre_transform = policy2transform(args.sample_configs.sample_policy, relabel=args.sample_configs.remove_node)
    else:
        pre_transform = lambda x: x  # no deck

    pre_transform = Compose([GraphCoalesce(), pre_transform])

    transform = None
    data_path = args.data_path
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

    dataset = PygGraphPropPredDataset(name=args.dataset,
                                      root=data_path,
                                      transform=transform,
                                      pre_transform=pre_transform)
    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"] if not args.debug else split_idx["train"][:16]
    val_idx = split_idx["valid"] if not args.debug else split_idx["valid"][:16]
    test_idx = split_idx["test"] if not args.debug else split_idx["test"][:16]

    train_loader = AttributedDataLoader(
        loader=MYDataLoader(dataset[train_idx],
                            batch_size=args.batch_size,
                            shuffle=not args.debug,
                            subgraph_loader=sample_collator),
        mean=None,
        std=None)
    test_loader = AttributedDataLoader(
        loader=MYDataLoader(dataset[test_idx],
                            batch_size=args.batch_size,
                            shuffle=False,
                            subgraph_loader=sample_collator),
        mean=None,
        std=None)
    val_loader = AttributedDataLoader(
        loader=MYDataLoader(dataset[val_idx],
                            batch_size=args.batch_size,
                            shuffle=False,
                            subgraph_loader=sample_collator),
        mean=None,
        std=None)

    return train_loader, val_loader, test_loader, dataset.num_tasks


def get_qm9(args, device):
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    np.random.seed(42)
    idx = np.random.permutation(130831)
    train_indices = idx[:10000]
    val_indices = idx[10000:11000]
    test_indices = idx[11000:12000]

    if args.debug:
        train_indices = train_indices[:16]
        val_indices = val_indices[:16]
        test_indices = test_indices[:16]

    # =============================================================================
    # get pre_transform: to_directed + create ESAN deck
    if args.imle_configs is None and args.sample_configs.sample_with_esan:
        if args.sample_configs.sample_policy in ['node', 'edge']:
            assert args.sample_configs.sample_k == -1, "ESAN supports remove one substance only"
        pre_transform = policy2transform(args.sample_configs.sample_policy, relabel=args.sample_configs.remove_node)
    else:
        pre_transform = lambda x: x  # no deck

    pre_transform = Compose([Distance(norm=False), pre_transform])

    # ==============================================================================
    # get transform: ESAN -> sample from deck; IMLE or normal -> None; On the fly -> customed function
    transform = None
    data_path = args.data_path
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

    # ==============================================================================
    # get dataset
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

    train_loader = AttributedDataLoader(
        loader=MYDataLoader(train_set,
                            batch_size=args.batch_size,
                            shuffle=not args.debug,
                            subgraph_loader=sample_collator),
        mean=mean,
        std=std)
    test_loader = AttributedDataLoader(
        loader=MYDataLoader(test_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            subgraph_loader=sample_collator),
        mean=mean,
        std=std)
    val_loader = AttributedDataLoader(
        loader=MYDataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            subgraph_loader=sample_collator),
        mean=mean,
        std=std)

    return train_loader, val_loader, test_loader
