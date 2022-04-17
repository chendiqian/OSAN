import os
from typing import Tuple, Union
from argparse import Namespace
from ml_collections import ConfigDict

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from data.custom_dataloader import MYDataLoader
from data.data_utils import GraphToUndirected
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


def get_data(args: Union[Namespace, ConfigDict]) -> Tuple[MYDataLoader, MYDataLoader, MYDataLoader]:
    """

    :param args
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

    if args.normalize_label:
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std

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

    train_loader = MYDataLoader(train_set, batch_size=args.batch_size, shuffle=not args.debug,
                                subgraph_loader=sample_collator)
    test_loader = MYDataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                               subgraph_loader=sample_collator)
    val_loader = MYDataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              subgraph_loader=sample_collator)

    return train_loader, val_loader, test_loader
