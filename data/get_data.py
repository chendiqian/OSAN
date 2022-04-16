import os
from typing import Tuple, Optional
from argparse import Namespace

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from data.custom_dataloader import MYDataLoader
from data.custom_datasets import CustomTUDataset
from data.data_utils import GraphToUndirected
from subgraph.subgraph_policy import policy2transform, DeckSampler, RawNodeSampler, RawEdgeSampler, RawKhopSampler, \
    RawGreedyExpand, RawMSTSampler


TRANSFORM_DICT = {'node': RawNodeSampler,
                  'edge': RawEdgeSampler,
                  'khop_subgraph': RawKhopSampler,
                  'greedy_exp': RawGreedyExpand,
                  'mst': RawMSTSampler, }

NAME_DICT = {'zinc': "ZINC_full",
             'mutag': "MUTAG",
             'alchemy': "alchemy_full"}


def get_data(args: Namespace) -> Tuple[MYDataLoader, MYDataLoader, Optional[MYDataLoader]]:
    """

    :param args
    :return:
    """
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

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

    if args.dataset.lower() == 'zinc':
        # for my version of PyG, ZINC is directed
        pre_transform = Compose([GraphToUndirected(),
                                 policy2transform(args.esan_policy,
                                                  relabel=args.remove_node)])
    elif args.dataset.lower() in ['alchemy']:
        pre_transform = policy2transform(args.esan_policy,
                                         relabel=args.remove_node)
    else:
        raise NotImplementedError

    if args.esan_policy == 'null':  # I-MLE, or normal training, or sample on the fly
        transform = None
        if (not args.train_embd_model) and (args.num_subgraphs > 0):  # sample-on-the-fly
            transform = TRANSFORM_DICT[args.sample_policy](args.num_subgraphs,
                                                           args.sample_k,
                                                           args.remove_node,
                                                           args.add_full_graph)
        dataset = TUDataset(args.data_path, transform=transform, name=NAME_DICT[args.dataset.lower()],
                            pre_transform=pre_transform)
    else:  # ESAN: sample from the deck
        transform = DeckSampler(args.sample_mode, args.esan_frac, args.esan_k, add_full_graph=args.add_full_graph)
        dataset = CustomTUDataset(args.data_path + f'/deck/{args.esan_policy}', name=NAME_DICT[args.dataset.lower()],
                                  transform=transform, pre_transform=pre_transform)

    # use subgraph collator when sample from deck or a graph
    # either case the batch will be [[g11, g12, g13], [g21, g22, g23], ...]
    sample_collator = (args.esan_policy != 'null') or ((not args.train_embd_model) and (args.num_subgraphs > 0))

    if args.dataset.lower() == 'zinc':
        train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=args.batch_size, shuffle=not args.debug,
                                    subgraph_loader=sample_collator)
        test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=args.batch_size, shuffle=False,
                                   subgraph_loader=pre_transform is not None)
        val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=args.batch_size, shuffle=False,
                                  subgraph_loader=sample_collator)
    else:
        train_loader = MYDataLoader(dataset[train_indices], batch_size=args.batch_size, shuffle=not args.debug,
                                    subgraph_loader=sample_collator)
        test_loader = MYDataLoader(dataset[test_indices], batch_size=args.batch_size, shuffle=False,
                                   subgraph_loader=pre_transform is not None)
        val_loader = MYDataLoader(dataset[val_indices], batch_size=args.batch_size, shuffle=False,
                                  subgraph_loader=sample_collator)

    return train_loader, val_loader, test_loader
