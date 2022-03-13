import os
from typing import Tuple, Optional
from argparse import Namespace

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from data.custom_dataloader import MYDataLoader
from data.custom_datasets import CustomTUDataset
from data.data_utils import GraphToUndirected
from subgraph.subgraph_policy import policy2transform, DeckSampler, RawNodeSampler, RawEdgeSampler, RawKhopSampler


def get_data(args: Namespace) -> Tuple[MYDataLoader, MYDataLoader, Optional[MYDataLoader]]:
    """

    :param args
    :return:
    """

    if args.dataset.lower() == 'zinc':
        if not os.path.isdir(args.data_path):
            os.mkdir(args.data_path)

        pre_transform = Compose([GraphToUndirected(), policy2transform(args.esan_policy)])

        if args.esan_policy == 'null':   # I-MLE, or normal training, or sample on the fly
            transform = None
            if (not args.train_embd_model) and (args.num_subgraphs > 0):   # sample-on-the-fly
                if args.sample_policy == 'node':
                    transform = RawNodeSampler(args.num_subgraphs, args.sample_node_k)
                elif args.sample_policy == 'edge':
                    transform = RawEdgeSampler(args.num_subgraphs, args.sample_edge_k)
                elif args.sample_policy == 'khop_subgraph':
                    transform = RawKhopSampler(args.num_subgraphs, args.khop, args.prune_policy)
                else:
                    raise NotImplementedError(f"Not support {args.sample_policy} for sample on the fly.")
            dataset = TUDataset(args.data_path, transform=transform, name="ZINC_full", pre_transform=pre_transform)
        else:   # ESAN: sample from the deck
            transform = DeckSampler(args.sample_mode, args.esan_frac, args.esan_k)
            dataset = CustomTUDataset(args.data_path + f'/deck/{args.esan_policy}', name="ZINC_full",
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
            if args.debug:
                val_indices = val_indices[:16]
            val_indices = [int(i) for i in val_indices]

        infile = open("./datasets/indices/train.index.txt", "r")
        for line in infile:
            train_indices = line.split(",")
            if args.debug:
                train_indices = train_indices[:16]
            train_indices = [int(i) for i in train_indices]

        # use subgraph collator when sample from deck or a graph
        # either case the batch will be [[g11, g12, g13], [g21, g22, g23], ...]
        sample_collator = (args.esan_policy != 'null') or ((not args.train_embd_model) and (args.num_subgraphs > 0))

        train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=args.batch_size, shuffle=True,
                                    subgraph_loader=sample_collator)
        # test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=batch_size, shuffle=False,
        #                            subgraph_loader=pre_transform is not None)
        val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=args.batch_size, shuffle=False,
                                  subgraph_loader=sample_collator)
    else:
        raise NotImplementedError

    return train_loader, val_loader, None
