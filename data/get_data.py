import os
from typing import Tuple, Optional, List, Dict, Mapping
from argparse import Namespace

from torch_geometric.datasets import TUDataset

from data.custom_dataloader import MYDataLoader
from data.subgraph_policy import policy2transform, Sampler
from data.custom_datasets import CustomTUDataset


def get_data(args: Namespace) -> Tuple[MYDataLoader, MYDataLoader, Optional[MYDataLoader]]:
    """

    :param args
    :return:
    """

    if args.dataset.lower() == 'zinc':
        if not os.path.isdir(args.data_path):
            os.mkdir(args.data_path)

        pre_transform = policy2transform(args.policy)

        if pre_transform is None:
            dataset = TUDataset(args.data_path, name="ZINC_full")
        else:
            transform = Sampler(args.sample_mode, args.esan_frac, args.esan_k)
            dataset = CustomTUDataset(args.data_path + '/deck', name="ZINC_full",
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

        train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=args.batch_size, shuffle=True,
                                    subgraph_loader=pre_transform is not None,
                                    sample=not args.train_embd_model,
                                    n_subgraphs=args.num_subgraphs,
                                    node_per_subgraph=args.sample_k)
        # test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=batch_size, shuffle=False,
        #                            subgraph_loader=pre_transform is not None)
        val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=args.batch_size, shuffle=False,
                                  subgraph_loader=pre_transform is not None,
                                  sample=not args.train_embd_model,
                                  n_subgraphs=args.num_subgraphs,
                                  node_per_subgraph=args.sample_k)
    else:
        raise NotImplementedError

    return train_loader, val_loader, None
