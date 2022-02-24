from typing import List, Optional, Union, Tuple
from functools import partial
import itertools

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch, Data, Dataset, HeteroData

from subgraph_utils import construct_subgraph_batch


class SubgraphSetCollator:
    """
    Given subgraphs [[g1_1, g1_2, g1_3], [g2_1, g2_2, g2_3], ...]
    Collate them as a batch
    """

    def __init__(self,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None):
        assert follow_batch is None and exclude_keys is None, "Not supported"
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch_list: List[List[Data]]):
        list_subgraphs = list(itertools.chain.from_iterable(batch_list))

        return construct_subgraph_batch(list_subgraphs,
                                        [len(g_list) for g_list in batch_list],
                                        None,
                                        batch_list[0][0].x.device)


class MYDataLoader(torch.utils.data.DataLoader):
    """
    Customed data loader modified from data loader of PyG.
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
    Enable sampling subgraphs of the original graph, and merge the sampled graphs in a batch
    batch_size = batch_size * n_subgraphs

    :param dataset:
    :param batch_size:
    :param shuffle:
    :param n_subgraphs:
    :param subgraph_loader:
    :param follow_batch:
    :param exclude_keys:
    :param kwargs:
    """

    def __init__(
            self,
            dataset: Union[Dataset, List[Data], List[HeteroData]],
            batch_size: int = 1,
            shuffle: bool = False,
            subgraph_loader: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        # Save for PyTorch Lightning:
        assert follow_batch is None and exclude_keys is None
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collate_fn = SubgraphSetCollator if subgraph_loader else Collater

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn(follow_batch=follow_batch, exclude_keys=exclude_keys),
            **kwargs,
        )
