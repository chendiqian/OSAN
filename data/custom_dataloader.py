from typing import List, Optional, Union, Tuple
from functools import partial
import itertools

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch, Data, Dataset, HeteroData

from subgraph_utils import rand_sampling, construct_subgraph_batch


# class Collater:
#     def __init__(self, follow_batch, exclude_keys):
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys
#
#     def __call__(self, batch):
#         elem = batch[0]
#         if isinstance(elem, (Data, HeteroData)):
#             return Batch.from_data_list(batch, self.follow_batch,
#                                         self.exclude_keys)
#         elif isinstance(elem, torch.Tensor):
#             return default_collate(batch)
#         elif isinstance(elem, float):
#             return torch.tensor(batch, dtype=torch.float)
#         elif isinstance(elem, int):
#             return torch.tensor(batch)
#         elif isinstance(elem, str):
#             return batch
#         elif isinstance(elem, Mapping):
#             return {key: self([data[key] for data in batch]) for key in elem}
#         elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
#             return type(elem)(*(self(s) for s in zip(*batch)))
#         elif isinstance(elem, Sequence) and not isinstance(elem, str):
#             return [self(s) for s in zip(*batch)]
#
#         raise TypeError(f'DataLoader found invalid type: {type(elem)}')
#
#     def collate(self, batch):  # Deprecated...
#         return self(batch)


class SampleCollater:
    """
    Customized data collator
    Return batches with randomly sampled subgraphs
    e.g.
    Given a batch of graphs [g1, g2, g3 ...] from the dataset
    Returns augmented batch of [g1_1, g1_2, g1_3, g2_1, g2_2, g2_3, ...]

    Requires no pre_transform for the dataset

    Drawback: can get same subgraphs of an original graph

    """

    def __init__(self,
                 n_subgraphs: int = 0,
                 node_per_subgraph: int = -1,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None):
        self.n_subgraphs = n_subgraphs
        self.node_per_subgraph = node_per_subgraph
        assert follow_batch is None and exclude_keys is None, "Not supported"
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Data]):
        graph_list = []
        for i, g in enumerate(batch):
            subgraphs, _ = rand_sampling(g, self.n_subgraphs, self.node_per_subgraph)
            graph_list += subgraphs

        return construct_subgraph_batch(graph_list, [self.n_subgraphs] * len(batch), None, batch[0].x.device)


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
            sample: bool = False,
            n_subgraphs: int = 0,
            node_per_subgraph: int = -1,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        # Save for PyTorch Lightning:
        assert follow_batch is None and exclude_keys is None
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collate_fn = Collater
        if subgraph_loader:
            collate_fn = SubgraphSetCollator
        elif sample and n_subgraphs > 0:
            collate_fn = partial(SampleCollater, n_subgraphs=n_subgraphs, node_per_subgraph=node_per_subgraph)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn(follow_batch=follow_batch, exclude_keys=exclude_keys),
            **kwargs,
        )
