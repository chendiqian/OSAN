from typing import List, Optional, Union, Tuple
from functools import partial
import itertools

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch, Data, Dataset, HeteroData

from subgraph_utils import rand_sampling
from data import SubgraphSetBatch


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
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None):
        self.n_subgraphs = n_subgraphs
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Data]):
        # TODO: pass number of n_drop_nodes into args

        graph_list = []
        inter_graph_idx = []
        for i, g in enumerate(batch):
            subgraphs, _ = rand_sampling(g, self.n_subgraphs)
            graph_list += subgraphs
            inter_graph_idx += torch.full((self.n_subgraphs,), i)

        assert isinstance(graph_list[0], (Data, HeteroData)) and isinstance(inter_graph_idx[0], Tensor)

        res_data = Batch.from_data_list(graph_list, self.follow_batch, self.exclude_keys)
        inter_graph_idx = default_collate(inter_graph_idx)
        res_data.inter_graph_idx = inter_graph_idx
        inter_graph_idx_aux = torch.cat((torch.tensor([-1]), inter_graph_idx), dim=0)
        ptr = (inter_graph_idx_aux[1:] > inter_graph_idx_aux[:-1]).nonzero().reshape(-1)

        # TODO: duplicate labels or aggregate the embeddings for original labels? potential problem: cannot split the
        #  batch because y shape inconsistent:
        #  https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Batch.to_data_list.
        #  need to check `batch._slice_dict` and `batch._inc_dict`
        res_data.y = res_data.y[ptr]

        return res_data


class SubgraphSetCollator:
    """
    Given subgraphs [[g1_1, g1_2, g1_3], [g2_1, g2_2, g2_3], ...]
    Collate them as a batch
    """

    def __init__(self,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch_list: List[List[Data]]):
        list_subgraphs = list(itertools.chain.from_iterable(batch_list))
        batch = Batch.from_data_list(list_subgraphs, self.follow_batch, self.exclude_keys)

        original_graph_mask = torch.cat([torch.full((len(g_list),), i, device=batch.x.device)
                                         for i, g_list in enumerate(batch_list)], dim=0)
        ptr = torch.cat((torch.tensor([0], device=batch.x.device),
                         (original_graph_mask[1:] > original_graph_mask[:-1]).nonzero().reshape(-1) + 1,
                         torch.tensor([len(original_graph_mask)], device=batch.x.device)), dim=0)

        return SubgraphSetBatch(x=batch.x,
                                # flip the direction of message
                                edge_index=batch.edge_index[torch.LongTensor([1, 0]), :],
                                edge_attr=batch.edge_attr,
                                edge_weight=None,
                                y=batch.y[ptr[:-1]],
                                batch=batch.batch,
                                inter_graph_idx=original_graph_mask,
                                ptr=ptr,
                                num_graphs=batch.num_graphs)


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
            n_subgraphs: int = 0,
            subgraph_loader: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        # Save for PyTorch Lightning:
        assert follow_batch is None and exclude_keys is None
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        # collate_fn = partial(SampleCollater, n_subgraphs=n_subgraphs) if n_subgraphs > 0 else Collater
        collate_fn = SubgraphSetCollator if subgraph_loader else Collater

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn(follow_batch=follow_batch, exclude_keys=exclude_keys),
            **kwargs,
        )
