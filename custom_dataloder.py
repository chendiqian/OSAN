from typing import List, Optional, Union, Tuple
from functools import partial

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch, Data, Dataset, HeteroData
from torch_geometric.utils import subgraph


@torch.no_grad()
def subgraph_sampling(graph: Data,
                      n_subgraphs: int,
                      n_drop_nodes: int = 1) -> Tuple[List[Data], int]:
    """
    Sample subgraphs.
    TODO: replace for-loop with functorch.vmap https://pytorch.org/tutorials/prototype/vmap_recipe.html?highlight=vmap


    :param graph:
    :param n_subgraphs:
    :param n_drop_nodes:
    :return:
        A list of graphs and their index masks
    """

    n_nodes = graph.x.shape[0]
    res_list = []
    batch_num_nodes = 0
    for i in range(n_subgraphs):
        indices = torch.randperm(n_nodes)[:-n_drop_nodes]
        sort_indices = torch.sort(indices).values
        batch_num_nodes += indices
        edge_index, edge_attr = subgraph(sort_indices, graph.edge_index, graph.edge_attr, relabel_nodes=True)
        res_list.append(
            Data(
                x=graph.x[sort_indices, :],
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=graph.y
            )
        )

    return res_list, batch_num_nodes


class MyCollater:
    def __init__(self,
                 n_subgraphs: int = 0,
                 follow_batch: Optional[List[str]] = None,
                 exclude_keys: Optional[List[str]] = None):
        self.n_subgraphs = n_subgraphs
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self,
                 batch: List[Data]):

        # TODO: pass number of n_drop_nodes into args

        graph_list = []
        inter_graph_idx = []
        for i, g in enumerate(batch):
            subgraphs, _ = subgraph_sampling(g, self.n_subgraphs)
            graph_list += subgraphs
            inter_graph_idx += torch.full((self.n_subgraphs,), i)

        assert isinstance(graph_list[0], (Data, HeteroData)) and isinstance(inter_graph_idx[0], Tensor)

        res_data = Batch.from_data_list(graph_list, self.follow_batch, self.exclude_keys)
        inter_graph_idx = default_collate(inter_graph_idx)
        res_data.inter_graph_idx = inter_graph_idx
        inter_graph_idx_aux = torch.cat((torch.tensor([-1]), inter_graph_idx), dim=0)
        ptr = (inter_graph_idx_aux[1:] > inter_graph_idx_aux[:-1]).nonzero().reshape(-1)
        res_data.inter_graph_ptr = ptr

        # TODO: duplicate labels or aggregate the embeddings for original labels?
        res_data.y = res_data.y[ptr]

        return res_data


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
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        # Save for PyTorch Lightning:
        assert follow_batch is None and exclude_keys is None
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collate_fn = partial(MyCollater, n_subgraphs=n_subgraphs) if n_subgraphs > 0 else Collater

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn(follow_batch=follow_batch, exclude_keys=exclude_keys),
            **kwargs,
        )
