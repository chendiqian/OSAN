from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, Dataset, HeteroData
from torch_geometric.utils import subgraph


def rand_sampling(graph: Data,
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
        batch_num_nodes += len(indices)

        res_list += subgraphs_from_index(graph, sort_indices)

    return res_list, batch_num_nodes


def subgraphs_from_index(graph: Data, indices: Tensor) -> List[Data]:
    """
    Given a graph and a tensor whose rows are indices (The indices are sorted)
    Returns a list of subgraphs indicated by the indices

    :param graph:
    :param indices:
    :return:
    """
    if indices.ndim < 2:
        indices = indices[None]

    graphs = []
    for idx in indices:
        edge_index, edge_attr = subgraph(idx, graph.edge_index, graph.edge_attr, relabel_nodes=True)
        graphs.append(Data(x=graph.x[idx],
                           edge_index=edge_index,
                           edge_attr=edge_attr,
                           y=graph.y))

    return graphs
