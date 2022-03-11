from typing import List, Optional, Tuple, Sequence
import random

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected, k_hop_subgraph

from subgraph.construct import nodesubset_to_subgraph


def node_rand_sampling(graph: Data,
                       n_subgraphs: int,
                       node_per_subgraph: int = -1) -> List[Data]:
    """
    Sample subgraphs.
    TODO: replace for-loop with functorch.vmap https://pytorch.org/tutorials/prototype/vmap_recipe.html?highlight=vmap


    :param graph:
    :param n_subgraphs:
    :param node_per_subgraph:
    :return:
        A list of graphs and their index masks
    """
    n_nodes = graph.num_nodes

    if node_per_subgraph < 0:  # drop nodes
        node_per_subgraph += n_nodes

    graphs = []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_nodes)[:node_per_subgraph]
        sort_indices = torch.sort(indices).values
        graphs.append(nodesubset_to_subgraph(graph, sort_indices, relabel=False))

    return graphs


def edge_rand_sampling(graph: Data, n_subgraphs: int, edge_per_subgraph: int = -1) -> List[Data]:
    """

    :param graph:
    :param n_subgraphs:
    :param edge_per_subgraph:
    :return:
    """
    n_edge = graph.num_edges
    if n_edge == 0:
        return [graph for _ in range(n_subgraphs)]

    if edge_per_subgraph < 0:  # drop edges
        edge_per_subgraph += n_edge

    edge_index, edge_attr, undirected = edge_sample_preproc(graph)

    graphs = []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_edge)[:edge_per_subgraph]
        sort_indices = torch.sort(indices).values

        subgraph_edge_index = edge_index[:, sort_indices]
        subgraph_edge_attr = edge_attr[sort_indices, :] if edge_attr is not None else None

        if undirected:
            if subgraph_edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                        num_nodes=graph.num_nodes)
            else:
                subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                    num_nodes=graph.num_nodes)

        graphs.append(Data(
            x=graph.x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            num_nodes=graph.num_nodes,
            y=graph.y,
        ))

    return graphs


def edge_sample_preproc(data: Data) -> Tuple[LongTensor, Tensor, bool]:
    """
    If undirected, return the non-duplicate directed edges for sampling

    :param data:
    :return:
    """
    if data.edge_attr is not None and data.edge_attr.ndim == 1:
        edge_attr = data.edge_attr.unsqueeze(-1)
    else:
        edge_attr = data.edge_attr

    undirected = is_undirected(data.edge_index, edge_attr, data.num_nodes)

    if undirected:
        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = edge_attr[keep_edge, :] if edge_attr is not None else edge_attr
    else:
        edge_index = data.edge_index

    return edge_index, edge_attr, undirected


def khop_subgraph_sampling(data: Data, n_subgraphs: int, khop: int = 3) -> List[Data]:
    """
    Sample the k-hop-neighbors subgraphs randomly

    :param data:
    :param n_subgraphs:
    :param khop:
    :return:
    """
    sample_indices = random.sample(range(data.num_nodes), n_subgraphs)
    graphs = []

    for idx in sample_indices:
        _, edge_index, _, edge_mask = k_hop_subgraph(idx,
                                                     khop,
                                                     data.edge_index,
                                                     relabel_nodes=False,
                                                     num_nodes=data.num_nodes)
        graphs.append(Data(x=data.x,
                           edge_index=edge_index,
                           edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                           num_nodes=data.num_nodes,
                           y=data.y))

    return graphs
