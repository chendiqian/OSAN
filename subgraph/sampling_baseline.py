from typing import List, Tuple
import random

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected, k_hop_subgraph

from subgraph.construct import nodesubset_to_subgraph
from subgraph.mst_subgraph import kruskal_max_span_tree
from subgraph.greedy_expanding_tree import numba_greedy_expand_tree


def node_rand_sampling(graph: Data,
                       n_subgraphs: int,
                       node_per_subgraph: int = -1,
                       relabel: bool = False,
                       add_full_graph: bool = False) -> List[Data]:
    """
    Sample subgraphs.

    :param graph:
    :param n_subgraphs:
    :param node_per_subgraph:
    :param relabel:
    :param add_full_graph:
    :return:
        A list of graphs and their index masks
    """
    n_nodes = graph.num_nodes

    if node_per_subgraph < 0:  # drop nodes
        node_per_subgraph += n_nodes

    graphs = [graph] if add_full_graph else []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_nodes)[:node_per_subgraph]
        sort_indices = torch.sort(indices).values
        graphs.append(nodesubset_to_subgraph(graph, sort_indices, relabel=relabel))

    return graphs


def edge_rand_sampling(graph: Data,
                       n_subgraphs: int,
                       edge_per_subgraph: int = -1,
                       add_full_graph: bool = False) -> List[Data]:
    """

    :param graph:
    :param n_subgraphs:
    :param edge_per_subgraph:
    :param add_full_graph:
    :return:
    """
    edge_index, edge_attr, undirected = edge_sample_preproc(graph)

    n_edge = edge_index.shape[1]
    if n_edge == 0 or edge_per_subgraph > n_edge:
        return [graph for _ in range(n_subgraphs)]

    if edge_per_subgraph < 0:  # drop edges
        edge_per_subgraph += n_edge

    graphs = [graph] if add_full_graph else []
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


def khop_subgraph_sampling(data: Data,
                           n_subgraphs: int,
                           khop: int = 3,
                           relabel: bool = False,
                           add_full_graph: bool = False) -> List[Data]:
    """
    Sample the k-hop-neighbors subgraphs randomly

    :param data:
    :param n_subgraphs:
    :param khop:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    sample_indices = random.sample(range(data.num_nodes), n_subgraphs)
    graphs = [data] if add_full_graph else []

    for idx in sample_indices:
        node_idx, edge_index, _, edge_mask = k_hop_subgraph(idx,
                                                            khop,
                                                            data.edge_index,
                                                            relabel_nodes=relabel,
                                                            num_nodes=data.num_nodes)

        num_nodes = data.num_nodes
        if relabel:
            if node_idx.dtype in [torch.bool, torch.uint8]:
                num_nodes = node_idx.sum()
            elif node_idx.dtype in [torch.int32, torch.int64]:
                num_nodes = node_idx.numel()

        graphs.append(Data(x=data.x if not relabel else data.x[node_idx],
                           edge_index=edge_index,
                           edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                           num_nodes=num_nodes,
                           y=data.y))
    return graphs


def max_spanning_tree_subgraph(data: Data, n_subgraphs: int, add_full_graph: bool = False) -> List[Data]:
    """

    :param data:
    :param n_subgraphs:
    :param add_full_graph:
    :return:
    """
    graphs = [data] if add_full_graph else []
    np_edge_index = data.edge_index.cpu().numpy().T
    for i in range(n_subgraphs):
        edge_mask = kruskal_max_span_tree(np_edge_index, None, data.num_nodes)
        graphs.append(Data(x=data.x,
                           edge_index=data.edge_index[:, edge_mask],
                           edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                           num_nodes=data.num_nodes,
                           y=data.y))
    return graphs


def greedy_expand_sampling(graph: Data,
                           n_subgraphs: int,
                           node_per_subgraph: int = -1,
                           relabel: bool = False,
                           add_full_graph: bool = False) -> List[Data]:
    """

    :param graph:
    :param n_subgraphs:
    :param node_per_subgraph:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    if node_per_subgraph >= graph.num_nodes:
        return [graph] * n_subgraphs

    edge_index = graph.edge_index.cpu().numpy()
    graphs = [graph] if add_full_graph else []
    for _ in range(n_subgraphs):
        node_mask = numba_greedy_expand_tree(edge_index, node_per_subgraph, None, graph.num_nodes, repeat=1)
        graphs.append(nodesubset_to_subgraph(graph, torch.from_numpy(node_mask), relabel=relabel))

    return graphs
