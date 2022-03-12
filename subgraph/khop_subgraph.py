from typing import Optional, Union, Tuple

import numpy as np
import numba
import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
# from networkx import maximum_spanning_tree as ntx_maximum_spanning_tree
# from networkx import Graph as NTXGraph

# cugraph MST, see:
# https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph.tree.minimum_spanning_tree.maximum_spanning_tree.html
# from cugraph.tree import minimum_spanning_tree_wrapper
# from cugraph.structure.graph_classes import Graph
# from torch_geometric.utils import k_hop_subgraph, to_cugraph
# from torch_geometric.utils.convert import from_cugraph


# def _maximum_spanning_tree_subgraph(G: Graph):
#     mst_subgraph = Graph()
#     if type(G) is not Graph:
#         raise Exception("input graph must be undirected")

#     if not G.adjlist:
#         G.view_adj_list()

#     if G.adjlist.weights is not None:
#         G.adjlist.weights = G.adjlist.weights.mul(-1)

#     with HideOutput():
#         mst_df = minimum_spanning_tree_wrapper.minimum_spanning_tree(G)

#     # revert to original weights
#     if G.adjlist.weights is not None:
#         G.adjlist.weights = G.adjlist.weights.mul(-1)
#         mst_df["weight"] = mst_df["weight"].mul(-1)

#     if G.renumbered:
#         mst_df = G.unrenumber(mst_df, "src")
#         mst_df = G.unrenumber(mst_df, "dst")

#     mst_subgraph.from_cudf_edgelist(
#         mst_df, source="src", destination="dst", renumber=False
#     )
#     return mst_subgraph

# def CuGraphMSTsample(graph: Data, khop: int = 3, edge_weight: Optional[Tensor] = None):
#     n_nodes, n_edges = graph.num_nodes, graph.num_edges
#     graph_list = []
#     for i in range(n_nodes):
#         node_idx, edge_index_list, _, edge_mask = k_hop_subgraph(i, khop, graph.edge_index_list, relabel_nodes=False)
#         sub_edge_weight = edge_weight[edge_mask] if edge_weight is not None else None
#         new_g = to_cugraph(edge_index_list, sub_edge_weight, relabel_nodes=False)
#         mst_g = _maximum_spanning_tree_subgraph(new_g)
#         mst_edge_index, mst_edge_weight = from_cugraph(mst_g)
#         graph_list.append(new_g)
#     return graph_list


# def to_networkx(num_nodes: int,
#                 edge_index: Tensor,
#                 node_idx: Tensor = None,
#                 edge_weight: Optional[Tensor] = None,
#                 remove_self_loops: bool = False) -> NTXGraph:
#     """
#     Adapted from
#     https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=to_networkx#torch_geometric.utils.to_networkx
#
#     :param num_nodes:
#     :param edge_index:
#     :param node_idx:
#     :param edge_weight:
#     :param remove_self_loops:
#     :return:
#     """
#
#     G = NTXGraph()
#
#     if node_idx is None:  # relabel
#         node_idx = list(range(num_nodes))
#     elif isinstance(node_idx, Tensor):
#         node_idx = node_idx.cpu().tolist()
#
#     G.add_nodes_from(node_idx)
#
#     if edge_weight is not None and isinstance(edge_weight, Tensor):
#         edge_weight = edge_weight.cpu().tolist()
#
#     for i, (u, v) in enumerate(edge_index.cpu().t().tolist()):
#         if u > v:
#             continue
#
#         if remove_self_loops and u == v:
#             continue
#
#         if edge_weight is not None:
#             G.add_edge(u, v, weight=edge_weight[i])
#         else:
#             G.add_edge(u, v, weight=1.)
#
#     return G
#
#
# def NetXMSTsample(node_idx: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
#     """
#     For a graph (subgraph already), get the kruskal max span tree with networkx implementation.
#     Drawback: need to change datatype back and forth, and cannot return the mask for which edges are selected
#
#     :param node_idx:
#     :param edge_index:
#     :param edge_weight:
#     :return:
#     """
#     ntx_g = to_networkx(node_idx.numel(),
#                         edge_index,
#                         node_idx,
#                         edge_weight=edge_weight if edge_weight is not None else None,
#                         remove_self_loops=True)
#     mst_g = ntx_maximum_spanning_tree(ntx_g)
#     edges = list(mst_g.edges)
#     edge_index = torch.tensor(edges, dtype=torch.long, device=edge_index.device).t().contiguous()
#     edge_index = to_undirected(edge_index, num_nodes=node_idx.numel())
#
#     return edge_index


@numba.njit(cache=True, locals={'parts': numba.int32[::1], 'edge_selected': numba.bool_[::1]})
def numba_kruskal(sort_index: np.ndarray, edge_index_list: np.ndarray, num_nodes: int) -> np.ndarray:
    parts = np.full(num_nodes, -1, dtype=np.int32)  # -1: unvisited
    edge_selected = np.zeros_like(sort_index, dtype=np.bool_)
    edge_selected[sort_index[0]] = True
    n1, n2 = edge_index_list[sort_index[0]]
    parts[n1] = 0
    parts[n2] = 0

    edge_selected_set = {(n1, n2,)}

    parts_hash = 1

    for idx in sort_index[1:]:
        n1, n2 = edge_index_list[idx]
        if (n2, n1) in edge_selected_set:
            edge_selected[idx] = True
            continue

        if parts[n1] == -1 and parts[n2] == -1:
            parts[n1] = parts_hash
            parts[n2] = parts_hash
            parts_hash += 1
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))
        elif parts[n1] != -1 and parts[n2] == -1:
            parts[n2] = parts[n1]
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))
        elif parts[n2] != -1 and parts[n1] == -1:
            parts[n1] = parts[n2]
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))
        elif parts[n1] != -1 and parts[n2] != -1 and parts[n1] != parts[n2]:
            parts[parts == parts[n2]] = parts[n1]
            edge_selected[idx] = True
            edge_selected_set.add((n1, n2,))

    return edge_selected


def kruskal_max_span_tree(edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
    """
    My own implementation

    :param edge_index:
    :param edge_weight:
    :param num_nodes:
    :return:
    """
    edge_index_list = edge_index.t().cpu().numpy()
    sort_index = torch.argsort(edge_weight, descending=True).cpu().numpy()

    edge_mask = numba_kruskal(sort_index, edge_index_list, num_nodes)
    edge_mask = torch.from_numpy(edge_mask).to(edge_index.device)
    return edge_mask


def khop_subgraphs(graph: Data,
                   khop: int = 3,
                   node_weight: Optional[Tensor] = None,
                   edge_weight: Optional[Tensor] = None,
                   prune_policy: str = 'mst') -> Tensor:
    """
    Code for IMLE scheme, not sample on the fly
    For each seed node, get the k-hop neighbors first, then prune the graph as e.g. max spanning tree

    :param graph:
    :param khop:
    :param node_weight:
    :param edge_weight:
    :param prune_policy:
    :return:
    """
    n_nodes, n_edges = graph.num_nodes, graph.num_edges
    node_weight4grad_list = []

    def add_subgraph(idx: Union[int, Tensor]):
        _node_idx, _edge_index, _, edge_mask = k_hop_subgraph(idx, khop, graph.edge_index, relabel_nodes=False)

        if prune_policy == 'mst':
            raise NotImplementedError
        elif prune_policy is None:
            node_mask = torch.zeros(n_nodes, device=_node_idx.device, dtype=torch.float32)
            node_mask[_node_idx] = 1.0
        else:
            raise NotImplementedError(f"Not supported policy: {prune_policy}")

        node_weight4grad_list.append(node_mask)

    indices = torch.argmax(node_weight, dim=0)

    for idx in indices:
        add_subgraph(idx[None])

    return torch.vstack(node_weight4grad_list)
