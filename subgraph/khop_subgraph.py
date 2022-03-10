from typing import Optional, Union, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph, to_undirected
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


def kruskal_max_span_tree(edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
    """
    My own implementation

    :param edge_index:
    :param edge_weight:
    :param num_nodes:
    :return:
    """
    directed_mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, directed_mask]  # to_directed
    edge_index_list = edge_index.t().cpu().tolist()
    sort_index = torch.argsort(edge_weight[directed_mask], descending=True) if edge_weight is not None else \
        torch.arange(len(edge_index_list), device=edge_index.device)
    parts = [set(edge_index_list[sort_index[0]])]
    edge_mask = [sort_index[:1]]
    node_close_list = set(edge_index_list[sort_index[0]])

    for idx in sort_index[1:]:
        n1, n2 = edge_index_list[idx]
        if n1 not in node_close_list and n2 not in node_close_list:  # new connected component
            parts.append({n1, n2})
            edge_mask.append(idx[None])
            node_close_list.add(n1)
            node_close_list.add(n2)
        elif n1 in node_close_list and n2 not in node_close_list:
            for s in parts:
                if n1 in s:
                    s.add(n2)
                    node_close_list.add(n2)
                    edge_mask.append(idx[None])
                    break
        elif n2 in node_close_list and n1 not in node_close_list:
            for s in parts:
                if n2 in s:
                    s.add(n1)
                    node_close_list.add(n1)
                    edge_mask.append(idx[None])
                    break
        elif n1 in node_close_list and n2 in node_close_list:
            s_idx1, s_idx2 = 0, 0
            for si, s in enumerate(parts):
                if n1 in s:
                    s_idx1 = si
                if n2 in s:
                    s_idx2 = si
            if s_idx1 != s_idx2:
                s_idx1, s_idx2 = (s_idx1, s_idx2) if (s_idx1 < s_idx2) else (s_idx2, s_idx1)
                s1 = parts.pop(s_idx2)
                s2 = parts.pop(s_idx1)
                parts.append(s1 | s2)
                edge_mask.append(idx[None])

    assert len(parts) == 1  # if the original graph is a connected component, then the span tree is also a component
    edge_mask = torch.cat(edge_mask)
    edge_index, edge_mask = to_undirected(edge_index[:, edge_mask],
                                          edge_mask,
                                          num_nodes=num_nodes)
    return edge_index, edge_mask


def khop_subgraphs(graph: Data,
                   khop: int = 3,
                   edge_weight: Optional[Tensor] = None,
                   prune_policy: str = 'mst',
                   coverage: str = 'full',
                   n_subgraphs: int = None):
    """
    For each seed node, get the k-hop neighbors first, then prune the graph as e.g. max spanning tree

    :param graph:
    :param khop:
    :param edge_weight:
    :param prune_policy:
    :param coverage:
    :param n_subgraphs:
    :return:
    """
    n_nodes, n_edges = graph.num_nodes, graph.num_edges
    graph_list = []
    edge_weight4grad_list = []

    def add_subgraph(idx: Union[int, list, Tensor], _edge_weight: Optional[Tensor]) -> Tuple[Tensor, Union[Tensor, None]]:
        _node_idx, _edge_index, _, edge_mask = k_hop_subgraph(idx, khop, graph.edge_index, relabel_nodes=False)
        sub_edge_weight = _edge_weight[edge_mask] if _edge_weight is not None else None

        _edge_attr = graph.edge_attr[edge_mask]
        if prune_policy == 'mst':
            _edge_index, edge_mask = kruskal_max_span_tree(_edge_index, sub_edge_weight, graph.num_nodes)
            _edge_attr = _edge_attr[edge_mask]
        elif prune_policy is None:
            pass
        else:
            raise NotImplementedError(f"Not supported policy: {prune_policy}")

        graph_list.append(Data(x=graph.x,
                               edge_index=_edge_index,
                               edge_attr=_edge_attr,
                               num_nodes=graph.num_nodes,
                               y=graph.y))
        edge_weight4grad = torch.zeros(_edge_index.shape[1], device=_edge_index.device, dtype=torch.float32)
        edge_weight4grad[edge_mask] = 1.0
        edge_weight4grad_list.append(edge_weight4grad)
        return _node_idx, edge_mask

    if coverage == 'full':  # sample for each seed node
        for i in range(n_nodes):
            _ = add_subgraph(i, edge_weight)

    elif coverage == 'k_subgraph':  # sample at least k subgraphs so that the whole graph is covered
        assert n_subgraphs is not None
        if edge_weight is None:
            edge_weight = torch.ones(n_edges, dtype=torch.float32, device=graph.edge_index.device)

        close_list = set()
        weight_for_sample = edge_weight.clone()
        max_val = weight_for_sample.max().abs()

        while len(graph_list) < n_subgraphs or len(close_list) < n_nodes:
            idx = torch.argmax(weight_for_sample)
            idx = graph.edge_index[0, idx][None]
            node_idx, visited_edges = add_subgraph(idx, edge_weight)
            weight_for_sample[visited_edges] -= abs(max_val)
            close_list.update(node_idx.cpu().tolist())

    return graph_list, edge_weight4grad_list
