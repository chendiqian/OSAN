from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_scatter import scatter
from subgraph.mst_subgraph import kruskal_max_span_tree


def khop_subgraphs(graph: Data,
                   khop: int = 3,
                   instance_weight: Optional[Tensor] = None,
                   prune_policy: str = None) -> Tensor:
    """
    Code for IMLE scheme, not sample on the fly
    For each seed node, get the k-hop neighbors first, then prune the graph as e.g. max spanning tree

    If not pruning, return the node masks, else edge masks

    :param graph:
    :param khop:
    :param instance_weight: if not pruning, this should be node weight, so that the seed node is picked according to
    the highest node weight. if pruning with MST algorithm, this should be edge weight, and node weight is the scatter
     of the incident edge weights
    :param prune_policy:
    :return: return node mask if not pruned, else edge mask
    """
    n_nodes, n_edges = graph.num_nodes, graph.num_edges
    sampled_masks = []

    if prune_policy == 'mst':
        node_weight = scatter(instance_weight, graph.edge_index[0], dim=0, reduce='sum')
    elif prune_policy is None:
        node_weight = instance_weight
    else:
        raise NotImplementedError(f"Not supported policy: {prune_policy}")

    def add_subgraph(ith: int, idx: Union[int, Tensor]):
        _node_idx, _edge_index, _, edge_mask = k_hop_subgraph(idx, khop, graph.edge_index, relabel_nodes=False)

        if prune_policy == 'mst':
            raise NotImplementedError
            # sub_edge_weight = instance_weight[:, ith][edge_mask]
            # sub_edge_mask = kruskal_max_span_tree(_edge_index, sub_edge_weight, graph.num_nodes)
            # edge_mask = torch.where(edge_mask)[0][sub_edge_mask]
            # instance_mask = torch.zeros(n_edges, device=_edge_index.device, dtype=torch.float32)
            # instance_mask[edge_mask] = 1.0
        elif prune_policy is None:
            instance_mask = torch.zeros(n_nodes, device=_node_idx.device, dtype=torch.float32)
            instance_mask[_node_idx] = 1.0
        else:
            raise NotImplementedError(f"Not supported policy: {prune_policy}")

        sampled_masks.append(instance_mask)

    indices = torch.argmax(node_weight, dim=0)

    for i, idx in enumerate(indices):
        add_subgraph(i, idx[None])

    return torch.vstack(sampled_masks)
