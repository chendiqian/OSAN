from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


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
    sampled_masks = torch.zeros_like(instance_weight, dtype=torch.float32, device=instance_weight.device)
    indices = torch.argmax(instance_weight, dim=0)

    for i, idx in enumerate(indices):
        _node_idx, _edge_index, _, edge_mask = k_hop_subgraph(idx, khop, graph.edge_index, relabel_nodes=False)
        sampled_masks[_node_idx, i] = 1.0

    return sampled_masks
