from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
import numpy as np
import numba

from data.data_utils import edgeindex2neighbordict


@numba.njit(cache=True, locals={'edge_mask': numba.bool_[::1], 'np_node_idx': numba.int64[::1]})
def numba_k_hop_subgraph(edge_index: np.ndarray, seed_node: int, khop: int, num_nodes: int, relabel: bool)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k_hop_subgraph of PyG is too slow

    :param edge_index:
    :param seed_node:
    :param khop:
    :param num_nodes:
    :param relabel:
    :return:
    """
    node_idx = {seed_node}
    visited_nodes = {seed_node}

    neighbor_dict = edgeindex2neighbordict(edge_index, num_nodes)

    cur_neighbors = {seed_node}
    for hop in range(khop):
        last_neighbors, cur_neighbors = cur_neighbors, set()

        for node in last_neighbors:
            for neighbor in neighbor_dict[node]:
                if neighbor not in visited_nodes:
                    cur_neighbors.add(neighbor)
                    visited_nodes.add(neighbor)
        node_idx.update(cur_neighbors)

    edge_mask = np.zeros(edge_index.shape[1], dtype=np.bool_)
    for i in range(edge_index.shape[1]):
        if edge_index[0][i] in node_idx and edge_index[1][i] in node_idx:
            edge_mask[i] = True

    edge_index = edge_index[:, edge_mask]
    node_idx = sorted(list(node_idx))

    if relabel:
        new_idx_dict = dict()
        for i, idx in enumerate(node_idx):
            new_idx_dict[idx] = i

        for i in range(2):
            for j in range(edge_index.shape[1]):
                edge_index[i, j] = new_idx_dict[edge_index[i, j]]

    np_node_idx = np.array(node_idx, dtype=np.int64)
    return np_node_idx, edge_index, edge_mask


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
    indices = torch.argmax(instance_weight, dim=0).cpu().tolist()

    edge_index = graph.edge_index.cpu().numpy()

    for i, idx in enumerate(indices):
        _node_idx, _edge_index, edge_mask = numba_k_hop_subgraph(edge_index, idx, khop, graph.num_nodes, False)
        sampled_masks[_node_idx, i] = 1.0

    return sampled_masks
