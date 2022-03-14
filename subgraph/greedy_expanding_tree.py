from typing import List
from heapq import *

import torch
import numba
import numpy as np
from torch_geometric.data import Data


@numba.njit(cache=True, locals={'node_selected': numba.bool_[::1]})
def numba_greedy_expand_tree(row: List, col: List, node_weight: List, k: int) -> np.ndarray:
    """
    numba accelerated process

    :param row:
    :param col:
    :param node_weight:
    :param k:
    :return:
    """
    cur_node = node_weight.index(max(node_weight))
    node_selected = np.zeros(len(node_weight), dtype=np.bool_)
    q = [(-node_weight[cur_node], cur_node)]
    heapify(q)

    cnt = 0

    while cnt < k:
        if not len(q):
            break

        _, cur_node = heappop(q)
        if node_selected[cur_node]:
            continue

        node_selected[cur_node] = True
        for i, node in enumerate(row):
            if node == cur_node:
                neighbor = col[i]
                if not node_selected[neighbor]:
                    heappush(q, (-node_weight[neighbor], neighbor))
            elif node > cur_node:
                break

        cnt += 1

    return node_selected


def greedy_expand_tree(graph: Data, node_weight: torch.Tensor, k: int) -> torch.Tensor:
    """

    :param graph:
    :param node_weight: shape (N, n_sugraphs)
    :param k: k nodes to pick for each subgraph
    :return:
    """
    if k >= graph.num_nodes:
        return torch.ones(node_weight.shape[1], node_weight.shape[0], device=node_weight.device)

    edge_index = graph.edge_index.cpu().tolist()
    n_subgraphs = node_weight.shape[1]
    node_weight = node_weight.t().cpu().tolist()

    node_masks = []

    for i in range(n_subgraphs):
        node_mask = numba_greedy_expand_tree(*edge_index, node_weight[i], k)
        node_masks.append(node_mask)

    node_masks = np.vstack(node_masks)
    return torch.from_numpy(node_masks).to(torch.float32).to(graph.edge_index.device)
