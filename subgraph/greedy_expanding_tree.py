from typing import List, Optional
from heapq import *
import random

import torch
import numba
import numpy as np
from torch_geometric.data import Data


@numba.njit(cache=True, locals={'node_selected': numba.bool_[::1]})
def numba_greedy_expand_tree(edge_index: np.ndarray, k: int, node_weight: Optional[np.ndarray], num_nodes: Optional[int])\
        -> np.ndarray:
    """
    numba accelerated process

    :param edge_index:
    :param node_weight:
    :param k:
    :param num_nodes:
    :return:
    """
    row, col = edge_index
    cur_node = np.argmax(node_weight) if node_weight is not None else random.randint(0, num_nodes - 1)
    node_selected = np.zeros(num_nodes, dtype=np.bool_)
    q = [(-np.max(node_weight), cur_node)] if node_weight is not None else [(random.random(), cur_node)]
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
                    if node_weight is not None:
                        heappush(q, (-node_weight[neighbor], neighbor))
                    else:
                        heappush(q, (random.random(), neighbor))
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

    edge_index = graph.edge_index.cpu().numpy()
    n_subgraphs = node_weight.shape[1]
    node_weight = node_weight.t().cpu().numpy().astype(np.float64)

    node_masks = []

    for i in range(n_subgraphs):
        node_mask = numba_greedy_expand_tree(edge_index, k, node_weight[i], graph.num_nodes)
        node_masks.append(node_mask)

    node_masks = np.vstack(node_masks)
    return torch.from_numpy(node_masks).to(torch.float32).to(graph.edge_index.device)
