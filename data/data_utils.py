from collections import namedtuple, deque
from typing import List

import numba
import numpy as np
import torch
from torch import Tensor
from torch import device as TorchDevice
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class SubgraphSetBatch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device: TorchDevice):
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.to(device))
        return self

    def __repr__(self):
        string = []
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                string.append(k + f': Tensor {list(v.shape)}')
            else:
                string.append(k + ': ' + type(v).__name__)
        return ' '.join(string)


# SubgraphSetBatch = namedtuple(
#     'SubgraphSetBatch', [
#         'x',
#         'edge_index',
#         'edge_attr',
#         'edge_weight',
#         'y',
#         'batch',
#         'inter_graph_idx',
#         'ptr',
#         'num_graphs',
#     ])


class GraphToUndirected:
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
        if graph.edge_attr is not None:
            edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
        else:
            edge_index = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
            edge_attr = None
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


@numba.njit(cache=True)
def edgeindex2neighbordict(edge_index: np.ndarray, num_nodes: int) -> List[List[int]]:
    """

    :param edge_index: shape (2, E)
    :param num_nodes:
    :return:
    """
    neighbors = [[-1] for _ in range(num_nodes)]
    for i, node in enumerate(edge_index[0]):
        neighbors[node].append(edge_index[1][i])

    for i, n in enumerate(neighbors):
        n.pop(0)
    return neighbors


def get_ptr(graph_idx: Tensor, device: torch.device = None) -> Tensor:
    """
    Given indices of graph, return ptr
    e.g. [1,1,2,2,2,3,3,4] -> [0, 2, 5, 7]

    :param graph_idx:
    :param device:
    :return:
    """
    if device is None:
        device = graph_idx.device
    return torch.cat((torch.tensor([0], device=device),
                      (graph_idx[1:] > graph_idx[:-1]).nonzero().reshape(-1) + 1,
                      torch.tensor([len(graph_idx)], device=device)), dim=0)


def get_connected_components(subset, neighbor_dict):
    components = []

    while subset:
        cur_node = subset.pop()
        q = deque()
        q.append(cur_node)

        component = [cur_node]
        while q:
            cur_node = q.popleft()
            i = 0
            while i < len(subset):
                candidate = subset[i]
                if candidate in neighbor_dict[cur_node]:
                    subset.pop(i)
                    component.append(candidate)
                    q.append(candidate)
                else:
                    i += 1
        components.append(component)

    return components
