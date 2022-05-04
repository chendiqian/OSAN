from collections import namedtuple, deque
from typing import List, Union, Tuple, Optional

import numba
import numpy as np
import torch
from torch import Tensor
from torch import device as TorchDevice
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce


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


AttributedDataLoader = namedtuple(
    'AttributedDataLoader', [
        'loader',
        'mean',
        'std'
    ])


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


class GraphCoalesce:

    def __call__(self, graph: Data):
        if graph.edge_attr is None:
            edge_index = coalesce(graph.edge_index, None, num_nodes=graph.num_nodes)
            edge_attr = None
        else:
            edge_index, edge_attr = coalesce(graph.edge_index, graph.edge_attr, graph.num_nodes)
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


def scale_grad(model: torch.nn.Module, scalar: Union[int, float]) -> torch.nn.Module:
    """
    Scale down the grad, typically for accumulated grad while micro batching

    :param model:
    :param scalar:
    :return:
    """
    assert scalar > 0

    if scalar == 1:
        return model

    for p in model.parameters():
        if p.grad is not None:
            p.grad /= scalar

    return model


def pair_wise_idx(n_subg: int, device: TorchDevice) -> Tuple[Tensor, Tensor]:
    senders = torch.arange(n_subg, device=device).repeat(n_subg - 1)
    receivers = torch.arange(n_subg, device=device)[None].repeat(n_subg - 1, 1)
    receivers = (receivers + torch.arange(n_subg - 1, device=device)[:, None] + 1) % n_subg
    receivers = receivers.reshape(-1)
    return senders, receivers


class IsBetter:
    """
    A comparator for different metrics, to unify >= and <=

    """
    def __init__(self, task_type):
        self.task_type = task_type

    def __call__(self, val1: float, val2: Optional[float]) -> bool:
        if val2 is None:
            return True

        if self.task_type in ['regression']:
            return val1 <= val2
        elif self.task_type in ['rocauc', 'acc']:
            return val1 >= val2
        else:
            raise ValueError
