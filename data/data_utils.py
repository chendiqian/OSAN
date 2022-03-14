from collections import namedtuple
from typing import Dict, Union, Tuple

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


def edgeindex2neighbordict(edge_index: torch.Tensor, num_nodes: int, num_edges: int) \
        -> Tuple[Dict[int, range], Dict[int, Tuple]]:
    """

    :param edge_index:
    :param num_nodes:
    :param num_edges:
    :return:
    """
    assert edge_index[0][0] == 0 and edge_index[0][-1] == num_nodes - 1
    mask = (torch.where(edge_index[0, 1:] > edge_index[0, :-1])[0] + 1).cpu().tolist()
    cols = tuple(edge_index[1].cpu().tolist())

    neighbor_idx_dict = {0: range(mask[0])}
    neighbor_dict = {0: cols[0: mask[0]]}

    for i in range(len(mask) - 1):
        neighbor_idx_dict[i + 1] = range(mask[i], mask[i + 1])
        neighbor_dict[i + 1] = cols[mask[i]: mask[i + 1]]
    neighbor_idx_dict[num_nodes - 1] = range(mask[-1], num_edges)
    neighbor_dict[num_nodes - 1] = cols[mask[-1]:]
    return neighbor_idx_dict, neighbor_dict
