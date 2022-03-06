from collections import namedtuple

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
