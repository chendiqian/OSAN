from collections import namedtuple

from torch import Tensor
from torch import device as TorchDevice


class SubgraphSetBatch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device: TorchDevice):
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.to(device))
        return self


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

