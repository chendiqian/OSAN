from torch import Tensor
from .GINE_gnn import NetGINE
from .GCN_gnn import NetGCN


def residual(x: Tensor, y: Tensor) -> Tensor:
    if x.shape == y.shape:
        return (x + y) / 2 ** 0.5
    else:
        return y
