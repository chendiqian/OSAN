import torch
from torch_scatter import scatter


class Nodemask2Edgemask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask, *args):
        """
        Given node masks, return edge masks as edge weights for message passing.

        :param ctx:
        :param mask:
        :param args:
        :return:
        """
        assert mask.dtype == torch.float  # must be differentiable
        edge_index, n_nodes = args
        ctx.save_for_backward(mask, edge_index[1], n_nodes)
        return nodemask2edgemask(mask, edge_index)

    @staticmethod
    def backward(ctx, grad_output):
        _, edge_index_col, n_nodes = ctx.saved_tensors
        final_grad = scatter(grad_output, edge_index_col, dim=-1, reduce='sum', dim_size=n_nodes)
        return final_grad, None, None


def nodemask2edgemask(mask: torch.Tensor, edge_index: torch.Tensor, placeholder=None) -> torch.Tensor:
    """
    util function without grad

    :param mask:
    :param edge_index:
    :param placeholder:
    :return:
    """
    single = mask.ndim == 1
    return mask[edge_index[0]] * mask[edge_index[1]] if single else mask[:, edge_index[0]] * mask[:, edge_index[1]]
