from typing import Tuple

import torch

from subgraph.khop_subgraph import khop_subgraphs
from subgraph.greedy_expanding_tree import greedy_expand_tree
from subgraph.mst_subgraph import mst_subgraph_sampling
from subgraph.or_optimal_subgraph import get_or_suboptim_subgraphs, get_or_optim_subgraphs


def get_split_idx(inc_tensor: torch.Tensor) -> Tuple:
    """
    Get splits from accumulative vector

    :param inc_tensor:
    :return:
    """
    return tuple((inc_tensor[1:] - inc_tensor[:-1]).detach().cpu().tolist())


class IMLEScheme:
    def __init__(self, imle_sample_policy, ptr, graphs, sample_k, return_list, sample):
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k
        self._ptr = ptr
        self._graphs = graphs
        self._return_list = return_list
        self._sample = sample

    @property
    def ptr(self):
        return self._ptr

    @ptr.setter
    def ptr(self, value):
        self._ptr = value

    @ptr.deleter
    def ptr(self):
        del self._ptr

    @property
    def graphs(self):
        return self._graphs

    @graphs.setter
    def graphs(self, new_graphs):
        self._graphs = new_graphs

    @graphs.deleter
    def graphs(self):
        del self._graphs

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

    @property
    def return_list(self):
        return self._return_list

    @return_list.setter
    def return_list(self, value):
        self._return_list = value

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):
        logits = logits.detach()
        logits = torch.split(logits, self.ptr, dim=0)

        sample_instance_idx = []
        for i, logit in enumerate(logits):
            k = self.sample_k + logit.shape[0] if self.sample_k < 0 else self.sample_k  # e.g. -1 -> remove 1 node

            if self.sample:
                noise = torch.randn(logit.shape, device=logit.device) * logit.std(0, keepdims=True)
                logit = logit.clone() + noise

            if self.imle_sample_policy in ['node', 'edge']:
                thresh = torch.topk(logit, k=min(k, logit.shape[0]), dim=0, sorted=True).values[-1, :]  # kth largest
                # shape (n_nodes, dim)
                mask = (logit >= thresh[None]).to(torch.float)
            elif self.imle_sample_policy == 'khop_subgraph':
                mask = khop_subgraphs(self.graphs[i], self.sample_k, instance_weight=logit)
            elif self.imle_sample_policy == 'mst':
                mask = mst_subgraph_sampling(self.graphs[i], logit).T
            elif self.imle_sample_policy == 'greedy_exp':
                mask = greedy_expand_tree(self.graphs[i], logit, self.sample_k).T
            elif self.imle_sample_policy == 'or':
                mask = get_or_suboptim_subgraphs(logit, self.sample_k)
            elif self.imle_sample_policy == 'or_optim':
                mask = get_or_optim_subgraphs(self.graphs[i], logit, self.sample_k)
            else:
                raise NotImplementedError

            mask.requires_grad = False
            sample_instance_idx.append(mask)

        if not self.return_list:
            sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
            sample_instance_idx.requires_grad = False
        return sample_instance_idx
