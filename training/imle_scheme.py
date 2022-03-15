from typing import Tuple

import torch

from subgraph.khop_subgraph import khop_subgraphs
from subgraph.greedy_expanding_tree import greedy_expand_tree
from subgraph.mst_subgraph import mst_subgraph_sampling


def get_split_idx(inc_tensor: torch.Tensor) -> Tuple:
    """
    Get splits from accumulative vector

    :param inc_tensor:
    :return:
    """
    return tuple((inc_tensor[1:] - inc_tensor[:-1]).detach().cpu().tolist())


def make_get_batch_topk(ptr, sample_k, return_list, sample):
    @torch.no_grad()
    def torch_get_batch_topk(logits: torch.Tensor) -> torch.Tensor:
        """
        Select topk in each col, return as float of 0 and 1s
        Works for both edge and node sampling
        Not guaranteed to be connected

        :param logits:Batch
        :return:
        """
        logits = logits.detach()
        logits = torch.split(logits, ptr, dim=0)

        sample_instance_idx = []
        for l in logits:
            k = sample_k + l.shape[0] if sample_k < 0 else sample_k  # e.g. -1 -> remove 1 node

            if sample:
                noise = torch.randn(l.shape, device=l.device) * (l.std(0) * 0.1)
                l = l.clone() + noise

            thresh = torch.topk(l, k=min(k, l.shape[0]), dim=0, sorted=True).values[-1, :]  # kth largest
            # shape (n_nodes, dim)
            mask = (l >= thresh[None]).to(torch.float)
            mask.requires_grad = False
            sample_instance_idx.append(mask)

        if not return_list:
            sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
            sample_instance_idx.requires_grad = False
        return sample_instance_idx

    return torch_get_batch_topk


def make_khop_subpgrah(ptr, graphs, khop, return_list, sample):
    @torch.no_grad()
    def torch_khop_subgraph(logits):
        """
        Connected khop-subgraphs, can be pruned with max spanning tree algorithm

        :param logits:
        :return:
        """
        logits = logits.detach()
        logits = torch.split(logits, ptr, dim=0)

        sample_instance_idx = []
        for i, l in enumerate(logits):
            if sample:
                noise = torch.randn(l.shape, device=l.device) * (l.std(0) * 0.1)
                l = l.clone() + noise

            mask = khop_subgraphs(graphs[i],
                                  khop,
                                  instance_weight=l).T
            mask.requires_grad = False
            sample_instance_idx.append(mask)

        if not return_list:
            sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
            sample_instance_idx.requires_grad = False
        return sample_instance_idx

    return torch_khop_subgraph


def make_mst_subgraph(ptr, graphs, return_list, sample):
    @torch.no_grad()
    def torch_mst_subgraph(logits):
        logits = logits.detach()
        logits = torch.split(logits, ptr, dim=0)

        sample_instance_idx = []
        for i, l in enumerate(logits):
            if sample:
                noise = torch.randn(l.shape, device=l.device) * (l.std(0) * 0.1)
                l = l.clone() + noise

            mask = mst_subgraph_sampling(graphs[i], l).T
            mask.requires_grad = False
            sample_instance_idx.append(mask)

        if not return_list:
            sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
            sample_instance_idx.requires_grad = False
        return sample_instance_idx

    return torch_mst_subgraph


def make_greedy_expand_subpgrah(ptr, graphs, sample_k, return_list, sample):
    @torch.no_grad()
    def torch_greedy_expand_subgraph(logits):
        """
        Connected khop-subgraphs, can be pruned with max spanning tree algorithm

        :param logits:
        :return:
        """
        logits = logits.detach()
        logits = torch.split(logits, ptr, dim=0)

        sample_instance_idx = []
        for i, l in enumerate(logits):
            if sample:
                noise = torch.randn(l.shape, device=l.device) * (l.std(0) * 0.1)
                l = l.clone() + noise

            mask = greedy_expand_tree(graphs[i], l, sample_k).T
            mask.requires_grad = False
            sample_instance_idx.append(mask)

        if not return_list:
            sample_instance_idx = torch.cat(sample_instance_idx, dim=0)
            sample_instance_idx.requires_grad = False
        return sample_instance_idx

    return torch_greedy_expand_subgraph
