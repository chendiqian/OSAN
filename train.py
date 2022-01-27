from typing import Union, List
import itertools

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from models import NetGINE, NetGCN
from custom_dataloder import MYDataLoader
from sample_subgraph import subgraphs_from_index, subgraphs_from_mask

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Emb_model = Union[NetGCN]
Train_model = Union[NetGINE]
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


def make_get_batch_topk(ptr, sample_k):
    @torch.no_grad()
    def torch_get_batch_topk(logits: torch.Tensor) -> torch.Tensor:
        """
        Select topk in each col, return as float of 0 and 1s

        :param logits:
        :return:
        """
        logits = logits.detach()
        logits = torch.tensor_split(logits, ptr)

        sample_node_idx = []
        for l in logits:
            thresh = torch.topk(l, k=min(sample_k, l.shape[0]), dim=0, sorted=True).values[-1, :]   # kth largest
            # shape (n_nodes, dim)
            mask = (l >= thresh[None]).to(torch.float)
            sample_node_idx.append(mask)

        sample_node_idx = torch.cat(sample_node_idx, dim=0)
        sample_node_idx.requires_grad = False
        return sample_node_idx

    return torch_get_batch_topk


def train(sample_k: int,
          dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
          emb_model: Emb_model,
          model: Train_model,
          optimizer: Optimizer,
          criterion: Loss,
          device: Union[str, torch.device]) -> Union[torch.Tensor, torch.FloatType, float]:
    """
    A train step

    :param sample_k:
    :param dataloader:
    :param emb_model:
    :param model:
    :param optimizer:
    :param criterion:
    :param device:
    :return:
    """
    emb_model.train()
    model.train()
    train_losses = torch.tensor(0., device=device)

    target_distribution = TargetDistribution(alpha=1.0, beta=10.0)
    noise_distribution = SumOfGammaNoiseDistribution(k=10, nb_iterations=100)

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        logits = emb_model(data)

        torch_get_batch_topk = make_get_batch_topk(data.ptr[1:-1], sample_k)

        @imle(target_distribution=target_distribution,
              noise_distribution=noise_distribution,
              input_noise_temperature=1.0,
              target_noise_temperature=1.0,
              nb_samples=1)
        def imle_get_batch_topk(logits: torch.Tensor):
            return torch_get_batch_topk(logits)

        sample_node_idx = imle_get_batch_topk(logits)
        # each mask has shape (n_nodes, hid_dims)
        sample_node_idx = torch.tensor_split(sample_node_idx, data.ptr[1:-1])
        graphs = Batch.to_data_list(data)
        list_subgraphs = [subgraphs_from_mask(g, i.T) for g, i in zip(graphs, sample_node_idx)]
        list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))

        # new batch
        batch = Batch.from_data_list(list_subgraphs, None, None)
        original_graph_mask = torch.cat([torch.full((idx.shape[1],), i)
                                         for i, idx in enumerate(sample_node_idx)], dim=0)
        ptr = torch.cat((torch.tensor([0]),
                         (original_graph_mask[1:] > original_graph_mask[:-1]).nonzero().reshape(-1) + 1,
                         torch.tensor([len(original_graph_mask)])), dim=0)
        batch.inter_graph_idx = original_graph_mask
        batch.ptr = ptr
        batch.y = batch.y[ptr[:-1]]

        pred = model(batch)
        loss = criterion(pred, batch.y)

        loss.backward()
        train_losses += loss * batch.num_graphs
        optimizer.step()

    return train_losses.item() / len(dataloader.dataset)


@torch.no_grad()
def validation(sample_k: int,
               dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
               emb_model: Emb_model,
               model: Train_model,
               criterion: Loss,
               device: Union[str, torch.device]) -> Union[torch.Tensor, torch.FloatType, float]:
    emb_model.eval()
    model.eval()
    val_losses = torch.tensor(0., device=device)

    for data in dataloader:
        data = data.to(device)
        logits = emb_model(data)

        torch_get_batch_topk = make_get_batch_topk(data.ptr[1:-1], sample_k)
        sample_node_idx = torch_get_batch_topk(logits)
        # each mask has shape (n_nodes, hid_dims)
        sample_node_idx = torch.tensor_split(sample_node_idx, data.ptr[1:-1])
        graphs = Batch.to_data_list(data)
        list_subgraphs = [subgraphs_from_mask(g, i.T) for g, i in zip(graphs, sample_node_idx)]
        list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))

        # new batch
        batch = Batch.from_data_list(list_subgraphs, None, None)
        original_graph_mask = torch.cat([torch.full((idx.shape[1],), i)
                                         for i, idx in enumerate(sample_node_idx)], dim=0)
        ptr = torch.cat((torch.tensor([0]),
                         (original_graph_mask[1:] > original_graph_mask[:-1]).nonzero().reshape(-1) + 1,
                         torch.tensor([len(original_graph_mask)])), dim=0)
        batch.inter_graph_idx = original_graph_mask
        batch.ptr = ptr
        batch.y = batch.y[ptr[:-1]]

        pred = model(batch)
        loss = criterion(pred, batch.y)

        val_losses += loss * batch.num_graphs

    return val_losses.item() / len(dataloader.dataset)
