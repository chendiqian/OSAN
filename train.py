from typing import Union
import itertools

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from models import NetGINE, NetGCN
from custom_dataloder import MYDataLoader
from sample_subgraph import subgraphs_from_index


Optimizer = Union[torch.optim.adam.Adam,
                  torch.optim.sgd.SGD]
Emb_model = Union[NetGCN]
Train_model = Union[NetGINE]
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


def train(sample_k: int,
          dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
          emb_model: Emb_model,
          model: Train_model,
          optimizer: Optimizer,
          criterion: Loss,
          device: Union[str, torch.device]) -> Union[torch.Tensor, torch.float, float]:
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

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        # split batch to graphs
        logits = torch.tensor_split(emb_model(data), data.ptr[1:-1])
        sample_node_idx = [torch.topk(l, k=sample_k, dim=0, sorted=False).indices.T.sort(-1).values
                           for l in logits]
        graphs = Batch.to_data_list(data)
        # sample subgraphs, each has k nodes
        list_subgraphs = [subgraphs_from_index(g, i) for g, i in zip(graphs, sample_node_idx)]
        list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))

        # new batch
        batch = Batch.from_data_list(list_subgraphs, None, None)
        original_graph_mask = torch.cat([torch.full((idx.shape[0],), i)
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
