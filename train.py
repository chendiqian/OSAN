import os
from typing import Union
from collections import defaultdict
import itertools
import pickle

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from models import NetGINE, NetGCN
from data.custom_dataloader import MYDataLoader
from subgraph_utils import edgemasked_graphs_from_nodemask, construct_subgraph_batch

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

DATASET_MEAN_NUM_NODE_DICT = {'zinc': 23.1514}

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau]
Emb_model = Union[NetGCN]
Train_model = Union[NetGINE]
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


def make_get_batch_topk(ptr, sample_k, return_list, sample):
    @torch.no_grad()
    def torch_get_batch_topk(logits: torch.Tensor) -> torch.Tensor:
        """
        Select topk in each col, return as float of 0 and 1s

        :param logits:
        :return:
        """
        logits = logits.detach()
        logits = torch.split(logits, ptr)

        sample_node_idx = []
        for l in logits:
            k = sample_k + l.shape[0] if sample_k < 0 else sample_k  # e.g. -1 -> remove 1 node

            if sample:
                noise = torch.randn(l.shape, device=l.device) * (l.std(0) * 0.1)
                l = l.clone() + noise

            thresh = torch.topk(l, k=min(k, l.shape[0]), dim=0, sorted=True).values[-1, :]  # kth largest
            # shape (n_nodes, dim)
            mask = (l >= thresh[None]).to(torch.float)
            mask.requires_grad = False
            sample_node_idx.append(mask)

        if not return_list:
            sample_node_idx = torch.cat(sample_node_idx, dim=0)
            sample_node_idx.requires_grad = False
        return sample_node_idx

    return torch_get_batch_topk


class Trainer:
    def __init__(self,
                 task_type: str,
                 sample_node_k: int,
                 voting: int,
                 max_patience: int,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 criterion: Loss,
                 train_embd_model: bool,
                 beta: float,
                 device: Union[str, torch.device]):
        """

        :param task_type:
        :param sample_node_k:
        :param voting:
        :param max_patience:
        :param optimizer:
        :param scheduler:
        :param criterion:
        :param train_embd_model:
        :param beta:
        :param device:
        """
        super(Trainer, self).__init__()

        assert task_type == 'regression', "Does not support tasks other than regression"
        self.task_type = task_type
        self.voting = voting
        self.sample_node_k = sample_node_k
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.best_val_loss = 1e5
        self.patience = 0
        self.max_patience = max_patience

        self.curves = defaultdict(list)

        if train_embd_model:
            self.temp = float(sample_node_k if sample_node_k > 0 else DATASET_MEAN_NUM_NODE_DICT['zinc'] + sample_node_k)
            self.target_distribution = TargetDistribution(alpha=1.0, beta=beta)
            self.noise_distribution = SumOfGammaNoiseDistribution(k=self.temp,
                                                                  nb_iterations=100,
                                                                  device=device)

    def train(self,
              dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
              emb_model: Emb_model,
              model: Train_model):

        if emb_model is not None:
            emb_model.train()
        model.train()
        train_losses = torch.tensor(0., device=self.device)
        num_graphs = 0

        for data in dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            if emb_model is not None:
                split_idx = tuple((data.ptr[1:] - data.ptr[:-1]).detach().cpu().tolist())
                logits = emb_model(data)
                torch_get_batch_topk = make_get_batch_topk(split_idx, self.sample_node_k, return_list=False, sample=False)

                @imle(target_distribution=self.target_distribution,
                      noise_distribution=self.noise_distribution,
                      input_noise_temperature=self.temp,
                      target_noise_temperature=self.temp,
                      nb_samples=1)
                def imle_get_batch_topk(logits: torch.Tensor):
                    return torch_get_batch_topk(logits)

                sample_node_idx = imle_get_batch_topk(logits)
                # each mask has shape (n_nodes, n_subgraphs)
                sample_node_idx = torch.split(sample_node_idx, split_idx)
                # original graphs
                graphs = Batch.to_data_list(data)
                list_subgraphs, edge_weights = zip(
                    *[edgemasked_graphs_from_nodemask(g, i.T, grad=True) for g, i in
                      zip(graphs, sample_node_idx)])
                list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))
                data = construct_subgraph_batch(list_subgraphs, [_.shape[1] for _ in sample_node_idx], edge_weights,
                                                self.device)

            pred = model(data)
            loss = self.criterion(pred, data.y)

            loss.backward()
            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
            self.optimizer.step()

        train_loss = train_losses.item() / num_graphs
        self.curves['train'].append(train_loss)
        return train_loss

    @torch.no_grad()
    def validation(self,
                   dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
                   emb_model: Emb_model,
                   model: Train_model,):
        if emb_model is not None:
            emb_model.eval()
        model.eval()
        val_losses = torch.tensor(0., device=self.device)
        num_graphs = 0

        for v in range(self.voting):
            for data in dataloader:
                data = data.to(self.device)

                if emb_model is not None:
                    split_idx = tuple((data.ptr[1:] - data.ptr[:-1]).detach().cpu().tolist())
                    logits = emb_model(data)
                    torch_get_batch_topk = make_get_batch_topk(split_idx, self.sample_node_k, return_list=True, sample=True)
                    sample_node_idx = torch_get_batch_topk(logits)
                    graphs = Batch.to_data_list(data)
                    list_subgraphs, edge_weights = zip(*[edgemasked_graphs_from_nodemask(g, i.T, grad=False) for g, i in
                                                         zip(graphs, sample_node_idx)])
                    list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))

                    # new batch
                    data = construct_subgraph_batch(list_subgraphs, [_.shape[1] for _ in sample_node_idx], edge_weights,
                                                    self.device)

                pred = model(data)
                loss = self.criterion(pred, data.y)

                val_losses += loss * data.num_graphs
                num_graphs += data.num_graphs

        val_loss = val_losses.item() / num_graphs
        self.scheduler.step(val_loss)

        early_stop = False
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience = 0
        else:
            self.patience += 1
            if self.patience > self.max_patience:
                early_stop = True

        self.curves['val'].append(val_loss)
        return val_loss, early_stop

    def save_curve(self, path):
        pickle.dump(self.curves, open(os.path.join(path, 'curves.pkl'), "wb"))
