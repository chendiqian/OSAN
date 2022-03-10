import itertools
import os
import pickle
from collections import defaultdict
from typing import Union

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.custom_dataloader import MYDataLoader
from imle.noise import SumOfGammaNoiseDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from subgraph.construct import (edgemasked_graphs_from_nodemask, edgemasked_graphs_from_edgemask,
                                construct_subgraph_batch, )
from training.imle_scheme import get_split_idx, make_get_batch_topk, make_khop_subpgrah

from models import NetGINE, NetGCN

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau]
Emb_model = Union[NetGCN]
Train_model = Union[NetGINE]
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


class Trainer:
    def __init__(self,
                 task_type: str,
                 imle_sample_policy: str,
                 sample_k: int,
                 sample_khop: dict,
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
        :param imle_sample_policy:
        :param sample_k: sample nodes or edges
        :param sample_khop: the dictionary for sampling k-hop neighbors
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
        self.imle_sample_policy = imle_sample_policy
        self.sample_k = sample_k
        self.sample_khop = sample_khop
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.best_val_loss = 1e5
        self.patience = 0
        self.max_patience = max_patience

        self.curves = defaultdict(list)

        if train_embd_model:
            self.temp = 1.
            self.target_distribution = TargetDistribution(alpha=1.0, beta=beta)
            self.noise_distribution = SumOfGammaNoiseDistribution(k=self.temp,
                                                                  nb_iterations=100,
                                                                  device=device)
            # select scheme
            if self.imle_sample_policy == 'node':
                self.imle_graphs_from_masks = edgemasked_graphs_from_nodemask
            elif self.imle_sample_policy == 'edge':
                self.imle_graphs_from_masks = edgemasked_graphs_from_edgemask
            elif self.imle_sample_policy == 'khop_subgraph':
                self.imle_graphs_from_masks = edgemasked_graphs_from_edgemask

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
                split_idx = get_split_idx(data.ptr if self.imle_sample_policy == 'node' else
                                          data._slice_dict['edge_index'])
                logits = emb_model(data)
                graphs = Batch.to_data_list(data)

                if self.imle_sample_policy in ['edge', 'node']:
                    torch_sample_scheme = make_get_batch_topk(split_idx, self.sample_k, return_list=False, sample=False)
                elif self.imle_sample_policy == 'khop_subgraph':
                    torch_sample_scheme = make_khop_subpgrah(split_idx, graphs, False, False, **self.sample_khop)
                else:
                    raise NotImplementedError

                @imle(target_distribution=self.target_distribution,
                      noise_distribution=self.noise_distribution,
                      input_noise_temperature=self.temp,
                      target_noise_temperature=self.temp,
                      nb_samples=1)
                def imle_sample_scheme(logits: torch.Tensor):
                    return torch_sample_scheme(logits)

                sample_idx = imle_sample_scheme(logits)
                # each mask has shape (n_nodes, n_subgraphs)
                sample_idx = torch.split(sample_idx, split_idx, dim=0)
                # original graphs
                list_subgraphs, edge_weights = zip(
                    *[self.imle_graphs_from_masks(g, i.T, grad=True) for g, i in
                      zip(graphs, sample_idx)])
                list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))
                data = construct_subgraph_batch(list_subgraphs, [_.shape[1] for _ in sample_idx], edge_weights,
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
                   model: Train_model, ):
        if emb_model is not None:
            emb_model.eval()
        model.eval()
        val_losses = torch.tensor(0., device=self.device)
        num_graphs = 0

        for v in range(self.voting):
            for data in dataloader:
                data = data.to(self.device)

                if emb_model is not None:
                    split_idx = get_split_idx(data.ptr if self.imle_sample_policy == 'node' else
                                              data._slice_dict['edge_index'])
                    logits = emb_model(data)
                    if self.imle_sample_policy in ['edge', 'node']:
                        torch_sample_scheme = make_get_batch_topk(split_idx, self.sample_k, return_list=True,
                                                                  sample=True)
                    elif self.imle_sample_policy == 'khop_subgraph':
                        torch_sample_scheme = make_khop_subpgrah(split_idx, data, True, True, **self.sample_khop)
                    else:
                        raise NotImplementedError
                    sample_node_idx = torch_sample_scheme(logits)
                    graphs = Batch.to_data_list(data)
                    list_subgraphs, edge_weights = zip(*[self.imle_graphs_from_masks(g, i.T, grad=False) for g, i in
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
