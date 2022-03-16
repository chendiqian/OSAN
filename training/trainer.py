import itertools
import os
import pickle
from collections import defaultdict
from typing import Union

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.custom_dataloader import MYDataLoader
from imle.noise import SumOfGammaNoiseDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from subgraph.construct import (edgemasked_graphs_from_nodemask, edgemasked_graphs_from_edgemask,
                                construct_subgraph_batch, )
from training.imle_scheme import *

from models import NetGINE, NetGCN

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau]
Emb_model = Union[NetGCN]
Train_model = Union[NetGINE]
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]

LOGITS_SELECTION = {'node': lambda x, y: x,
                    'edge': lambda x, y: y,
                    'khop_subgraph': lambda x, y: x,
                    'greedy_exp': lambda x, y: x,
                    'mst': lambda x, y: y, }

SPLIT_IDX_SELECTION = {'node': lambda data: get_split_idx(data.ptr),
                       'edge': lambda data: get_split_idx(data._slice_dict['edge_index']),
                       'khop_subgraph': lambda data: get_split_idx(data.ptr),
                       'greedy_exp': lambda data: get_split_idx(data.ptr),
                       'mst': lambda data: get_split_idx(data._slice_dict['edge_index']), }

IMLE_SUBGRAPHS_FROM_MASK = {'node': edgemasked_graphs_from_nodemask,
                            'edge': edgemasked_graphs_from_edgemask,
                            'khop_subgraph': edgemasked_graphs_from_nodemask,
                            'greedy_exp': edgemasked_graphs_from_nodemask,
                            'mst': edgemasked_graphs_from_edgemask}


class Trainer:
    def __init__(self,
                 task_type: str,
                 imle_sample_policy: str,
                 sample_k: int,
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
            self.imle_scheduler = IMLEScheme(self.imle_sample_policy,
                                             None,
                                             None,
                                             self.sample_k,
                                             return_list=False,
                                             sample=False)

    def emb_model_forward(self, data: Union[Data, Batch], emb_model: Emb_model, train: bool) -> Union[Data, Batch]:
        """
        Common forward propagation for train and val, only called when embedding model is trained.

        :param data:
        :param emb_model:
        :param train:
        :return:
        """
        # select idx for splitting edge / node logits
        split_idx = SPLIT_IDX_SELECTION[self.imle_sample_policy](data)

        # forward
        logits_n, logits_e = emb_model(data)
        logits = LOGITS_SELECTION[self.imle_sample_policy](logits_n, logits_e)
        graphs = Batch.to_data_list(data)

        self.imle_scheduler.graphs = graphs
        self.imle_scheduler.ptr = split_idx

        if train:
            @imle(target_distribution=self.target_distribution,
                  noise_distribution=self.noise_distribution,
                  input_noise_temperature=self.temp,
                  target_noise_temperature=self.temp,
                  nb_samples=1)
            def imle_sample_scheme(logits: torch.Tensor):
                return self.imle_scheduler.torch_sample_scheme(logits)

            sample_idx = imle_sample_scheme(logits)
            sample_idx = torch.split(sample_idx, split_idx, dim=0)
        else:
            sample_idx = self.imle_scheduler.torch_sample_scheme(logits)

        # original graphs
        subgraphs_from_mask = IMLE_SUBGRAPHS_FROM_MASK[self.imle_sample_policy]
        list_subgraphs, edge_weights = zip(
            *[subgraphs_from_mask(g, i.T, grad=train) for g, i in
              zip(graphs, sample_idx)])
        list_subgraphs = list(itertools.chain.from_iterable(list_subgraphs))
        data = construct_subgraph_batch(list_subgraphs, [_.shape[1] for _ in sample_idx], edge_weights,
                                        self.device)

        return data

    def train(self,
              dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
              emb_model: Emb_model,
              model: Train_model):

        if emb_model is not None:
            emb_model.train()
            self.imle_scheduler.return_list = False
            self.imle_scheduler.sample = False

        model.train()
        train_losses = torch.tensor(0., device=self.device)
        num_graphs = 0

        for data in dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            if emb_model is not None:
                data = self.emb_model_forward(data, emb_model, train=True)

            pred = model(data)
            loss = self.criterion(pred, data.y)

            loss.backward()
            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
            self.optimizer.step()

        train_loss = train_losses.item() / num_graphs
        self.curves['train'].append(train_loss)

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return train_loss

    @torch.no_grad()
    def validation(self,
                   dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
                   emb_model: Emb_model,
                   model: Train_model, ):
        if emb_model is not None:
            emb_model.eval()
            self.imle_scheduler.return_list = True
            self.imle_scheduler.sample = True

        model.eval()
        val_losses = torch.tensor(0., device=self.device)
        num_graphs = 0

        for v in range(self.voting):
            for data in dataloader:
                data = data.to(self.device)

                if emb_model is not None:
                    data = self.emb_model_forward(data, emb_model, train=False)

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

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return val_loss, early_stop

    def save_curve(self, path):
        pickle.dump(self.curves, open(os.path.join(path, 'curves.pkl'), "wb"))
