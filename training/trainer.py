import itertools
import os
import pickle
from collections import defaultdict
from typing import Union, Optional
from ml_collections import ConfigDict

import torch.linalg
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.custom_scheduler import BaseScheduler, StepScheduler
from data.custom_dataloader import MYDataLoader
from data.data_utils import scale_grad
from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from subgraph.construct import (edgemasked_graphs_from_nodemask, edgemasked_graphs_from_edgemask,
                                construct_subgraph_batch, )
from training.imle_scheme import *

from models import NetGINE, NetGCN

Optimizer = Union[torch.optim.Adam,
                  torch.optim.SGD]
Scheduler = Union[torch.optim.lr_scheduler.ReduceLROnPlateau,
                  torch.optim.lr_scheduler.MultiStepLR]
Emb_model = Union[NetGCN]
Train_model = Union[NetGINE]
Loss = Union[torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]


class Trainer:
    def __init__(self,
                 task_type: str,
                 voting: int,
                 max_patience: int,
                 optimizer: Tuple[Optimizer, Optional[Optimizer]],
                 scheduler: Tuple[Scheduler, Optional[Scheduler]],
                 criterion: Loss,
                 device: Union[str, torch.device],
                 imle_configs: ConfigDict,

                 sample_policy: str = 'node',
                 sample_k: int = -1,
                 remove_node: bool = True,
                 add_full_graph: bool = True,
                 **kwargs):
        """

        :param task_type:
        :param voting:
        :param max_patience:
        :param optimizer:
        :param scheduler:
        :param criterion:
        :param device:
        :param imle_configs:
        :param sample_policy:
        :param sample_k:
        :param remove_node:
        :param add_full_graph:
        :param kwargs:
        """
        super(Trainer, self).__init__()

        self.task_type = task_type
        self.voting = voting
        self.optimizer, self.optimizer_embd = optimizer
        self.scheduler, self.scheduler_embd = scheduler
        self.criterion = criterion
        self.device = device

        self.best_val_loss = 1e5
        self.best_val_acc = 0.
        self.patience = 0
        self.max_patience = max_patience

        self.curves = defaultdict(list)

        if imle_configs is not None:  # need to cache some configs, otherwise everything's in the dataloader already
            self.aux_loss_weight = imle_configs.aux_loss_weight
            self.imle_sample_rand = imle_configs.imle_sample_rand
            self.micro_batch_embd = imle_configs.micro_batch_embd
            self.imle_sample_policy = sample_policy
            self.remove_node = remove_node
            self.add_full_graph = add_full_graph
            self.temp = 1.
            self.target_distribution = TargetDistribution(alpha=1.0, beta=imle_configs.beta)
            self.noise_distribution = GumbelDistribution(0., imle_configs.noise_scale, self.device)
            self.noise_scale_scheduler = BaseScheduler(imle_configs.noise_scale)
            self.imle_scheduler = IMLEScheme(sample_policy,
                                             None,
                                             None,
                                             sample_k,
                                             return_list=False,
                                             perturb=False,
                                             sample_rand=imle_configs.imle_sample_rand)

    def get_aux_loss(self, logits: torch.Tensor):
        """
        Aux loss that the sampled masks should be different

        :param logits:
        :return:
        """
        logits = logits / torch.linalg.norm(logits, ord=None, dim=0, keepdim=True)
        eye = 1 - torch.eye(logits.shape[1], device=logits.device)
        loss = ((logits.t() @ logits) * eye).mean()
        return loss * self.aux_loss_weight

    def emb_model_forward(self, data: Union[Data, Batch], emb_model: Emb_model, train: bool) \
            -> Tuple[Union[Data, Batch], Optional[torch.FloatType]]:
        """
        Common forward propagation for train and val, only called when embedding model is trained.

        :param data:
        :param emb_model:
        :param train:
        :return:
        """
        logits_n, logits_e = emb_model(data)

        if self.imle_sample_policy in ['node', 'khop_subgraph', 'khop_global', 'greedy_exp', 'or', 'or_optim']:
            split_idx = get_split_idx(data.ptr)
            logits = logits_n
            subgraphs_from_mask = edgemasked_graphs_from_nodemask
        elif self.imle_sample_policy in ['edge', 'mst']:
            split_idx = get_split_idx(data._slice_dict['edge_index'])
            logits = logits_e
            subgraphs_from_mask = edgemasked_graphs_from_edgemask
        else:
            raise NotImplementedError

        graphs = Batch.to_data_list(data)

        self.imle_scheduler.graphs = graphs
        self.imle_scheduler.ptr = split_idx

        aux_loss = None
        if train:
            @imle(target_distribution=self.target_distribution,
                  noise_distribution=self.noise_distribution,
                  input_noise_temperature=self.temp,
                  target_noise_temperature=self.temp,
                  nb_samples=1)
            def imle_sample_scheme(logits: torch.Tensor):
                return self.imle_scheduler.torch_sample_scheme(logits)

            sample_idx = imle_sample_scheme(logits)
            if self.aux_loss_weight > 0:
                aux_loss = self.get_aux_loss(sample_idx)

            sample_idx = torch.split(sample_idx, split_idx, dim=0)
            self.noise_distribution.scale = self.noise_scale_scheduler()

        else:
            sample_idx = self.imle_scheduler.torch_sample_scheme(logits)

        list_list_subgraphs, edge_weights, selected_node_masks = zip(
            *[subgraphs_from_mask(g, i.T,
                                  grad=train,
                                  remove_node=self.remove_node,
                                  add_full_graph=self.add_full_graph) for g, i in
              zip(graphs, sample_idx)])
        list_subgraphs = list(itertools.chain.from_iterable(list_list_subgraphs))
        data = construct_subgraph_batch(list_subgraphs,
                                        [len(g_list) for g_list in list_list_subgraphs],
                                        edge_weights,
                                        selected_node_masks,
                                        self.device)

        return data, aux_loss

    def train(self,
              dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
              emb_model: Emb_model,
              model: Train_model):

        if emb_model is not None:
            emb_model.train()
            self.imle_scheduler.return_list = False
            self.imle_scheduler.perturb = False
            self.imle_scheduler.sample_rand = self.imle_sample_rand
            self.optimizer_embd.zero_grad()

        model.train()
        train_losses = torch.tensor(0., device=self.device)
        if self.task_type == 'classification':
            preds = []
            labels = []
        num_graphs = 0

        for batch_id, data in enumerate(dataloader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            aux_loss = None
            if emb_model is not None:
                data, aux_loss = self.emb_model_forward(data, emb_model, train=True)

            pred = model(data)
            loss = self.criterion(pred, data.y.to(torch.float))
            if aux_loss is not None:
                loss += aux_loss

            loss.backward()
            self.optimizer.step()
            if self.optimizer_embd is not None:
                if (batch_id % self.micro_batch_embd == self.micro_batch_embd - 1) or (batch_id >= len(dataloader) - 1):
                    emb_model = scale_grad(emb_model, (batch_id % self.micro_batch_embd) + 1)
                    self.optimizer_embd.step()
                    self.optimizer_embd.zero_grad()

            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
            if self.task_type == 'classification':
                preds.append(pred)
                labels.append(data.y)

        if self.task_type == 'classification':
            preds = torch.cat(preds, dim=0) > 0.
            labels = torch.cat(labels, dim=0)
            train_acc = ((preds == labels).sum() / labels.numel()).item()
            self.curves['train_acc'].append(train_acc)
        else:
            train_acc = None

        train_loss = train_losses.item() / num_graphs
        self.curves['train_loss'].append(train_loss)

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return train_loss, train_acc

    @torch.no_grad()
    def inference(self,
                  dataloader: Union[TorchDataLoader, PyGDataLoader, MYDataLoader],
                  emb_model: Emb_model,
                  model: Train_model,
                  test: bool = False):
        if emb_model is not None:
            emb_model.eval()
            self.imle_scheduler.return_list = True
            self.imle_scheduler.perturb = True
            self.imle_scheduler.sample_rand = False  # test time, always take topk, inspite of noise perturbation

        model.eval()
        val_losses = torch.tensor(0., device=self.device)
        if self.task_type == 'classification':
            preds = []
            labels = []
        num_graphs = 0

        for v in range(self.voting):
            for data in dataloader:
                data = data.to(self.device)

                if emb_model is not None:
                    data, _ = self.emb_model_forward(data, emb_model, train=False)

                pred = model(data)
                loss = self.criterion(pred, data.y.to(torch.float))

                val_losses += loss * data.num_graphs
                if self.task_type == 'classification':
                    preds.append(pred)
                    labels.append(data.y)
                num_graphs += data.num_graphs

        if self.task_type == 'classification':
            preds = torch.cat(preds, dim=0) > 0.
            labels = torch.cat(labels, dim=0)
            val_acc = ((preds == labels).sum() / labels.numel()).item()
            if not test:
                self.curves['val_acc'].append(val_acc)
                self.best_val_acc = max(self.best_val_acc, val_acc)
        else:
            val_acc = None

        val_loss = val_losses.item() / num_graphs

        early_stop = False
        if not test:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            if self.scheduler_embd is not None:
                if isinstance(self.scheduler_embd, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler_embd.step(val_loss)
                else:
                    self.scheduler_embd.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience = 0
            else:
                self.patience += 1
                if self.patience > self.max_patience:
                    early_stop = True

            self.curves['val_loss'].append(val_loss)

        if emb_model is not None:
            del self.imle_scheduler.graphs
            del self.imle_scheduler.ptr

        return val_loss, val_acc, early_stop

    def save_curve(self, path):
        pickle.dump(self.curves, open(os.path.join(path, 'curves.pkl'), "wb"))
