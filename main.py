import argparse
import logging
from logging import Logger
import os
from argparse import Namespace
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from models import NetGINE, NetGCN
from training.trainer import Trainer
from data.get_data import get_data
from data.const import DATASET_FEATURE_STAT_DICT


def get_parse() -> Namespace:
    parser = argparse.ArgumentParser(description='GNN baselines')
    parser.add_argument('--model', type=str, default='gine')
    parser.add_argument('--dataset', type=str, default='zinc')
    parser.add_argument('--hid_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=100, help='for early stop')
    parser.add_argument('--lr_patience', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--reg', type=float, default=0.)
    parser.add_argument('--num_convlayers', type=int, default=4)
    parser.add_argument('--gnn_jk', type=str, default=None, choices=[None, 'concat', 'residual'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=100)

    parser.add_argument('--remove_node', action='store_true', help='if true, the nodes are discarded wrt node '
                                                                   'masks, otherwise keep the nodes for forward pass '
                                                                   'and graph pooling. Only works for node-based '
                                                                   'sampling methods. ')
    parser.add_argument('--add_full_graph', action='store_true', help='if true, add the original graph into the batch')

    # I-MLE
    parser.add_argument('--sample_policy', type=str, default='node',
                        choices=['node', 'edge', 'khop_subgraph', 'mst', 'greedy_exp', 'or', 'or_optim'])
    parser.add_argument('--sample_k', type=int, default=-1, help='Instance to be sampled, can be num nodes, num edges '
                                                                 'or k-hop neigbors')
    parser.add_argument('--num_subgraphs', type=int, default=5, help='number of subgraphs to sample for a graph')
    parser.add_argument('--train_embd_model', action='store_true', help='get differentiable logits')
    parser.add_argument('--beta', type=float, default=10.)
    parser.add_argument('--aux_loss_weight', type=float, default=0.)

    # ESAN
    parser.add_argument('--esan_policy', type=str, default='null', choices=['null', 'node_deleted', 'edge_deleted'])
    parser.add_argument('--sample_mode', type=str, default='int', choices=['float', 'int'], help="Only for baselines "
                                                                                                 "e.g. ESAN sampling, "
                                                                                                 "choose subgraphs by "
                                                                                                 "fraction or number "
                                                                                                 "k")
    parser.add_argument('--esan_frac', type=float, default=0.1, help="Only for baselines, see --sample_mode")
    parser.add_argument('--esan_k', type=int, default=3, help="Only for baselines, see --sample_mode")
    parser.add_argument('--voting', type=int, default=5, help="Random sampling for majority")

    parser.add_argument('--debug', action='store_true', help='when debugging, take a small subset of the datasets')

    args = parser.parse_args()
    return args


def get_logger(folder_path: str) -> Logger:
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler(os.path.join(folder_path, 'mylog.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def naming(args: Namespace) -> str:
    name = f'{args.dataset}_' \
           f'hid_{args.hid_size}_' \
           f'lr_{args.lr}_' \
           f'lr_patience_{args.lr_patience}_' \
           f'dp_{args.dropout}_' \
           f'reg_{args.reg}_' \
           f'n_lay_{args.num_convlayers}_' \
           f'bsize_{args.batch_size}_' \
           f'jk_{args.gnn_jk}_'

    if args.esan_policy == 'null':
        if args.num_subgraphs == 0:
            name += 'normal_train_'
        else:
            name += f'policy_{args.sample_policy}_' \
                    f'n_subg_{args.num_subgraphs}_' \
                    f'IMLE_{args.train_embd_model}_' \
                    f'samplek_{args.sample_k}_'
    else:
        name += f'esanpolicy_{args.esan_policy}_' \
                f'esan_{args.esan_frac if args.sample_mode == "float" else args.esan_k}_'

    return name + f'voting_{args.voting}'


if __name__ == '__main__':
    args = get_parse()

    assert ((not args.train_embd_model) or (args.esan_policy == 'null')), "No sampling the original data with I-MLE"
    train_loader, val_loader, test_loader = get_data(args)

    if args.dataset.lower() in ['zinc']:
        task_type = 'regression'
        criterion = torch.nn.L1Loss()
    elif args.dataset.lower() in ['mutag']:
        task_type = 'classification'
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    hparams = naming(args)

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.isdir(os.path.join(args.log_path, hparams)):
        os.mkdir(os.path.join(args.log_path, hparams))
    folder_name = os.path.join(args.log_path, hparams, str(datetime.now()))
    os.mkdir(folder_name)
    writer = SummaryWriter(folder_name)
    logger = get_logger(folder_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if args.model.lower() == 'gine':
        model = NetGINE(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                        DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                        args.hid_size,
                        args.dropout,
                        args.num_convlayers,
                        jk=args.gnn_jk,
                        num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class']).to(device)
    else:
        raise NotImplementedError

    if args.train_embd_model:
        emb_model = NetGCN(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                           DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                           args.hid_size,
                           args.num_subgraphs).to(device)
        train_params = list(emb_model.parameters()) + list(model.parameters())
    else:
        emb_model = None
        train_params = model.parameters()

    optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.reg)
    trainer = Trainer(task_type=task_type,
                      imle_sample_policy=args.sample_policy,
                      aux_loss_weight=args.aux_loss_weight,
                      sample_k=args.sample_k,
                      remove_node=args.remove_node,
                      add_full_graph=args.add_full_graph,
                      voting=args.voting,
                      max_patience=args.patience,
                      optimizer=optimizer,
                      scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=0.316227766,
                                                                           patience=args.lr_patience,
                                                                           min_lr=1e-5),
                      criterion=criterion,
                      train_embd_model=args.train_embd_model,
                      beta=args.beta,
                      device=device)

    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train(train_loader, emb_model, model)
        val_loss, val_acc, early_stop = trainer.validation(val_loader, emb_model, model)

        if early_stop:
            logger.info('early stopping')
            break

        logger.info(f'epoch: {epoch}, '
                    f'training loss: {train_loss}, '
                    f'val loss: {val_loss}, '
                    f'patience: {trainer.patience}, '
                    f'training acc: {train_acc}, '
                    f'val acc: {val_acc}')
        writer.add_scalar('loss/training loss', train_loss, epoch)
        writer.add_scalar('loss/val loss', val_loss, epoch)
        if train_acc is not None:
            writer.add_scalar('acc/training acc', train_acc, epoch)
        if val_acc is not None:
            writer.add_scalar('acc/valacc', val_acc, epoch)
        writer.add_scalar('lr', trainer.scheduler.optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), f'{folder_name}/model{epoch}.pt')
            if args.train_embd_model:
                torch.save(emb_model.state_dict(), f'{folder_name}/embd_model{epoch}.pt')

    logger.info(f'Best val loss: {trainer.best_val_loss}')
    logger.info(f'Best val acc: {trainer.best_val_acc}')
    trainer.save_curve(folder_name)
    if args.train_embd_model:
        torch.save(emb_model.state_dict(), f'{folder_name}/embd_model_final.pt')
    torch.save(model.state_dict(), f'{folder_name}/model_final.pt')
