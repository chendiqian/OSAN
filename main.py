import argparse
import logging
from logging import Logger
import os
from argparse import Namespace
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset

from custom_dataloder import MYDataLoader
from models import NetGINE, NetGCN
from train import train, validation


def get_parse() -> Namespace:
    parser = argparse.ArgumentParser(description='GNN baselines')
    parser.add_argument('--model', type=str, default='gine')
    parser.add_argument('--dataset', type=str, default='zinc')
    parser.add_argument('--hid_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=50, help='for early stop')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--reg', type=float, default=0.)
    parser.add_argument('--num_convlayers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--sample_k', type=int, default=30, help='top-k nodes, i.e. n_nodes of each subgraph')
    parser.add_argument('--num_subgraphs', type=int, default=3, help='number of subgraphs to sample for a graph')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--train_embd_model', action='store_true', help='train the embedding model to get '
                                                                        'differentiable logits, otherwise randomly '
                                                                        'select') 

    return parser.parse_args()


def get_logger(folder_path: str) -> Logger:
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler(os.path.join(folder_path, 'mylog.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    args = get_parse()
    
    if args.dataset.lower() == 'zinc':
        if not os.path.isdir(args.data_path):
            os.mkdir(args.data_path)
        dataset = TUDataset(args.data_path, name="ZINC_full")
    else:
        raise NotImplementedError

    hparams = f'hid_{args.hid_size}_' \
              f'dp_{args.dropout}_' \
              f'reg_{args.reg}_' \
              f'n_lay_{args.num_convlayers}_' \
              f'bsize_{args.batch_size}_' \
              f'k_{args.sample_k}_' \
              f'n_subg_{args.num_subgraphs}'

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

    # TODO: use full indices
    # with open(os.path.join(args.data_path, 'indices', 'train_indices.pkl'), 'rb') as handle:
    #     train_indices = pickle.load(handle)[:32]
    #
    # with open(os.path.join(args.data_path, 'indices', 'test_indices.pkl'), 'rb') as handle:
    #     test_indices = pickle.load(handle)[:32]
    #
    # with open(os.path.join(args.data_path, 'indices', 'val_indices.pkl'), 'rb') as handle:
    #     val_indices = pickle.load(handle)[:32]

    # infile = open("./datasets/indices/test.index.txt", "r")
    # for line in infile:
    #     test_indices = line.split(",")
    #     test_indices = [int(i) for i in test_indices]

    infile = open("./datasets/indices/val.index.txt", "r")
    for line in infile:
        val_indices = line.split(",")
        val_indices = [int(i) for i in val_indices[:32]]

    infile = open("./datasets/indices/train.index.txt", "r")
    for line in infile:
        train_indices = line.split(",")
        train_indices = [int(i) for i in train_indices[:32]]

    train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=args.batch_size, shuffle=False,
                                n_subgraphs=0)
    # test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=args.batch_size, shuffle=False,
    #                            n_subgraphs=0)
    val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=args.batch_size, shuffle=False, n_subgraphs=0)

    if args.model.lower() == 'gine':
        model = NetGINE(args.hid_size, args.dropout, args.num_convlayers).to(device)
    else:
        raise NotImplementedError

    if args.train_embd_model:
        emb_model = NetGCN(28, args.hid_size, args.num_subgraphs).to(device)
        train_params = list(emb_model.parameters()) + list(model.parameters())
    else:
        emb_model = None
        train_params = model.parameters()

    optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.reg)
    criterion = torch.nn.L1Loss()
    
    best_val_loss = 1e5
    patience = 0
    for epoch in range(args.epochs):
        train_loss = train(args.sample_k,
                           args.num_subgraphs,
                           train_loader,
                           emb_model,
                           model,
                           optimizer,
                           criterion,
                           device)

        val_loss = validation(args.sample_k,
                              args.num_subgraphs,
                              val_loader,
                              emb_model,
                              model,
                              criterion,
                              device)

        logger.info(f'epoch: {epoch}, '
                    f'training loss: {train_loss}, '
                    f'val loss: {val_loss}, '
                    f'patience: {patience}')
        writer.add_scalar('loss/training loss', train_loss, epoch)
        writer.add_scalar('loss/val loss', val_loss, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                logger.info('early stopping')

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), f'{folder_name}/model{epoch}.pt')
            if args.train_embd_model:
                torch.save(emb_model.state_dict(), f'{folder_name}/embd_model{epoch}.pt')
