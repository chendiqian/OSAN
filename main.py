import argparse
from argparse import Namespace
import os
import pickle
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
    parser.add_argument('--reg', type=float, default=0.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--sample_k', type=int, default=15, help='top-k nodes, i.e. n_nodes of each subgraph')
    parser.add_argument('--num_subgraphs', type=int, default=3, help='number of subgraphs to sample for a graph')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset.lower() == 'zinc':
        if not os.path.isdir(args.data_path):
            os.mkdir(args.data_path)
        dataset = TUDataset(args.data_path, name="ZINC_full")
    else:
        raise NotImplementedError

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    log_path = os.path.join(args.log_path, str(datetime.now()))
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)

    # TODO: use full indices
    # with open(os.path.join(args.data_path, 'indices', 'train_indices.pkl'), 'rb') as handle:
    #     train_indices = pickle.load(handle)[:32]
    #
    # with open(os.path.join(args.data_path, 'indices', 'test_indices.pkl'), 'rb') as handle:
    #     test_indices = pickle.load(handle)[:32]
    #
    # with open(os.path.join(args.data_path, 'indices', 'val_indices.pkl'), 'rb') as handle:
    #     val_indices = pickle.load(handle)[:32]

    infile = open("./datasets/indices/test.index.txt", "r")
    for line in infile:
        test_indices = line.split(",")
        test_indices = [int(i) for i in test_indices]

    infile = open("./datasets/indices/val.index.txt", "r")
    for line in infile:
        val_indices = line.split(",")
        val_indices = [int(i) for i in val_indices]

    infile = open("./datasets/indices/train.index.txt", "r")
    for line in infile:
        train_indices = line.split(",")
        train_indices = [int(i) for i in train_indices]

    train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=args.batch_size, shuffle=False,
                                n_subgraphs=0)
    test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=args.batch_size, shuffle=False,
                               n_subgraphs=0)
    val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=args.batch_size, shuffle=False, n_subgraphs=0)

    if args.model.lower() == 'gine':
        model = NetGINE(args.hid_size).to(device)
    else:
        raise NotImplementedError

    emb_model = NetGCN(28, args.hid_size, args.num_subgraphs).to(device)

    optimizer = torch.optim.Adam(list(emb_model.parameters()) + list(model.parameters()),
                                 lr=args.lr, weight_decay=args.reg)
    criterion = torch.nn.L1Loss()

    for epoch in range(args.epochs):
        train_loss = train(args.sample_k, train_loader, emb_model, model, optimizer, criterion, device)
        val_loss = validation(args.sample_k, val_loader, emb_model, model, criterion, device)

        print(f'epoch: {epoch}, '
              f'training loss: {train_loss}, '
              f'val loss: {val_loss}')
        writer.add_scalar('loss/training loss', train_loss, epoch)
        writer.add_scalar('loss/val loss', val_loss, epoch)

        if epoch % args.save_freq == 0:
            torch.save(emb_model.state_dict(), f'{log_path}/model{epoch}.pt')
