import argparse
from argparse import Namespace
import os
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset

from custom_dataloder import MYDataLoader
from models import NetGINE, NetGCN
from train import train


def get_parse() -> Namespace:
    parser = argparse.ArgumentParser(description='GNN baselines')
    parser.add_argument('--model', type=str, default='gine')
    parser.add_argument('--dataset', type=str, default='zinc')
    parser.add_argument('--hid_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--sample_k', type=int, default=15, help='top-k nodes, i.e. n_nodes of each subgraph')
    parser.add_argument('--num_subgraphs', type=int, default=3, help='number of subgraphs to sample for a graph')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--log_path', type=str, default='./logs')

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
    writer = SummaryWriter(args.log_path)

    # TODO: use full indices
    with open(os.path.join(args.data_path, 'indices', 'train_indices.pkl'), 'rb') as handle:
        train_indices = pickle.load(handle)[:32]

    with open(os.path.join(args.data_path, 'indices', 'test_indices.pkl'), 'rb') as handle:
        test_indices = pickle.load(handle)[:32]

    with open(os.path.join(args.data_path, 'indices', 'val_indices.pkl'), 'rb') as handle:
        val_indices = pickle.load(handle)[:32]

    train_loader = MYDataLoader(dataset[:220011][train_indices], batch_size=args.batch_size, shuffle=True,
                                n_subgraphs=0)
    test_loader = MYDataLoader(dataset[220011:225011][test_indices], batch_size=args.batch_size, shuffle=True,
                               n_subgraphs=0)
    val_loader = MYDataLoader(dataset[225011:][val_indices], batch_size=args.batch_size, shuffle=True, n_subgraphs=0)

    if args.model.lower() == 'gine':
        model = NetGINE(args.hid_size).to(device)
    else:
        raise NotImplementedError

    emb_model = NetGCN(28, args.hid_size, args.num_subgraphs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()

    for epoch in range(args.epochs):
        train_loss = train(args.sample_k, train_loader, emb_model, model, optimizer, criterion, device)

        # TODO: implement val function
        model.eval()
        emb_model.eval()
        val_losses = 0.
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data)
                loss = criterion(pred, data.y)
                val_losses += loss.item() * data.num_graphs

        print(f'epoch: {epoch}, '
              f'training loss: {train_loss}, '
              f'val loss: {val_losses / len(val_loader.dataset)}')
        writer.add_scalar('loss/training loss', train_loss, epoch)
        writer.add_scalar('loss/val loss', val_losses / len(val_loader.dataset), epoch)
