from typing import Tuple
import logging
import os
from datetime import datetime
import yaml
from ml_collections import ConfigDict
from sacred import Experiment

import torch
from torch.utils.tensorboard import SummaryWriter
from numpy import mean as np_mean
from numpy import std as np_std

from models import NetGINE, NetGCN, NetGINEAlchemy, OGBGNN
from training.trainer import Trainer
from data.get_data import get_data, get_ogb_data, get_qm9
from data.const import DATASET_FEATURE_STAT_DICT

ex = Experiment()


def get_logger(folder_path: str) -> logging.Logger:
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler(os.path.join(folder_path, 'training_logs.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def naming(args) -> str:
    name = f'{args.dataset}_{args.model}_'

    if args.imle_configs is not None:
        name += 'IMLE_'
    elif args.sample_configs.sample_with_esan:
        name += 'ESAN_'
    elif args.sample_configs.num_subgraphs == 0:
        name += 'normal_train_'
    else:
        name += 'OnTheFly_'

    name += f'policy_{args.sample_configs.sample_policy}_'
    name += f'samplek_{args.sample_configs.sample_k}_'
    name += f'subg_{args.sample_configs.num_subgraphs}_'
    name += f'rm_node_{args.sample_configs.remove_node}_'
    name += f'fullg_{args.sample_configs.add_full_graph}'

    return name


def prepare_exp(folder_name: str) -> Tuple[SummaryWriter, str]:
    run_folder = os.path.join(folder_name, str(datetime.now()))
    os.mkdir(run_folder)
    writer = SummaryWriter(run_folder)
    return writer, run_folder


@ex.automain
def run(fixed):
    with open(f"./configs/{fixed['dataset'].lower()}/common_configs.yaml", 'r') as stream:
        try:
            common_configs = yaml.safe_load(stream)
            common_configs = common_configs['common']
        except yaml.YAMLError as exc:
            print(exc)

    common_configs.update(fixed)
    args = ConfigDict(common_configs)
    hparams = naming(args)

    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.isdir(os.path.join(args.log_path, hparams)):
        os.mkdir(os.path.join(args.log_path, hparams))
    folder_name = os.path.join(args.log_path, hparams)

    with open(os.path.join(folder_name, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    logger = get_logger(folder_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tasks = None
    if 'ogb' in args.dataset.lower():
        train_loader, val_loader, test_loader, num_tasks = get_ogb_data(args)
    elif args.dataset == 'QM9':
        train_loader, val_loader, test_loader = get_qm9(args, device)
    else:
        train_loader, val_loader, test_loader = get_data(args, device)

    if args.dataset.lower() in ['zinc', 'alchemy']:
        task_type = 'regression'
        criterion = torch.nn.L1Loss()
    elif args.dataset.lower() in ['ogbg-molesol']:
        task_type = 'regression'
        criterion = torch.nn.MSELoss()
    elif args.dataset.lower() in ['ogbg-molbace']:
        task_type = 'rocauc'
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.dataset == 'QM9':
        task_type = 'regression'
        criterion = torch.nn.L1Loss()
    else:
        raise NotImplementedError

    if args.model.lower() == 'gine':
        model = NetGINE(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                        DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                        args.hid_size,
                        args.dropout,
                        args.num_convlayers,
                        jk=args.gnn_jk,
                        num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class']).to(device)
    elif args.model.lower() == 'gine_alchemy':
        model = NetGINEAlchemy(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                               DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                               args.hid_size,
                               num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                               num_layers=args.num_convlayers).to(device)
    elif args.model.lower() == 'gin-virtual':
        model = OGBGNN(gnn_type='gin',
                       num_tasks=num_tasks,
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=True).to(device)
    elif args.model.lower() == 'ogb_gin':
        model = OGBGNN(gnn_type='gin',
                       num_tasks=num_tasks,
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=False).to(device)
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        emb_model = NetGCN(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                           DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                           args.hid_size,
                           args.sample_configs.num_subgraphs,
                           normalize=args.imle_configs.norm_logits,
                           encoder='ogb' in args.dataset.lower()).to(device)
    else:
        emb_model = None

    trainer = Trainer(task_type=task_type,
                      voting=args.voting,
                      max_patience=args.patience,
                      criterion=criterion,
                      device=device,
                      imle_configs=args.imle_configs,
                      **args.sample_configs)

    best_val_losses = []
    test_losses = []
    best_val_metrics = []
    test_metrics = []

    for _run in range(args.num_runs):
        if emb_model is not None:
            emb_model.reset_parameters()
            optimizer_embd = torch.optim.Adam(emb_model.parameters(),
                                              lr=args.imle_configs.embd_lr,
                                              weight_decay=args.imle_configs.reg_embd)
            scheduler_embd = None
        else:
            optimizer_embd = None
            scheduler_embd = None
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1 ** 0.5)
        writer, run_folder = prepare_exp(folder_name)

        best_epoch = 0
        for epoch in range(args.max_epochs):
            train_loss, train_metric = trainer.train(train_loader,
                                                     emb_model,
                                                     model,
                                                     optimizer_embd,
                                                     optimizer)
            val_loss, val_metric, early_stop = trainer.inference(val_loader,
                                                                 emb_model,
                                                                 model,
                                                                 scheduler_embd,
                                                                 scheduler,
                                                                 test=False)

            if epoch > args.min_epochs and early_stop:
                logger.info('early stopping')
                break

            logger.info(f'epoch: {epoch}, '
                        f'training loss: {train_loss}, '
                        f'val loss: {val_loss}, '
                        f'patience: {trainer.patience}, '
                        f'training metric: {train_metric}, '
                        f'val metric: {val_metric}, '
                        f'lr: {scheduler.optimizer.param_groups[0]["lr"]}')
            writer.add_scalar('loss/training loss', train_loss, epoch)
            writer.add_scalar('loss/val loss', val_loss, epoch)
            writer.add_scalar('metric/training metric', train_metric, epoch)
            writer.add_scalar('metric/val metric', val_metric, epoch)
            writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], epoch)

            if trainer.patience == 0:
                best_epoch = epoch
                torch.save(model.state_dict(), f'{run_folder}/model_best.pt')
                if emb_model is not None:
                    torch.save(emb_model.state_dict(), f'{run_folder}/embd_model_best.pt')

        writer.flush()
        writer.close()

        model.load_state_dict(torch.load(f'{run_folder}/model_best.pt'))
        logger.info(f'loaded best model at epoch {best_epoch}')
        if emb_model is not None:
            emb_model.load_state_dict(torch.load(f'{run_folder}/embd_model_best.pt'))

        test_loss, test_metric, _ = trainer.inference(test_loader, emb_model, model, test=True)
        logger.info(f'Best val loss: {trainer.best_val_loss}')
        logger.info(f'Best val metric: {trainer.best_val_metric}')
        logger.info(f'test loss: {test_loss}')
        logger.info(f'test metric: {test_metric}')
        logger.info(f'max_memory_allocated: {torch.cuda.max_memory_allocated()}')
        logger.info(f'memory_allocated: {torch.cuda.memory_allocated()}')

        best_val_losses.append(trainer.best_val_loss)
        test_losses.append(test_loss)
        best_val_metrics.append(trainer.best_val_metric)
        test_metrics.append(test_metric)

        trainer.save_curve(run_folder)
        trainer.clear_stats()

    results = {'best_val_losses': best_val_losses,
               'test_losses': test_losses,
               'best_val_metrics': best_val_metrics,
               'test_metrics': test_metrics,
               'val_loss_stats': f'mean: {np_mean(best_val_losses)}, std: {np_std(best_val_losses)}',
               'test_loss_stats': f'mean: {np_mean(test_losses)}, std: {np_std(test_losses)}',
               'val_metrics_stats': f'mean: {np_mean(best_val_metrics)}, std: {np_std(best_val_metrics)}',
               'test_metrics_stats': f'mean: {np_mean(test_metrics)}, std: {np_std(test_metrics)}'}

    with open(os.path.join(folder_name, 'results.txt'), 'wt') as f:
        f.write(str(results))
