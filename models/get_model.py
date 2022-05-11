from .esan_zinc_model import ZincAtomEncoder, GNN, DSnetwork
from .GINE_gnn import NetGINE
from .GCN_embd import NetGCN
from .GINE_alchemy import NetGINEAlchemy
from .ogb_mol_gnn import OGBGNN
from .GINE_qm9 import NetGINE_QM
from data.const import DATASET_FEATURE_STAT_DICT


def get_model(args):
    if args.model.lower() == 'gine':
        model = NetGINE(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                        DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                        args.hid_size,
                        args.dropout,
                        args.num_convlayers,
                        jk=args.gnn_jk,
                        num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'gine_alchemy':
        model = NetGINEAlchemy(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                               DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                               args.hid_size,
                               num_class=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                               num_layers=args.num_convlayers)
    elif args.model.lower() == 'gin-virtual':
        model = OGBGNN(gnn_type='gin',
                       num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=True)
    elif args.model.lower() == 'ogb_gin':
        model = OGBGNN(gnn_type='gin',
                       num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=False)
    elif args.model.lower() == 'ogb_originalgin':
        model = OGBGNN(gnn_type='originalgin',
                       num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                       num_layer=args.num_convlayers,
                       emb_dim=args.hid_size,
                       drop_ratio=args.dropout,
                       virtual_node=False)
    elif args.model.lower() == 'gine_qm9':
        model = NetGINE_QM(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                           DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                           args.hid_size,
                           args.num_convlayers,
                           DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'])
    elif args.model.lower() == 'zincgin':   # ESAN's model
        subgraph_gnn = GNN(gnn_type=args.model, num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                           num_layer=args.num_convlayers, in_dim=args.hid_size,
                           emb_dim=args.hid_size, drop_ratio=args.dropout, JK=args.jk,
                           graph_pooling='mean', feature_encoder=ZincAtomEncoder(policy=None, emb_dim=args.hid_size)
                           )
        model = DSnetwork(subgraph_gnn=subgraph_gnn,
                          channels=args.channels,
                          num_tasks=DATASET_FEATURE_STAT_DICT[args.dataset]['num_class'],
                          invariant=args.dataset.lower() == 'zinc')
    else:
        raise NotImplementedError

    if args.imle_configs is not None:
        emb_model = NetGCN(DATASET_FEATURE_STAT_DICT[args.dataset]['node'],
                           DATASET_FEATURE_STAT_DICT[args.dataset]['edge'],
                           args.hid_size,
                           args.sample_configs.num_subgraphs,
                           normalize=args.imle_configs.norm_logits,
                           encoder='ogb' in args.dataset.lower() or 'exp' in args.dataset.lower()).to(device)
    else:
        emb_model = None

    return model, emb_model
