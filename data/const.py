from torch import nn


DATASET_FEATURE_STAT_DICT = {
    'zinc': {'node': 28, 'edge': 3, 'num_class': 1},  # regression
    'mutag': {'node': 7, 'edge': 4, 'num_class': 1},  # bin classification
    'alchemy': {'node': 6, 'edge': 4, 'num_class': 12},  # regression, but 12 labels
    'ogbg-molesol': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'ogbg-molbace': {'node': 9, 'edge': 3, 'num_class': 1},  # bin classification
    'ogbg-molhiv': {'node': 9, 'edge': 3, 'num_class': 1},  # regression
    'qm9': {'node': 11, 'edge': 5, 'num_class': 12},  # regression, 12 labels
    'exp': {'node': 1, 'edge': 0, 'num_class': 1},  # bin classification
}

TASK_TYPE_DICT = {
    'zinc': 'regression',
    'alchemy': 'regression',
    'ogbg-molesol': 'regression',
    'ogbg-molbace': 'rocauc',
    'ogbg-molhiv': 'rocauc',
    'qm9': 'regression',
    'exp': 'acc',
}

CRITERION_DICT = {
    'zinc': nn.L1Loss(),
    'alchemy': nn.L1Loss(),
    'ogbg-molesol': nn.MSELoss(),
    'ogbg-molbace': nn.BCEWithLogitsLoss(),
    'ogbg-molhiv': nn.BCEWithLogitsLoss(),
    'qm9': nn.L1Loss(),
    'exp': nn.BCEWithLogitsLoss(),
}