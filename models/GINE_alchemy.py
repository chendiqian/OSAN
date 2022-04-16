import torch
from torch_geometric.nn import Set2Set, global_mean_pool

from .GINE_gnn import GINEConv


class NetGINEAlchemy(torch.nn.Module):
    def __init__(self, input_dims, edge_features, dim, num_class):
        super(NetGINEAlchemy, self).__init__()

        self.conv = torch.nn.ModuleList([GINEConv(edge_features, input_dims, dim)])
        for _ in range(5):
            self.conv.append(GINEConv(edge_features, dim, dim))

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc4 = torch.nn.Linear(dim, num_class)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.shape[0], dtype=torch.long)

        for conv in self.conv:
            x = torch.relu(conv(x, edge_index, edge_attr, edge_weight))

        x = self.set2set(x, batch)
        x = torch.relu(self.fc1(x))
        if hasattr(data, 'inter_graph_idx') and data.inter_graph_idx is not None:
            x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc4(x)
        if x.shape[-1] == 1:
            x = x.reshape(-1)
        return x
