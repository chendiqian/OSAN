import torch
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self, num_features, feature_size, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(feature_size, emb_dim)
        self.edge_encoder = torch.nn.Linear(num_features, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * torch.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGCN(torch.nn.Module):
    def __init__(self, input_dim, edge_features, hid_dim, emb_dim):
        super(NetGCN, self).__init__()
        self.conv1 = GCNConv(edge_features, input_dim, hid_dim)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)

        self.conv2 = GCNConv(edge_features, hid_dim, hid_dim)
        self.bn2 = torch.nn.BatchNorm1d(hid_dim)

        self.conv3 = GCNConv(edge_features, hid_dim, hid_dim)
        self.bn3 = torch.nn.BatchNorm1d(hid_dim)

        self.lin = torch.nn.Linear(hid_dim, emb_dim)

    def forward(self, data):
        x = data.x

        x = torch.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x = self.bn1(x)
        x = torch.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = self.bn2(x)
        x = torch.relu(self.conv3(x, data.edge_index, data.edge_attr))
        x = self.bn3(x)

        return self.lin(x)
