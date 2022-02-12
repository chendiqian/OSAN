import torch
from torch.nn import Sequential, Linear, ReLU
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
    def __init__(self, input_dim, hid_dim, emb_dim):
        super(NetGCN, self).__init__()
        num_features = 3
        self.conv1 = GCNConv(num_features, input_dim, hid_dim)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)

        self.conv2 = GCNConv(num_features, hid_dim, hid_dim)
        self.bn2 = torch.nn.BatchNorm1d(hid_dim)

        self.conv3 = GCNConv(num_features, hid_dim, hid_dim)
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


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2, use_bias=False):
        super(GINEConv, self).__init__(aggr="add")

        # disable the bias, otherwise the information will be nonzero
        self.bond_encoder = Sequential(Linear(emb_dim, dim1, bias=use_bias), ReLU(), Linear(dim1, dim1, bias=use_bias))
        self.mlp = Sequential(Linear(dim1, dim1, bias=use_bias), ReLU(), Linear(dim1, dim2, bias=use_bias))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if edge_weight.ndim < 2:
            edge_weight = edge_weight[:, None]

        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight))

        return out

    def message(self, x_j, edge_attr, edge_weight):
        # x_j has shape [E, out_channels]
        return torch.relu(x_j + edge_attr) * edge_weight

    def update(self, aggr_out):
        return aggr_out


class NetGINE(torch.nn.Module):
    def __init__(self, dim, dropout, num_convlayers, use_bias=True):
        super(NetGINE, self).__init__()

        self.dropout = dropout
        num_features = 3
        assert num_convlayers > 1

        self.conv = torch.nn.ModuleList([GINEConv(num_features, 28, dim, use_bias)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim)])

        for _ in range(num_convlayers - 1):
            self.conv.append(GINEConv(num_features, dim, dim, use_bias))
            self.bn.append(torch.nn.BatchNorm1d(dim))

        self.fc1 = Linear(num_convlayers * dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.fc2 = Linear(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.fc3 = Linear(dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.shape[0], dtype=torch.long)

        intermediate_x = []
        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            x = conv(x, edge_index, edge_attr, edge_weight)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            intermediate_x.append(x)

        x = torch.cat(intermediate_x, dim=-1)
        x = global_mean_pool(x, batch)

        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc3(x)
        return x.view(-1)
