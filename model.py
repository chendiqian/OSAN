import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, MessagePassing


class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return torch.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGINE(torch.nn.Module):
    def __init__(self, dim):
        super(NetGINE, self).__init__()

        num_features = 3
        dim = dim

        self.conv1 = GINConv(num_features, 28, 256)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GINConv(num_features, 256, 256)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GINConv(num_features, 256, 256)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GINConv(num_features, 256, 256)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(4 * dim, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, 1)

    def forward(self, data):
        x = data.x

        x_1 = torch.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x_1 = self.bn1(x_1)
        x_2 = torch.relu(self.conv2(x_1, data.edge_index, data.edge_attr))
        x_2 = self.bn2(x_2)
        x_3 = torch.relu(self.conv3(x_2, data.edge_index, data.edge_attr))
        x_3 = self.bn3(x_3)
        x_4 = torch.relu(self.conv3(x_3, data.edge_index, data.edge_attr))
        x_4 = self.bn4(x_4)

        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_mean_pool(x, data.batch)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # only sample during training
        if self.training:
            x = global_mean_pool(x, data.inter_graph_idx)
        x = self.fc4(x)
        return x.view(-1)
