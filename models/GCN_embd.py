import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self, edge_features, node_features, emb_dim, update_edge=False):
        super(GCNConv, self).__init__(aggr='add')

        self.update_edge = update_edge

        self.linear = torch.nn.Linear(node_features, emb_dim)
        self.edge_encoder = torch.nn.Linear(edge_features, emb_dim)
        if update_edge:
            self.edge_updating = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_embedding):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_embedding)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        new_x = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
        new_edge_attr = self.edge_updating(new_x[edge_index[0]] + new_x[edge_index[1]]) + edge_embedding \
            if self.update_edge else None
        return new_x, new_edge_attr

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * torch.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGCN(torch.nn.Module):
    def __init__(self, input_dim, edge_features, hid_dim, emb_dim):
        super(NetGCN, self).__init__()

        self.conv1 = GCNConv(edge_features, input_dim, hid_dim, update_edge=False)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        # self.bn_edge1 = torch.nn.BatchNorm1d(hid_dim)

        self.conv2 = GCNConv(edge_features, hid_dim, hid_dim, update_edge=False)
        self.bn2 = torch.nn.BatchNorm1d(hid_dim)
        # self.bn_edge2 = torch.nn.BatchNorm1d(hid_dim)

        self.conv3 = GCNConv(edge_features, hid_dim, hid_dim, update_edge=True)
        self.bn3 = torch.nn.BatchNorm1d(hid_dim)
        self.bn_edge3 = torch.nn.BatchNorm1d(hid_dim)

        self.lin_node = torch.nn.Linear(hid_dim, emb_dim)
        self.lin_edge = torch.nn.Linear(hid_dim, emb_dim)

    def forward(self, data):
        x, edge_attr = data.x, data.edge_attr

        x, _ = self.conv1(x, data.edge_index, edge_attr)
        x = self.bn1(torch.relu(x))
        # edge_attr = self.bn_edge1(torch.relu(edge_attr))
        x, _ = self.conv2(x, data.edge_index, edge_attr)
        x = self.bn2(torch.relu(x))
        # edge_attr = self.bn_edge2(torch.relu(edge_attr))
        x, edge_attr = self.conv3(x, data.edge_index, edge_attr)
        x = self.lin_node(self.bn3(torch.relu(x)))
        edge_attr = self.lin_edge(self.bn_edge3(torch.relu(edge_attr)))

        return x, edge_attr
