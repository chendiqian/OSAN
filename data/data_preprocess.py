import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, coalesce

from subgraph.khop_subgraph import parallel_khop_neighbor


class GraphToUndirected:
    """
    Wrapper of to_undirected:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=undirected#torch_geometric.utils.to_undirected
    """

    def __call__(self, graph: Data):
        if graph.edge_attr is not None:
            edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
        else:
            edge_index = to_undirected(graph.edge_index, graph.edge_attr, graph.num_nodes)
            edge_attr = None
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


class GraphCoalesce:

    def __call__(self, graph: Data):
        if graph.edge_attr is None:
            edge_index = coalesce(graph.edge_index, None, num_nodes=graph.num_nodes)
            edge_attr = None
        else:
            edge_index, edge_attr = coalesce(graph.edge_index, graph.edge_attr, graph.num_nodes)
        return Data(x=graph.x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph.y,
                    num_nodes=graph.num_nodes)


def khop_data_process(dataset: InMemoryDataset, khop: int):
    """
    Add attribute of khop subgraph indices, so that during sampling they don't need to be re-calculcated.

    :param dataset:
    :param khop:
    :return:
    """
    max_node = 0
    for g in dataset:
        max_node = max(max_node, g.num_nodes)

    for i, g in zip(dataset._indices, dataset):
        mask = parallel_khop_neighbor(g.edge_index.cpu().numpy(), g.num_nodes, max_node, khop)
        dataset._data_list[i].khop_idx = torch.from_numpy(mask).to(torch.float32)
    return dataset
