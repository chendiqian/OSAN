from typing import List, Optional, Union, Tuple
from collections import namedtuple

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, Dataset, HeteroData
from torch_geometric.utils import subgraph

from grad_utils import Nodemask2Edgemask, nodemask2edgemask

SubgraphSetBatch = namedtuple(
    'SubgraphSetBatch', [
        'x',
        'edge_index',
        'edge_attr',
        'edge_weight',
        'y',
        'batch',
        'inter_graph_idx',
        'ptr',
        'num_graphs',
    ])


def rand_sampling(graph: Data,
                  n_subgraphs: int,
                  n_drop_nodes: int = 1) -> Tuple[List[Data], int]:
    """
    Sample subgraphs.
    TODO: replace for-loop with functorch.vmap https://pytorch.org/tutorials/prototype/vmap_recipe.html?highlight=vmap


    :param graph:
    :param n_subgraphs:
    :param n_drop_nodes:
    :return:
        A list of graphs and their index masks
    """

    n_nodes = graph.x.shape[0]
    res_list = []
    batch_num_nodes = 0
    for i in range(n_subgraphs):
        indices = torch.randperm(n_nodes)[:-n_drop_nodes]
        sort_indices = torch.sort(indices).values
        batch_num_nodes += len(indices)

        res_list += subgraphs_from_index(graph, sort_indices)

    return res_list, batch_num_nodes


def subgraphs_from_index(graph: Data, indices: Tensor) -> List[Data]:
    """
    Given a graph and a tensor whose rows are indices (The indices are sorted)
    Returns a list of subgraphs indicated by the indices

    :param graph:
    :param indices:
    :return:
    """
    if indices.ndim < 2:
        indices = indices[None]

    graphs = []
    for idx in indices:
        edge_index, edge_attr = subgraph(idx, graph.edge_index, graph.edge_attr, relabel_nodes=True)
        graphs.append(Data(x=graph.x[idx],
                           edge_index=edge_index,
                           edge_attr=edge_attr,
                           y=graph.y))

    return graphs


def subgraphs_from_mask(graph: Data, masks: Tensor) -> List[Data]:
    """
    Return subgraphs containing gradients of mask

    :param graph:
    :param masks: shape(n_Subgraphs, n_nodes)
    :return:
    """
    assert masks.dtype == torch.float
    if masks.ndim < 2:
        masks = masks[None]

    edge_masks = masks[:, graph.edge_index[0]] * masks[:, graph.edge_index[1]]
    idx_masks = masks.detach().to(torch.bool)

    graphs = []
    for i, (m, id_m, em) in enumerate(zip(masks, idx_masks, edge_masks)):
        # relabel edge_index
        edge_attr = graph.edge_attr * em[:, None]
        edge_index, edge_attr = subgraph(id_m, graph.edge_index, edge_attr, relabel_nodes=True)
        # multiply the mask then slice to obtain the gradient
        graphs.append(Data(x=(graph.x * m[:, None])[id_m, :],
                           edge_index=edge_index,
                           edge_attr=edge_attr,
                           y=graph.y))

    return graphs


def edgemasked_graphs_from_nodemask(graph: Data, masks: Tensor, grad=True) -> Tuple[List[Data], Tensor]:
    """
    Create edge_weights which contain the back-propagated gradients

    :param graph:
    :param masks: shape (n_subgraphs, n_node_in_original_graph,) node masks
    :param grad: whether to contain gradient info
    :return:
    """
    transform_func = Nodemask2Edgemask.apply if grad else nodemask2edgemask
    edge_weights = transform_func(masks, graph.edge_index, torch.tensor(graph.num_nodes, device=graph.x.device))
    edge_weights = edge_weights.reshape(-1)
    graphs = [graph] * masks.shape[0]
    return graphs, edge_weights


def construct_subgraph_batch(graph_list: List[Data], sample_node_idx, edge_weights, device):
    # new batch
    batch = Batch.from_data_list(graph_list, None, None)
    original_graph_mask = torch.cat([torch.full((idx.shape[1],), i, device=device)
                                     for i, idx in enumerate(sample_node_idx)], dim=0)
    ptr = torch.cat((torch.tensor([0], device=device),
                     (original_graph_mask[1:] > original_graph_mask[:-1]).nonzero().reshape(-1) + 1,
                     torch.tensor([len(original_graph_mask)], device=device)), dim=0)

    return SubgraphSetBatch(x=batch.x,
                            edge_index=batch.edge_index[torch.LongTensor([1, 0]), :],  # flip the direction of message
                            edge_attr=batch.edge_attr,
                            edge_weight=torch.cat(edge_weights, dim=0),
                            y=batch.y[ptr[:-1]],
                            batch=batch.batch,
                            inter_graph_idx=original_graph_mask,
                            ptr=ptr,
                            num_graphs=batch.num_graphs)
