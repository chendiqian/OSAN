from typing import List, Optional, Union, Tuple, Sequence

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph, is_undirected, to_undirected

from grad_utils import Nodemask2Edgemask, nodemask2edgemask
from data import SubgraphSetBatch


def node_rand_sampling(graph: Data,
                       n_subgraphs: int,
                       node_per_subgraph: int = -1) -> List[Data]:
    """
    Sample subgraphs.
    TODO: replace for-loop with functorch.vmap https://pytorch.org/tutorials/prototype/vmap_recipe.html?highlight=vmap


    :param graph:
    :param n_subgraphs:
    :param node_per_subgraph:
    :return:
        A list of graphs and their index masks
    """
    n_nodes = graph.num_nodes

    if node_per_subgraph < 0:  # drop nodes
        node_per_subgraph += n_nodes

    graphs = []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_nodes)[:node_per_subgraph]
        sort_indices = torch.sort(indices).values
        graphs.append(nodesubset_to_subgraph(graph, sort_indices, relabel=False))

    return graphs


def edge_rand_sampling(graph: Data, n_subgraphs: int, edge_per_subgraph: int = -1) -> List[Data]:
    n_edge = graph.num_edges
    if n_edge == 0:
        return [graph for _ in range(n_subgraphs)]

    if edge_per_subgraph < 0:   # drop edges
        edge_per_subgraph += n_edge

    edge_index, edge_attr, undirected = edge_sample_preproc(graph)

    graphs = []
    for i in range(n_subgraphs):
        indices = torch.randperm(n_edge)[:edge_per_subgraph]
        sort_indices = torch.sort(indices).values

        subgraph_edge_index = edge_index[:, sort_indices]
        subgraph_edge_attr = edge_attr[sort_indices, :] if edge_attr is not None else None

        if undirected:
            if subgraph_edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                        num_nodes=graph.num_nodes)
            else:
                subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                    num_nodes=graph.num_nodes)

        graphs.append(Data(
                    x=graph.x,
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    num_nodes=graph.num_nodes,
                    y=graph.y,
                ))

    return graphs


def edge_sample_preproc(data: Data) -> Tuple[LongTensor, Tensor, bool]:
    """
    If undirected, return the non-duplicate directed edges for sampling

    :param data:
    :return:
    """
    if data.edge_attr is not None and data.edge_attr.ndim == 1:
        edge_attr = data.edge_attr.unsqueeze(-1)
    else:
        edge_attr = data.edge_attr

    undirected = is_undirected(data.edge_index, edge_attr, data.num_nodes)

    if undirected:
        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = edge_attr[keep_edge, :] if edge_attr is not None else edge_attr
    else:
        edge_index = data.edge_index

    return edge_index, edge_attr, undirected


def nodesubset_to_subgraph(graph: Data, subset: Tensor, relabel=False) -> Data:
    edge_index, edge_attr = subgraph(subset, graph.edge_index, graph.edge_attr,
                                     relabel_nodes=relabel, num_nodes=graph.num_nodes)

    x = graph.x[subset] if relabel else graph.x
    num_nodes = subset.numel() if relabel else graph.num_nodes
    return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                y=graph.y)


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


def get_ptr(graph_idx: Tensor, device: torch.device) -> Tensor:
    """
    Given indices of graph, return ptr
    e.g. [1,1,2,2,2,3,3,4] -> [0, 2, 5, 7]

    :param graph_idx:
    :param device:
    :return:
    """
    return torch.cat((torch.tensor([0], device=device),
                      (graph_idx[1:] > graph_idx[:-1]).nonzero().reshape(-1) + 1,
                      torch.tensor([len(graph_idx)], device=device)), dim=0)


def construct_subgraph_batch(graph_list: List[Data],
                             num_subgraphs: List[int],
                             edge_weights: Optional[Sequence[Tensor]] = None,
                             device: torch.device = 'cpu'):
    """

    :param graph_list: a list of [subgraph1_1, subgraph1_2, subgraph1_3, subgraph2_1, ...]
    :param num_subgraphs: a list number of subgraphs
    :param edge_weights:
    :param device:
    :return:
    """
    # new batch
    batch = Batch.from_data_list(graph_list, None, None)
    original_graph_mask = torch.cat([torch.full((n_subg,), i, device=device)
                                     for i, n_subg in enumerate(num_subgraphs)], dim=0)
    ptr = get_ptr(original_graph_mask, device)

    # TODO: duplicate labels or aggregate the embeddings for original labels? potential problem: cannot split the
    #  batch because y shape inconsistent:
    #  https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Batch.to_data_list.
    #  need to check `batch._slice_dict` and `batch._inc_dict`

    return SubgraphSetBatch(x=batch.x,
                            edge_index=batch.edge_index[torch.LongTensor([1, 0]), :],  # flip the direction of message
                            edge_attr=batch.edge_attr,
                            edge_weight=torch.cat(edge_weights, dim=0) if isinstance(edge_weights, (list, tuple))
                            else edge_weights,
                            y=batch.y[ptr[:-1]],
                            batch=batch.batch,
                            inter_graph_idx=original_graph_mask,
                            ptr=ptr,
                            num_graphs=batch.num_graphs)
