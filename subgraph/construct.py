from typing import Tuple, List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph

from data import SubgraphSetBatch
from data.data_utils import get_ptr
from subgraph.grad_utils import Nodemask2Edgemask, nodemask2edgemask


def nodesubset_to_subgraph(graph: Data, subset: Tensor, relabel=False) -> Data:
    edge_index, edge_attr = subgraph(subset, graph.edge_index, graph.edge_attr,
                                     relabel_nodes=relabel, num_nodes=graph.num_nodes)

    x = graph.x[subset] if relabel else graph.x
    if relabel:
        if subset.dtype in [torch.bool, torch.uint8]:
            num_nodes = subset.sum()
        else:
            num_nodes = subset.numel()
    else:
        num_nodes = graph.num_nodes

    return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                y=graph.y)


def edgemasked_graphs_from_nodemask(graph: Data, masks: Tensor, grad=True, add_full_graph: bool = False) \
        -> Tuple[List[Data], Tensor, Tensor]:
    """
    Create edge_weights which contain the back-propagated gradients

    :param graph:
    :param masks: shape (n_subgraphs, n_node_in_original_graph,) node masks
    :param grad: whether to contain gradient info
    :param add_full_graph:
    :return:
    """
    transform_func = Nodemask2Edgemask.apply if grad else nodemask2edgemask
    edge_weights = transform_func(masks, graph.edge_index, torch.tensor(graph.num_nodes, device=graph.x.device))
    graphs = [graph] * masks.shape[0]
    if add_full_graph:
        graphs.append(graph)
        edge_weights = torch.cat((edge_weights,
                                  torch.ones(1, edge_weights.shape[1],
                                             dtype=edge_weights.dtype,
                                             device=edge_weights.device)), dim=0)
        masks = torch.cat((masks,
                           torch.ones(1, masks.shape[1],
                                      dtype=masks.dtype,
                                      device=masks.device)), dim=0)
    edge_weights = edge_weights.reshape(-1)
    selected_node_masks = masks.reshape(-1).to(torch.bool)
    return graphs, edge_weights, selected_node_masks


def edgemasked_graphs_from_edgemask(graph: Data, masks: Tensor, grad: bool = True, add_full_graph: bool = False) \
        -> Tuple[List[Data], Tensor, Optional[Tensor]]:
    """
    Create edge_weights which contain the back-propagated gradients

    :param graph:
    :param masks: shape (n_subgraphs, n_edge_in_original_graph,) edge masks
    :param grad: whether to contain gradient info
    :param add_full_graph:
    :return:
    """
    graphs = [graph] * masks.shape[0]
    if add_full_graph:
        graphs.append(graph)
        masks = torch.cat((masks,
                           torch.ones(1, masks.shape[1],
                                      dtype=masks.dtype,
                                      device=masks.device)), dim=0)
    edge_weights = masks.reshape(-1)
    return graphs, edge_weights, None


def construct_subgraph_batch(graph_list: List[Data],
                             num_subgraphs: List[int],
                             edge_weights: Optional[Sequence[Tensor]] = None,
                             selected_node_masks: Optional[Sequence[Tensor]] = None,
                             device: torch.device = 'cpu'):
    """

    :param graph_list: a list of [subgraph1_1, subgraph1_2, subgraph1_3, subgraph2_1, ...]
    :param num_subgraphs: a list number of subgraphs
    :param edge_weights:
    :param selected_node_masks:
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

    batch_idx = batch.batch
    if selected_node_masks is not None:
        selected_node_masks = torch.cat(selected_node_masks, dim=0) if isinstance(selected_node_masks, (list, tuple)) \
            else selected_node_masks

        if selected_node_masks.dtype == torch.float:
            pass
        elif selected_node_masks.dtype == torch.bool:
            batch_idx = batch.batch[selected_node_masks]
        else:
            raise ValueError

    if edge_weights is not None:
        edge_weights = torch.cat(edge_weights, dim=0) if isinstance(edge_weights, (list, tuple)) else edge_weights

    return SubgraphSetBatch(x=batch.x,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            edge_weight=edge_weights,
                            selected_node_masks=selected_node_masks,
                            y=batch.y[ptr[:-1]],
                            batch=batch_idx,
                            inter_graph_idx=original_graph_mask,
                            ptr=ptr,
                            num_graphs=batch.num_graphs)
