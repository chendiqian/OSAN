from typing import Tuple, List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph

from data import SubgraphSetBatch
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


def edgemasked_graphs_from_nodemask(graphs: List[Data] = None,
                                    edge_index: Tensor = None,
                                    masks: Tensor = None,
                                    grad: bool = True,
                                    remove_node: bool = False,
                                    add_full_graph: bool = False) \
        -> Tuple[List[Data], Tensor, Optional[Tensor]]:
    """

    :param graphs:
    :param edge_index:
    :param masks:
    :param grad:
    :param remove_node:
    :param add_full_graph:
    :return:
    """
    num_nodes, num_subgraphs = masks.shape
    num_edges = edge_index.shape[1]

    graphs = graphs * num_subgraphs if not add_full_graph else graphs * (num_subgraphs + 1)

    transform_func = Nodemask2Edgemask.apply if grad else nodemask2edgemask
    edge_weights = transform_func(masks.T, edge_index, torch.tensor(num_nodes, device=masks.device))

    edge_weights = edge_weights.reshape(-1)
    if add_full_graph:
        edge_weights = torch.cat((edge_weights,
                                  torch.ones(num_edges, dtype=edge_weights.dtype, device=edge_weights.device)), dim=0)
    if remove_node:
        selected_node_masks = masks.reshape(-1)
        if add_full_graph:
            selected_node_masks = torch.cat((selected_node_masks,
                                             torch.ones(num_nodes, dtype=masks.dtype, device=masks.device)),
                                            dim=0)
    else:
        selected_node_masks = None

    return graphs, edge_weights, selected_node_masks


def edgemasked_graphs_from_edgemask(graphs: List[Data] = None,
                                    masks: Tensor = None,
                                    add_full_graph: bool = False,
                                    **kwargs) \
        -> Tuple[List[Data], Tensor, Optional[Tensor]]:
    """
    Create edge_weights which contain the back-propagated gradients

    :param graphs:
    :param masks: shape (n_edge_in_original_graph, n_subgraphs) edge masks
    :param add_full_graph:
    :return:
    """
    num_edges, num_subgraphs = masks.shape

    graphs = graphs * num_subgraphs if not add_full_graph else graphs * (num_subgraphs + 1)
    masks = masks.T.reshape(-1)
    if add_full_graph:
        masks = torch.cat((masks, torch.ones(num_edges, device=masks.device, dtype=masks.dtype)), dim=0)

    return graphs, masks, None


def construct_subgraph_batch(graph_list: List[Data],
                             num_graphs: int,
                             num_subgraphs: int,
                             edge_weights: Tensor,
                             selected_node_masks: Optional[Tensor] = None,
                             device: torch.device = torch.device('cpu')):
    """

    :param graph_list: a list of [subgraph1_1, subgraph2_1, subgraph3_1, subgraph1_2, subgraph2_2, ...]
    :param num_graphs:
    :param num_subgraphs:
    :param edge_weights:
    :param selected_node_masks:
    :param device:
    :return:
    """
    # new batch
    batch = Batch.from_data_list(graph_list, None, None)
    original_graph_mask = torch.arange(num_graphs, device=device).repeat(num_subgraphs)

    # TODO: duplicate labels or aggregate the embeddings for original labels? potential problem: cannot split the
    #  batch because y shape inconsistent:
    #  https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Batch.to_data_list.
    #  need to check `batch._slice_dict` and `batch._inc_dict`

    batch_idx = batch.batch
    if selected_node_masks is not None:
        if selected_node_masks.dtype == torch.float:
            pass
        elif selected_node_masks.dtype == torch.bool:
            batch_idx = batch.batch[selected_node_masks]
        else:
            raise ValueError

    return SubgraphSetBatch(x=batch.x,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            edge_weight=edge_weights,
                            selected_node_masks=selected_node_masks,
                            y=batch.y[:num_graphs],
                            batch=batch_idx,
                            inter_graph_idx=original_graph_mask,
                            num_graphs=batch.num_graphs)
