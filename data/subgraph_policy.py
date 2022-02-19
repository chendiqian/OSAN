# Some functions are adapted from ESAN paper:
# https://github.com/beabevi/ESAN/blob/master/data.py
# https://arxiv.org/abs/2110.02910

import math
import random
from typing import List, Callable, Optional, Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from data import SubgraphSetBatch


class Sampler:
    def __init__(self, mode: str, fraction: float, num_subgraph: int):
        """
        Sample subgraphs from a full deck (given by the dataset[idx])

        :param mode: Either fraction or k subgraphs
        :param fraction:
        :param num_subgraph:
        """
        assert mode in ['float', 'int']
        self.mode = mode
        self.fraction = fraction
        self.num_subgraph = num_subgraph

    def __call__(self, batch: Union[Data, Batch]) -> List[Data]:
        data_list = Batch.to_data_list(batch)
        count = math.ceil(self.fraction * len(data_list)) if self.mode == 'float' else self.num_subgraph
        sampled_subgraphs = random.sample(data_list, count) if count < len(data_list) else data_list
        return sampled_subgraphs


class Graph2Subgraph:
    def __init__(self, process_subgraphs: Callable = None):
        self.process_subgraphs = process_subgraphs

    def __call__(self, data: Data) -> List[Data]:
        subgraphs = self.to_subgraphs(data)
        if self.process_subgraphs is not None:
            subgraphs = [self.process_subgraphs(s) for s in subgraphs]

        return Batch.from_data_list(subgraphs)

    def to_subgraphs(self, data):
        raise NotImplementedError


class NodeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data: Data) -> List[Data]:
        """
        In the original code they don't relabel the new edge_index and don't remove the node attribute from data.x
        https://github.com/beabevi/ESAN/blob/master/data.py#L270
        But subgraph will remove the self-loops, so relabeling is equivalent.

        :param data:
        :return:
        """
        subgraphs = []
        all_nodes = torch.arange(data.num_nodes)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1:]])
            subgraph_edge_index, subgraph_edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                                               relabel_nodes=True, num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x[subset, :],
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    num_nodes=data.num_nodes - 1,
                    y=data.y,
                )
            )
        return subgraphs


def policy2transform(policy: str, process_subgraphs: Callable = None) -> Optional[Graph2Subgraph]:
    """
    Pre-transform for datasets
    e.g. make a deck of subgraphs for the original graph, each with size n - 1

    :param policy:
    :param process_subgraphs:
    :return:
    """
    if policy == "node_deleted":
        return NodeDeleted(process_subgraphs=process_subgraphs)
    elif policy == 'null':
        return None
    else:
        raise ValueError("Invalid subgraph policy type")
