# Some functions are adapted from ESAN paper:
# https://github.com/beabevi/ESAN/blob/master/data.py
# https://arxiv.org/abs/2110.02910

import math
import random
from typing import List, Callable, Optional, Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected

from subgraph.construct import nodesubset_to_subgraph
from subgraph.sampling_baseline import node_rand_sampling, edge_rand_sampling, edge_sample_preproc, khop_subgraph_sampling, \
    greedy_expand_sampling, max_spanning_tree_subgraph


class SamplerOnTheFly:
    def __init__(self, n_subgraphs: int = 1,
                 sample_k: Optional[int] = None,
                 remove_node: bool = False,
                 add_full_graph: bool = False):
        """
        Sample from a single graph, to create a batch of subgraphs.
        Especially suitable for situations where the deck is too large and inefficient.

        :param n_subgraphs:
        :param sample_k: nodes / edges per subgraphs
        :param remove_node:
        :param add_full_graph:
        """
        super(SamplerOnTheFly, self).__init__()
        self.n_subgraphs = n_subgraphs
        self.sample_k = sample_k
        self.relabel = remove_node
        self.add_full_graph = add_full_graph

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RawNodeSampler(SamplerOnTheFly):
    def __call__(self, data: Union[Data, Batch]) -> List[Data]:
        subgraphs = node_rand_sampling(data, self.n_subgraphs, self.sample_k, self.relabel, self.add_full_graph)
        return subgraphs


class RawEdgeSampler(SamplerOnTheFly):
    def __call__(self, data: Union[Data, Batch]) -> List[Data]:
        subgraphs = edge_rand_sampling(data, self.n_subgraphs, self.sample_k, self.add_full_graph)
        return subgraphs


class RawKhopSampler(SamplerOnTheFly):
    def __call__(self, data: Union[Data, Batch]) -> List[Data]:
        subgraphs = khop_subgraph_sampling(data, self.n_subgraphs, self.sample_k, self.relabel, self.add_full_graph)
        return subgraphs


class RawGreedyExpand(SamplerOnTheFly):
    def __call__(self, data: Union[Data, Batch]) -> List[Data]:
        subgraphs = greedy_expand_sampling(data, self.n_subgraphs, self.sample_k, self.relabel, self.add_full_graph)
        return subgraphs


class RawMSTSampler(SamplerOnTheFly):
    def __call__(self, data: Union[Data, Batch]) -> List[Data]:
        subgraphs = max_spanning_tree_subgraph(data, self.n_subgraphs, self.add_full_graph)
        return subgraphs


class DeckSampler:
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
    def __init__(self, process_subgraphs: Callable = None, remove_node: bool = False, add_full_graph: bool = True):
        self.process_subgraphs = process_subgraphs
        self.relabel = remove_node
        self.add_full_graph = add_full_graph

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

        Here we do not remove x, because in I-MLE the node is also not removed. If the node attribute is removed,
        GIN results into different results.

        :param data:
        :return:
        """
        subgraphs = [data] if self.add_full_graph else []
        all_nodes = torch.arange(data.num_nodes)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1:]])
            subgraphs.append(nodesubset_to_subgraph(data, subset, relabel=self.relabel))
        return subgraphs


class EdgeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data: Data) -> List[Data]:
        # no edge
        if data.edge_index.shape[1] == 0:
            return [data]

        edge_index, edge_attr, undirected = edge_sample_preproc(data)

        subgraphs = [data] if self.add_full_graph else []
        for i in range(edge_index.shape[1]):
            subgraph_edge_index = torch.hstack([edge_index[:, :i], edge_index[:, i + 1:]])
            subgraph_edge_attr = torch.vstack([edge_attr[:i], edge_attr[i + 1:]]) if edge_attr is not None else None

            if undirected:
                if subgraph_edge_attr is not None:
                    subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                            num_nodes=data.num_nodes)
                else:
                    subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                        num_nodes=data.num_nodes)

            subgraphs.append(Data(
                    x=data.x,
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    num_nodes=data.num_nodes,
                    y=data.y,
                ))

        return subgraphs


def policy2transform(policy: str,
                     process_subgraphs: Callable = None,
                     relabel: bool = False,
                     add_full_graph: bool = True) -> Union[Graph2Subgraph, Callable]:
    """
    Pre-transform for datasets
    e.g. make a deck of subgraphs for the original graph, each with size n - 1

    :param policy:
    :param process_subgraphs:
    :param relabel:
    :param add_full_graph:
    :return:
    """
    if policy == "node_deleted":
        return NodeDeleted(process_subgraphs=process_subgraphs, remove_node=relabel, add_full_graph=add_full_graph)
    elif policy == 'edge_deleted':
        return EdgeDeleted(process_subgraphs=process_subgraphs, add_full_graph=add_full_graph)
    elif policy == 'null':
        return lambda x: x
    else:
        raise ValueError("Invalid subgraph policy type")
