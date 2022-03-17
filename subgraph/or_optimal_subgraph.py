from typing import List

from ortools.linear_solver import pywraplp
import torch


def solve_or(value_list: List[List[int]], node_per_subgraphs) -> pywraplp.Solver:
    n_subgraph, n_nodes = len(value_list), len(value_list[0])
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # variable
    x = [[None] * n_nodes for _ in range(n_subgraph)]
    for i in range(n_subgraph):
        for j in range(n_nodes):
            x[i][j] = solver.IntVar(0, 1, f'x[{i}][{j}]')

    # y = [[None] * graph.num_edges for _ in range(n_subgraph)]
    # for i in range(n_subgraph):
    #     for j in range(graph.num_edges):
    #         y[i][j] = solver.IntVar(0, 1, f'y[{i}][{j}]')

    # obj
    objective = solver.Objective()
    for i in range(n_subgraph):
        for j in range(n_nodes):
            objective.SetCoefficient(x[i][j], value_list[i][j])
    objective.SetMaximization()

    # coveredness
    if node_per_subgraphs * n_subgraph >= n_nodes:
        for j in range(n_nodes):
            solver.Add(sum([x[i][j] for i in range(n_subgraph)]) >= 1)

    # size of subgraphs
    for i in range(n_subgraph):
        solver.Add(sum([x[i][j] for j in range(n_nodes)]) == node_per_subgraphs)

    # # for each edge selected, its nodes are selected
    # for i in range(n_subgraph):
    #     for j in range(graph.num_edges):
    #         solver.Add(x[i][edge_index[j][0]] >= y[i][j])
    #         solver.Add(x[i][edge_index[j][1]] >= y[i][j])
    #
    # # two adjacent nodes selected, then their edge is selected
    # for i, (n1, n2) in enumerate(edge_index):
    #     for j in range(n_subgraph):
    #         solver.Add(y[j][i] <= x[j][n1])
    #         solver.Add(y[j][i] <= x[j][n2])
    #         solver.Add(y[j][i] >= x[j][n1] + x[j][n2] - 1)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        return solver
    else:
        raise ValueError("No solution")


def get_or_optim_subgraphs(value_tensor: torch.Tensor, node_per_subgraphs: int) -> torch.Tensor:
    """

    :param value_tensor:  shape (n_nodes, n_subgraphs)
    :param node_per_subgraphs:
    :return:
    """
    n_nodes, n_subgraph = value_tensor.shape

    if node_per_subgraphs >= n_nodes:
        return torch.ones_like(value_tensor, dtype=torch.float32, device=value_tensor.device)

    value_list = value_tensor.t().cpu().tolist()

    raw_solution = solve_or(value_list, node_per_subgraphs)
    x = raw_solution.variables()

    solution = torch.zeros_like(value_tensor, dtype=torch.float32, device=value_tensor.device)
    for i in range(n_subgraph):
        for j in range(n_nodes):
            solution[j, i] = x[i * n_subgraph + j].solution_value()

    return solution
