from enum import Enum
from typing import Dict, List
import time

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model


class MaxCutCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, cut_var, node_vars):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._cut_var = cut_var
        self._node_vars = node_vars
        self.best_obj = 0
        self.best_partition = {}
        self.status = None
        self.runtime = 0

    def on_solution_callback(self):
        obj = self.ObjectiveValue()
        if obj > self.best_obj:
            self.best_obj = obj
            
            # Save partition assignment
            partition = {node: self.Value(var) for node, var in self._node_vars.items()}
            self.best_partition = partition
            
            part0 = [n for n, p in partition.items() if p == 0]
            part1 = [n for n, p in partition.items() if p == 1]
            print(f"New best cut: {int(obj)}.  Partition sizes: |A|={len(part0)}, |B|={len(part1)}")


def build_maxcut_model(graph: nx.Graph):
    # Create the CP-SAT model
    model = cp_model.CpModel()
    
    node_vars = {}
    for node in graph.nodes():
        node_vars[node] = model.new_bool_var(f'node_{node}')

    # Add constraints to maximize the number of edges between the two partitions
    prod_uv = {}
    cut_var = {}    
    for u, v in graph.edges():
        cut_var[(u,v)] = model.new_bool_var(f'cut_{u}_{v}')
        prod_uv[(u, v)] = model.new_int_var(0, 2, f"prod_{u}_{v}")
        model.add_multiplication_equality(prod_uv[(u, v)], [node_vars[u], node_vars[v]])
        model.add(cut_var[(u, v)] == (node_vars[u] + node_vars[v] - 2 * prod_uv[(u, v)]))
    
    return model, node_vars, cut_var  


def ortools_solve_maxcut(mygraph: nx.Graph, nworkers: int = 16, timeout: int = 60) -> Dict[int, int]:
    """
    Solve the Max-Cut problem using OR-Tools CP-SAT solver.
    
    Args:
        mygraph (nx.Graph): The input graph.
        nworkers (int): Number of workers for parallel solving.
        timeout (int): Timeout in seconds for the solver.
    
    Returns:
        Dict[int, int]: Partition assignment of nodes.
    """
    model, node_vars, cut_var = build_maxcut_model(mygraph)
    
    # Objective: maximize the sum of cut variables
    model.maximize(sum(cut_var.values()))
    
    # Create a solver instance
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = nworkers
    solver.parameters.max_time_in_seconds = timeout
    
    # Create a callback to capture the best solution
    callback = MaxCutCallback(cut_var, node_vars)
    
    # Solve the model
    t_start = time.time()
    status = solver.Solve(model, callback)
    t_duration = time.time() - t_start
    print(f"Time to solve the problem: {round(t_duration, 4)} sec")  
    
    callback.status = status
    callback.runtime = t_duration    

    if status == cp_model.OPTIMAL:
        print(f"Optimal value: {solver.ObjectiveValue()}")
        return callback
    elif status == cp_model.FEASIBLE:
        print(f"Feasible solution found with value: {solver.ObjectiveValue()}")
        return callback
    else:
        print("No solution found.")
        return {}