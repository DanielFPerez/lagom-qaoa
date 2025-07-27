import networkx as nx
from typing import List
import nlopt
import logging
import numpy as np

import src.utils.graphs as graph_utils
from src.qaoa.qaoa_model import eval_circuit


# Optimize QAOA parameters with gradient-based method
def objective_function(params, graph, p, reps: int = 500):
    gamma = params[:p]
    beta = params[p:]
    cut_val = eval_circuit(graph, p, gamma, beta, reps=reps)
    return -cut_val


def parameter_shift_rule(params, graph, p, i, shift=np.pi/8):
    forward = np.copy(params)
    backward = np.copy(params)
    forward[i] += shift
    backward[i] -= shift
    f_plus = objective_function(forward, graph, p)
    f_minus = objective_function(backward, graph, p)
    return (f_plus - f_minus) / (2 * shift)

def qaoa_gradient(params, graph, p, shift=np.pi/8):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        grad[i] = parameter_shift_rule(params, graph, p, i, shift=0.5)
    return grad


# ----------------- NLopt Optimization ----------------- #
def optimize_qaoa_nlopt(graph: nx.Graph, p: int, method='COBYLA', in_xtol_rel: float = 1e-4, in_ftol_abs: float = 1e-3,
                        gamma_init = None, beta_init=None):
    logger = logging.getLogger(__name__)
    dim = 2 * p
    method = method.upper()
    maxeval = p*100
    
    # Select algorithm
    algo_map = {
        'COBYLA': nlopt.LN_COBYLA,
        'NELDER-MEAD': nlopt.LN_NELDERMEAD,
        'BFGS': nlopt.LD_LBFGS
    }
    if method not in algo_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from COBYLA, NELDER-MEAD, BFGS.")
    algorithm = algo_map[method]

    opt = nlopt.opt(algorithm, dim)
    
    # Set bounds: gamma ∈ [0, 2π], beta ∈ [0, π]
    lower_bounds = [0] * p + [0] * p  # gamma lower, beta lower
    upper_bounds = [2 * np.pi] * p + [np.pi] * p  # gamma upper, beta upper
    
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    
    logging.debug(f"Setting NLopt Maxeval: {maxeval}, Xtol_rel: {in_xtol_rel}, Ftol_abs: {in_ftol_abs}")
    opt.set_xtol_rel(in_xtol_rel)   
    opt.set_ftol_abs(in_ftol_abs)
    opt.set_maxeval(maxeval)

    if method == 'BFGS':
        def obj_with_grad(params, grad_out):
            grad = qaoa_gradient(params, graph, p)
            for i in range(len(params)):
                grad_out[i] = grad[i]
            return objective_function(params, graph, p)
        opt.set_min_objective(obj_with_grad)
    else:
        def obj_nograd(params, grad):  # grad is ignored
            return objective_function(params, graph, p)
        opt.set_min_objective(obj_nograd)

    # First p values: gamma ∈ [0, 2π], next p values: beta ∈ [0, π]
    if gamma_init is None:
        initial_gamma = np.random.uniform(0, 2 * np.pi, p)
    else:
        initial_gamma = gamma_init
    if beta_init is None:
        initial_beta = np.random.uniform(0, np.pi, p)
    else:
        initial_beta = beta_init
    initial_params = np.concatenate([initial_gamma, initial_beta])
    
    result = opt.optimize(initial_params)
    f_opt = opt.last_optimum_value()
    rc = opt.last_optimize_result()   # status code
    return result, f_opt, rc