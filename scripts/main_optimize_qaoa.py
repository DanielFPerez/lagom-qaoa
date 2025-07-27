import cirq
import numpy as np
import networkx as nx
import scipy.optimize
from typing import List
import time

import src.utils.graphs as graph_utils

# #### 
# Define QAOA circuit
def maxcut_cost_operator(graph: nx.Graph, qubits, gamma: float):
    return [cirq.ZZ(qubits[u], qubits[v]) ** gamma for u, v in graph.edges]
    
def mixing_operator(graph: nx.Graph, qubits, beta: float):
    return [cirq.X(qubits[node]) ** beta for node in graph.nodes]

def create_qaoa_circuit(graph: nx.Graph, p: int, gamma: List[float], beta: List[float]) -> cirq.Circuit:
    print(f"Creating QAOA circuit for p={p}, gamma={gamma}, beta={beta}")
    qubits = cirq.LineQubit.range(len(graph.nodes))
    circuit = cirq.Circuit()
    # Initialize in superposition
    circuit.append(cirq.H.on_each(qubits))
    for i in range(p):
        circuit.append(maxcut_cost_operator(graph, qubits, gamma[i]))
        circuit.append(mixing_operator(graph, qubits, beta[i]))
    # Add measurements
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

def eval_circuit(graph: nx.Graph, p: int, gamma, beta, reps):
    circuit = create_qaoa_circuit(graph, p, gamma, beta)
    simulator = cirq.Simulator()
    print(f"Running circuit for {reps} repetitions...")
    result = simulator.run(circuit, repetitions=reps)
    measurements = result.measurements['result']

    node_to_index = {node: i for i, node in enumerate(graph.nodes)}
    maxcut_value = 0
    for bitstring in measurements:
        cut_value = sum(1 for u, v in graph.edges if bitstring[node_to_index[u]] != bitstring[node_to_index[v]])
        maxcut_value += cut_value
    
    return maxcut_value / reps

# #### 


# ####
# Optimize QAOA parameters with gradient-based method
def objective_function(params, graph, p):
    gamma = params[:p]
    beta = params[p:]
    print(f"Creating circuit with parameters: gamma={gamma}, beta={beta}")
    circuit = create_qaoa_circuit(graph, p, gamma, beta)
    
    simulator = cirq.Simulator()

    print(f"Running circuit for 1000 repetitions...")
    result = simulator.run(circuit, repetitions=1000)
    measurements = result.measurements['result']
    
    maxcut_value = 0
    for bitstring in measurements:
        cut_value = sum(1 for edge in graph.edges if bitstring[edge[0]] != bitstring[edge[1]])
        maxcut_value += cut_value
    
    return -maxcut_value / 1000


# ### PARAMETER SHIFT RULE
def parameter_shift_rule(params, graph, p, i, shift=np.pi/8):
    params_shifted_forward = np.copy(params)
    params_shifted_forward[i] += shift
    obj_forward = objective_function(params_shifted_forward, graph, p)
    
    params_shifted_backward = np.copy(params)
    params_shifted_backward[i] -= shift
    obj_backward = objective_function(params_shifted_backward, graph, p)
    
    return (obj_forward - obj_backward) / (2 * np.sin(shift))

def gradient(params, graph, p, shift=np.pi/8):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        grad[i] = parameter_shift_rule(params, graph, p, i, shift)
    return grad


def optimize_qaoa(graph: nx.Graph, p: int, method: str):
    initial_params = np.random.uniform(0, np.pi, 2 * p)
    bounds = [(0, 2*np.pi)] * (2*p)
    if method == 'gradient':
        result = scipy.optimize.minimize(
            fun=objective_function,
            x0=initial_params,
            args=(graph, p),
            method='BFGS',
            jac=gradient
        )
    else:
        result = scipy.optimize.minimize(
            fun=objective_function,
            x0=initial_params,
            args=(graph, p),
            method=method, 
            bounds=bounds
        )
    return result.x



#mygraph = nx.star_graph(5)
#p = 3
#params = optimize_qaoa(mygraph, p=p, method="COBYLA")
#gamma, beta = params[:p], params[p:]
#reps = 1000
#circ_cut = eval_circuit(mygraph, p, gamma, beta, reps)
#print(f"Cut value from circuit: {circ_cut}")

mygraph = graph_utils.erdos_renyi_with_retry(24, 0.3)
print(f"Graph: {mygraph.graph['topology']}, n={mygraph.graph['n']}, p={mygraph.graph['p']}")
p = 3
t_start = time.time()
params = None
print(f"Optimizing QAOA parameters for p={p}...")
try: 
    params = optimize_qaoa(mygraph, p=p, method="COBYLA")
    print(f"Solution found after {time.time()-t_start} sec.")
except MemoryError:
    print(f"Exception cached after {time.time()-t_start} sec.")

if params is not None: 
    gamma, beta = params[:p], params[p:]
    reps = 1000
    circ_cut = eval_circuit(mygraph, p, gamma, beta, reps)
    print(f"Cut value from circuit: {circ_cut}")