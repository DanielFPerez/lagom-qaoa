import cirq
import networkx as nx
from typing import List
import logging
import numpy as np

# Define QAOA circuit
def maxcut_cost_operator(graph: nx.Graph, qubits, gamma: float):
    return [cirq.ZZ(qubits[u], qubits[v]) ** gamma for u, v in graph.edges]
    
def mixing_operator(graph: nx.Graph, qubits, beta: float):
    return [cirq.X(qubits[node]) ** beta for node in graph.nodes]

def create_qaoa_circuit(graph: nx.Graph, p: int, gamma: List[float], beta: List[float]) -> cirq.Circuit:
    logger = logging.getLogger(__name__)

    # logger.debug(f"Creating QAOA circuit for p={p}, gamma={gamma}, beta={beta}")
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

def eval_circuit(graph: nx.Graph, p: int, gamma, beta, reps: int = 500):
    logger = logging.getLogger(__name__)

    circuit = create_qaoa_circuit(graph, p, gamma, beta)
    simulator = cirq.Simulator()
    # logger.debug(f"Running circuit for {reps} repetitions...")
    result = simulator.run(circuit, repetitions=reps)
    measurements = result.measurements['result']

    maxcut_value = 0
    node_to_index = {node: i for i, node in enumerate(graph.nodes)}
    for bitstring in measurements:
        cut_value = sum(1 for u, v in graph.edges if bitstring[node_to_index[u]] != bitstring[node_to_index[v]])
        maxcut_value += cut_value
    
    return maxcut_value / reps


def estimate_random_average_energy(graph: nx.Graph, p: int, reps: int = 128, n_samples: int = 100) -> float:
    avg_energy = 0.0

    for _ in range(n_samples):
        beta = np.random.uniform(0, np.pi, p)
        gamma = np.random.uniform(0, 2*np.pi, p)
        
        avg_energy += eval_circuit(graph, p, gamma, beta, reps)

    return avg_energy / n_samples