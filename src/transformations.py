import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import heapq
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit_aer import *
from qiskit.quantum_info import Operator, Kraus, average_gate_fidelity
from qiskit.synthesis import qs_decomposition
from qiskit.quantum_info import Statevector, state_fidelity

from circuits import *
from helper_functions import *
from noise_parametrs import *

MAX_BLOCK_SIZE = 3


def build_gate_costs() -> dict:
    noise_model = get_sherbrooke_noise_model()
    costs = {}
    for gate_name, quantum_error in noise_model._default_quantum_errors.items():
        n = quantum_error.num_qubits
        kraus = Kraus(quantum_error)
        fidelity = average_gate_fidelity(kraus, Operator(np.eye(2 ** n)))
        costs[gate_name] = 1.0 - fidelity
    return costs


GATE_COSTS = build_gate_costs()


class CircuitNode:
    def __init__(self, circuit, g_cost, h_cost, parent=None, action=None):
        self.circuit = circuit
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent
        self.action = action
        self.state_hash = qasm3.dumps(circuit)

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost


def find_blocks(circuit: QuantumCircuit, qubit_indices: list):
    target = {circuit.qubits[i] for i in qubit_indices}
    blocks = []
    current = []
    for i, instr in enumerate(circuit.data):
        instr_qubits = set(instr.qubits)
        is_unitary = instr.operation.name not in ("measure", "barrier", "reset") and not instr.clbits
        if instr_qubits and instr_qubits.issubset(target) and is_unitary:
            current.append(i)
        elif instr_qubits & target:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return blocks


def block_unitary(circuit: QuantumCircuit, indices: list, qubit_indices: list) -> np.ndarray:
    n = len(qubit_indices)
    sub = QuantumCircuit(n)
    qmap = {circuit.qubits[q]: pos for pos, q in enumerate(qubit_indices)}
    for i in indices:
        instr = circuit.data[i]
        sub.append(instr.operation, [qmap[q] for q in instr.qubits])
    return Operator(sub).data


def replace_block(circuit: QuantumCircuit, indices: list, qubit_indices: list,
                   new_block: QuantumCircuit) -> QuantumCircuit:
    new_circuit = circuit.copy_empty_like()
    index_set = set(indices)
    insert_at = min(indices)
    target_qubits = [circuit.qubits[q] for q in qubit_indices]
    qmap = {new_block.qubits[pos]: target_qubits[pos] for pos in range(len(qubit_indices))}
    for i, instr in enumerate(circuit.data):
        if i == insert_at:
            for new_instr in new_block.data:
                mapped = [qmap[nq] for nq in new_instr.qubits]
                new_circuit.append(new_instr.operation, mapped)
        if i in index_set:
            continue
        new_circuit.append(instr.operation, instr.qubits, instr.clbits)
    return new_circuit


def get_matrix_decompositions(circuit: QuantumCircuit, max_block_size: int = MAX_BLOCK_SIZE):
    successors = []
    num_qubits = circuit.num_qubits
    top_size = min(num_qubits, max_block_size)

    for size in range(2, top_size + 1):
        for start in range(num_qubits - size + 1):
            qubit_indices = list(range(start, start + size))
            for indices in find_blocks(circuit, qubit_indices):
                if len(indices) < 2:
                    continue
                unitary = block_unitary(circuit, indices, qubit_indices)
                new_block = qs_decomposition(unitary)
                new_circuit = replace_block(circuit, indices, qubit_indices, new_block)
                action_desc = f"Resynthesized block on qubits {qubit_indices} ({len(indices)} ops)"
                successors.append((new_circuit, action_desc))

    return successors


def calculate_noise_cost(circuit: QuantumCircuit, shots: int = 1000) -> float:
    meas_circuit = circuit.remove_final_measurements(inplace=False)
    meas_circuit.measure_all()

    sim_ideal = AerSimulator(max_parallel_threads=0)
    sim_noisy = AerSimulator(noise_model=get_sherbrooke_noise_model(), max_parallel_threads=0)

    counts_ideal = sim_ideal.run(meas_circuit, shots=shots).result().get_counts()
    counts_noisy = sim_noisy.run(meas_circuit, shots=shots).result().get_counts()

    fidelity = calculate_fidelity(counts_ideal, counts_noisy, shots)
    return 1.0 - fidelity

def heuristic(circuit: QuantumCircuit) -> float:
    return 0.0



def a_star_search(start_circuit: QuantumCircuit, max_iterations=1000):
    start_node = CircuitNode(
        circuit=start_circuit,
        g_cost=calculate_noise_cost(start_circuit),
        h_cost=heuristic(start_circuit)
    )

    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set()
    best_node = start_node

    iterations = 0
    while open_set and iterations < max_iterations:
        current_node = heapq.heappop(open_set)

        if current_node.state_hash in closed_set:
            continue
        closed_set.add(current_node.state_hash)

        if current_node.g_cost < best_node.g_cost:
            best_node = current_node

        for next_circ, action in get_matrix_decompositions(current_node.circuit):

            next_node = CircuitNode(
                circuit=next_circ,
                g_cost=calculate_noise_cost(next_circ),
                h_cost=heuristic(next_circ),
                parent=current_node,
                action=action
            )

            if next_node.state_hash not in closed_set:
                heapq.heappush(open_set, next_node)

        iterations += 1

    return best_node


def run_circuit_counts(circuit: QuantumCircuit, noise_model=None, shots=8192) -> dict:
    meas_circuit = circuit.remove_final_measurements(inplace=False)
    meas_circuit.measure_all()
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(meas_circuit, shots=shots).result()
    return result.get_counts()


def evaluate_circuit(circuit: QuantumCircuit, shots=8192) -> dict:
    noise_model = get_sherbrooke_noise_model()
    counts_ideal = run_circuit_counts(circuit, noise_model=None, shots=shots)
    counts_noisy = run_circuit_counts(circuit, noise_model=noise_model, shots=shots)
    return {
        "tvd": calculate_tvd(counts_ideal, counts_noisy, shots),
        "fidelity": calculate_fidelity(counts_ideal, counts_noisy, shots),
        "js_divergence": calculate_js_divergence(counts_ideal, counts_noisy, shots),
    }


def run(shots=1000, max_iterations=50):
    start_circuit = get_rnd_circuit_4()
    optimal_node = a_star_search(start_circuit, max_iterations=max_iterations)

    original_metrics = evaluate_circuit(start_circuit, shots=shots)
    optimal_metrics = evaluate_circuit(optimal_node.circuit, shots=shots)

    print("Original circuit:")
    print(start_circuit.draw())
    print(original_metrics)

    print("Optimal circuit:")
    print(optimal_node.circuit.draw())
    print(optimal_metrics)

    path = []
    curr = optimal_node
    while curr.parent is not None:
        path.append(curr.action)
        curr = curr.parent

    for step in reversed(path):
        print(step)

    return original_metrics, optimal_metrics


if __name__ == "__main__":
    run()