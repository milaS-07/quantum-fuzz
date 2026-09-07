import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import heapq
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit_aer import *
from qiskit.quantum_info import Operator, Kraus, average_gate_fidelity
from qiskit.synthesis import qs_decomposition

from circuits import *
from helper_functions import *
from noise_parametrs import *

from rqc_fidelity import fidelity_from_circuit, Architecture

MAX_BLOCK_SIZE = 3


def build_gate_costs() -> dict:
    noise_model = get_sherbrooke_noise_model()
    costs = {}
    for gate_name, quantum_error in noise_model._default_quantum_errors.items():
        n = quantum_error.num_qubits
        kraus = Kraus(quantum_error)
        fidelity = average_gate_fidelity(kraus, Operator(np.eye(2 ** n)))
        costs[gate_name] = {"num_qubits": n, "infidelity": 1.0 - fidelity}
    return costs


GATE_COSTS = build_gate_costs()


def _average_infidelity(num_qubits: int) -> float:
    vals = [c["infidelity"] for c in GATE_COSTS.values() if c["num_qubits"] == num_qubits]
    return sum(vals) / len(vals) if vals else 0.0


ALPHA = _average_infidelity(1)
P_TWO_Q = _average_infidelity(2)
DEFAULT_ARCHITECTURE: Architecture = "full"

PROXY_CORRELATION = 0.48


class CircuitNode:
    def __init__(self, circuit, g_cost, parent=None, action=None):
        self.circuit = circuit
        self.g_cost = g_cost
        self.parent = parent
        self.action = action
        self.state_hash = qasm3.dumps(circuit)
        self.gate_count = len(circuit.data)

    @property
    def sort_key(self):
        return (self.g_cost, self.gate_count)

    def __lt__(self, other):
        return self.sort_key < other.sort_key


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


def calculate_required_shots(num_qubits: int, target_tvd_error: float = 0.05,
                              max_shots: int | None = 100_000) -> int:
    states = 2 ** num_qubits
    shots = int(states / (2 * (target_tvd_error ** 2)))
    shots = max(1000, shots)
    if max_shots is not None:
        shots = min(shots, max_shots)
    return shots


def calculate_noise_cost(circuit: QuantumCircuit, shots: int | None = None) -> float:
    if shots is None:
        shots = calculate_required_shots(circuit.num_qubits)

    meas_circuit = circuit.remove_final_measurements(inplace=False)
    meas_circuit.measure_all()

    sim_ideal = AerSimulator(max_parallel_threads=0)
    sim_noisy = AerSimulator(noise_model=get_sherbrooke_noise_model(), max_parallel_threads=0)

    counts_ideal = sim_ideal.run(meas_circuit, shots=shots).result().get_counts()
    counts_noisy = sim_noisy.run(meas_circuit, shots=shots).result().get_counts()

    fidelity = calculate_fidelity(counts_ideal, counts_noisy, shots)
    return 1.0 - fidelity


def calculate_noise_cost_fast(
    circuit: QuantumCircuit,
    alpha: float = ALPHA,
    p: float = P_TWO_Q,
    architecture: Architecture = DEFAULT_ARCHITECTURE,
) -> float:
    result = fidelity_from_circuit(circuit, alpha=alpha, p=p, architecture=architecture)
    return 1.0 - result["fidelity_solvable_model"]


def _select_shortlist(all_nodes: dict, min_verify: int = 40, max_verify: int = 200) -> list:
    nodes = list(all_nodes.values())
    if len(nodes) <= min_verify:
        return nodes

    costs = np.array([n.g_cost for n in nodes])
    best_cost = costs.min()
    std = costs.std()
    residual_std = std * np.sqrt(max(0.0, 1 - PROXY_CORRELATION ** 2))

    within_margin = [n for n in nodes if (n.g_cost - best_cost) <= residual_std]

    if len(within_margin) < min_verify:
        within_margin = sorted(nodes, key=lambda n: n.sort_key)[:min_verify]
    elif len(within_margin) > max_verify:
        within_margin = sorted(within_margin, key=lambda n: n.sort_key)[:max_verify]

    return within_margin


def uniform_cost_search(start_circuit: QuantumCircuit, max_iterations=1000,
                         max_tracked_nodes=5000, min_verify=40, max_verify=200):
    start_node = CircuitNode(
        circuit=start_circuit,
        g_cost=calculate_noise_cost_fast(start_circuit),
    )

    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set()
    best_node = start_node

    all_nodes: dict[str, CircuitNode] = {start_node.state_hash: start_node}

    def _track(node):
        all_nodes[node.state_hash] = node
        if len(all_nodes) > max_tracked_nodes:
            worst_hash = max(all_nodes, key=lambda h: all_nodes[h].sort_key)
            if worst_hash != node.state_hash:
                del all_nodes[worst_hash]

    iterations = 0
    while open_set and iterations < max_iterations:
        current_node = heapq.heappop(open_set)

        if current_node.state_hash in closed_set:
            continue
        closed_set.add(current_node.state_hash)

        if current_node.sort_key < best_node.sort_key:
            tie_on_formula = current_node.g_cost == best_node.g_cost
            print(f"[best updated] g_cost={current_node.g_cost:.6f} "
                  f"gate_count={current_node.gate_count} "
                  f"(tied on formula, won on gate count: {tie_on_formula})")
            best_node = current_node
        _track(current_node)

        for next_circ, action in get_matrix_decompositions(current_node.circuit):

            next_node = CircuitNode(
                circuit=next_circ,
                g_cost=calculate_noise_cost_fast(next_circ),
                parent=current_node,
                action=action
            )

            if next_node.state_hash not in closed_set:
                heapq.heappush(open_set, next_node)
                _track(next_node)

        iterations += 1

    shortlist = _select_shortlist(all_nodes, min_verify=min_verify, max_verify=max_verify)
    return best_node, shortlist


def verify_best_empirically(candidates, shots: int | None = None):
    if shots is None and candidates:
        shots = calculate_required_shots(candidates[0].circuit.num_qubits)

    print(f"\nVerifying {len(candidates)} shortlisted candidates empirically "
          f"({shots} shots each)...")
    scored = []
    for node in candidates:
        metrics = evaluate_circuit(node.circuit, shots=shots)
        scored.append((metrics["fidelity"], node, metrics))
        print(f"  gate_count={node.gate_count:3d}  "
              f"fast_g_cost={node.g_cost:.6f}  "
              f"empirical_fidelity={metrics['fidelity']:.6f}")

    scored.sort(key=lambda t: t[0], reverse=True)
    best_fidelity, best_node, best_metrics = scored[0]
    print(f"Empirically best candidate: gate_count={best_node.gate_count}, "
          f"fidelity={best_fidelity:.6f}\n")
    return best_node, best_metrics


def run_circuit_counts(circuit: QuantumCircuit, noise_model=None, shots=8192) -> dict:
    meas_circuit = circuit.remove_final_measurements(inplace=False)
    meas_circuit.measure_all()
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(meas_circuit, shots=shots).result()
    return result.get_counts()


def evaluate_circuit(circuit: QuantumCircuit, shots: int | None = None) -> dict:
    if shots is None:
        shots = calculate_required_shots(circuit.num_qubits)

    noise_model = get_sherbrooke_noise_model()
    counts_ideal = run_circuit_counts(circuit, noise_model=None, shots=shots)
    counts_noisy = run_circuit_counts(circuit, noise_model=noise_model, shots=shots)
    return {
        "tvd": calculate_tvd(counts_ideal, counts_noisy, shots),
        "fidelity": calculate_fidelity(counts_ideal, counts_noisy, shots),
        "js_divergence": calculate_js_divergence(counts_ideal, counts_noisy, shots),
    }


def run(shots: int | None = None, max_iterations=2000):
    start_circuit = get_rnd_circuit_4()
    _, shortlist = uniform_cost_search(start_circuit, max_iterations=max_iterations)

    original_metrics = evaluate_circuit(start_circuit, shots=shots)
    best_node, optimal_metrics = verify_best_empirically(shortlist, shots=shots)

    if optimal_metrics["fidelity"] < original_metrics["fidelity"]:
        print("No shortlisted candidate beat the original empirically - "
              "keeping the original circuit.")
        optimal_circuit = start_circuit
        optimal_metrics = original_metrics
    else:
        optimal_circuit = best_node.circuit

    print("Original circuit:")
    print(start_circuit.draw())
    print(original_metrics)

    print("Optimal circuit:")
    print(optimal_circuit.draw())
    print(optimal_metrics)

    path = []
    curr = best_node
    while curr.parent is not None:
        path.append(curr.action)
        curr = curr.parent

    for step in reversed(path):
        print(step)

    return original_metrics, optimal_metrics


if __name__ == "__main__":
    run()