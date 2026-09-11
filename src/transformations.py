import sys
import os
import json
import heapq
import warnings

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_aer import *
from qiskit.quantum_info import Operator, Kraus, average_gate_fidelity
from qiskit.synthesis import qs_decomposition
from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.converters import circuit_to_dag

import mqt.bench as mqt
from mqt.bench import BenchmarkLevel

from src.noise.noise_metrics import *
from src.noise.noise_parametrs import *
from src.predictors.rqc_fidelity import fidelity_from_circuit, Architecture

warnings.filterwarnings("ignore", category=UserWarning)

from qiskit.passmanager import PropertySet

MAX_BLOCK_SIZE = 3
MAX_BLOCKS_PER_NODE = 12


class SuppressRustPanic:
    def __enter__(self):
        sys.stderr.flush()
        try:
            self.fd = sys.stderr.fileno()
            self.old_stderr = os.dup(self.fd)
            self.devnull = open(os.devnull, 'w')
            os.dup2(self.devnull.fileno(), self.fd)
        except Exception:
            self.fd = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            sys.stderr.flush()
            os.dup2(self.old_stderr, self.fd)
            self.devnull.close()
            os.close(self.old_stderr)


_NOISE_MODEL = None


def cached_noise_model():
    global _NOISE_MODEL
    if _NOISE_MODEL is None:
        _NOISE_MODEL = get_sherbrooke_noise_model()
    return _NOISE_MODEL


def build_gate_costs() -> dict:
    noise_model = cached_noise_model()
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


def cheap_state_hash(circuit: QuantumCircuit):
    parts = []
    for instr in circuit.data:
        qidx = tuple(circuit.find_bit(q).index for q in instr.qubits)
        params = tuple(round(float(p), 6) for p in instr.operation.params) if instr.operation.params else ()
        parts.append((instr.operation.name, qidx, params))
    return tuple(parts)


class CircuitNode:
    def __init__(self, circuit, g_cost, parent=None, action=None):
        self.circuit = circuit
        self.g_cost = g_cost
        self.parent = parent
        self.action = action
        self.state_hash = cheap_state_hash(circuit)
        self.gate_count = len(circuit.data)

    @property
    def sort_key(self):
        return (self.g_cost, self.gate_count)

    def __lt__(self, other):
        return self.sort_key < other.sort_key


def _block_to_unitary(block_nodes):
    qubits_in_block = []
    for node in block_nodes:
        for q in node.qargs:
            if q not in qubits_in_block:
                qubits_in_block.append(q)
    qmap = {q: i for i, q in enumerate(qubits_in_block)}
    sub = QuantumCircuit(len(qubits_in_block))
    for node in block_nodes:
        sub.append(node.op, [qmap[q] for q in node.qargs])
    return Operator(sub).data, qubits_in_block


def _replace_block(circuit, dag, block_nodes, qubits_in_block, new_block):
    block_ids = {id(n) for n in block_nodes}
    qmap = {new_block.qubits[i]: q for i, q in enumerate(qubits_in_block)}
    new_circuit = circuit.copy_empty_like()
    emitted = False
    for node in dag.topological_op_nodes():
        if id(node) in block_ids:
            if not emitted:
                for instr in new_block.data:
                    new_circuit.append(instr.operation, [qmap[q] for q in instr.qubits])
                emitted = True
            continue
        new_circuit.append(node.op, node.qargs, node.cargs)
    return new_circuit


def _round_unitary_key(unitary: np.ndarray) -> bytes:
    return np.round(unitary, 6).tobytes()


_DECOMP_CACHE: dict = {}


def _cached_qs_decomposition(unitary: np.ndarray):
    key = _round_unitary_key(unitary)
    if key not in _DECOMP_CACHE:
        with SuppressRustPanic():
            _DECOMP_CACHE[key] = qs_decomposition(unitary)
    return _DECOMP_CACHE[key]


def get_matrix_decompositions(circuit: QuantumCircuit, max_block_size: int = MAX_BLOCK_SIZE):
    dag = circuit_to_dag(circuit)

    collect_pass = CollectMultiQBlocks(max_block_size=max_block_size)
    collect_pass.property_set = PropertySet()
    collect_pass.run(dag)
    blocks = collect_pass.property_set.get("block_list", [])

    blocks = [b for b in blocks if len(b) >= 2]
    blocks.sort(key=lambda b: sum(1 for n in b if len(n.qargs) == 2), reverse=True)
    blocks = blocks[:MAX_BLOCKS_PER_NODE]

    successors = []
    for block_nodes in blocks:
        try:
            unitary, qubits_in_block = _block_to_unitary(block_nodes)
            orig_2q = sum(1 for n in block_nodes if len(n.qargs) == 2)
            new_block = _cached_qs_decomposition(unitary)
            new_2q = sum(1 for instr in new_block.data if len(instr.qubits) == 2)
            if new_2q > orig_2q:
                continue
            new_circuit = _replace_block(circuit, dag, block_nodes, qubits_in_block, new_block)
            successors.append((new_circuit, ""))
        except KeyboardInterrupt:
            raise
        except BaseException:
            continue
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

    sim_ideal = AerSimulator(method="statevector", max_parallel_threads=0)
    sim_noisy = AerSimulator(
        noise_model=cached_noise_model(),
        method="statevector",
        max_parallel_threads=0,
    )

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


def _select_shortlist(all_nodes: dict, min_verify: int = 5, max_verify: int = 200) -> list:
    nodes = list(all_nodes.values())
    if len(nodes) <= min_verify:
        return nodes

    nodes_sorted = sorted(nodes, key=lambda n: n.sort_key)
    costs = np.array([n.g_cost for n in nodes_sorted])
    gaps = np.diff(costs)

    if len(gaps) == 0:
        return nodes_sorted[:max_verify]

    mean_gap = gaps.mean()
    std_gap = gaps.std()
    threshold = mean_gap + std_gap

    cut = len(nodes_sorted)
    for i, g in enumerate(gaps):
        if g > threshold and (i + 1) >= min_verify:
            cut = i + 1
            break

    cut = max(min_verify, min(cut, max_verify))
    return nodes_sorted[:cut]


def uniform_cost_search(start_circuit: QuantumCircuit, max_iterations=1000,
                         max_tracked_nodes=5000, prune_batch=500,
                         min_verify=5, max_verify=200):
    start_node = CircuitNode(
        circuit=start_circuit,
        g_cost=calculate_noise_cost_fast(start_circuit),
    )

    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set()
    best_node = start_node

    all_nodes: dict = {start_node.state_hash: start_node}

    def _track(node):
        all_nodes[node.state_hash] = node
        if len(all_nodes) > max_tracked_nodes + prune_batch:
            sorted_hashes = sorted(all_nodes, key=lambda h: all_nodes[h].sort_key)
            for h in sorted_hashes[max_tracked_nodes:]:
                del all_nodes[h]

    iterations = 0
    while open_set and iterations < max_iterations:
        current_node = heapq.heappop(open_set)

        if current_node.state_hash in closed_set:
            continue
        closed_set.add(current_node.state_hash)

        if current_node.sort_key < best_node.sort_key:
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


def run_circuit_counts(circuit: QuantumCircuit, noise_model=None, shots=8192) -> dict:
    meas_circuit = circuit.remove_final_measurements(inplace=False)
    meas_circuit.measure_all()
    simulator = AerSimulator(
        noise_model=noise_model,
        method="statevector",
        max_parallel_threads=0,
        max_parallel_experiments=0,
    )
    result = simulator.run(meas_circuit, shots=shots).result()
    return result.get_counts()


def evaluate_circuit(circuit: QuantumCircuit, shots: int | None = None) -> dict:
    if shots is None:
        shots = calculate_required_shots(circuit.num_qubits)

    noise_model = cached_noise_model()
    counts_ideal = run_circuit_counts(circuit, noise_model=None, shots=shots)
    counts_noisy = run_circuit_counts(circuit, noise_model=noise_model, shots=shots)
    return {
        "tvd": calculate_tvd(counts_ideal, counts_noisy, shots),
        "fidelity": calculate_fidelity(counts_ideal, counts_noisy, shots),
        "js_divergence": calculate_js_divergence(counts_ideal, counts_noisy, shots),
    }


LOW_SHOTS_SCREEN = 1000


def verify_best_empirically(candidates, shots: int | None = None,
                             screen_shots: int = LOW_SHOTS_SCREEN):
    if not candidates:
        raise ValueError("candidates is empty")

    noise_model = cached_noise_model()

    print(f"  Screening {len(candidates)} shortlisted candidates "
          f"({screen_shots} shots each)...")
    screened = []
    for node in candidates:
        counts_ideal = run_circuit_counts(node.circuit, noise_model=None, shots=screen_shots)
        counts_noisy = run_circuit_counts(node.circuit, noise_model=noise_model, shots=screen_shots)
        tvd = calculate_tvd(counts_ideal, counts_noisy, screen_shots)
        screened.append((tvd, node))

    screened.sort(key=lambda t: t[0])
    best_tvd_screen, best_node = screened[0]

    if shots is None:
        shots = calculate_required_shots(best_node.circuit.num_qubits)

    print(f"  Re-verifying winner (gate_count={best_node.gate_count}) at full "
          f"{shots} shots for the reported number...")
    best_metrics = evaluate_circuit(best_node.circuit, shots=shots)
    print(f"  Empirically best candidate: gate_count={best_node.gate_count}, "
          f"fidelity={best_metrics['fidelity']:.6f}, tvd={best_metrics['tvd']:.6f}")

    return best_node, best_metrics


BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register(benchmarks: List[str], category: str, is_scalable: bool):
    for name in benchmarks:
        BENCHMARK_REGISTRY[name] = {
            "category": category,
            "scalable": is_scalable
        }


_register([
    "Amplitude Estimation (AE)", "Deutsch-Jozsa", "Graph State", "GHZ State",
    "Grover's", "Quantum Approximation Optimization Algorithm (QAOA)",
    "Quantum Fourier Transformation (QFT)", "Entangled QFT", "Quantum Neural Network (QNN)",
    "Quantum Phase Estimation (QPE) exact", "Quantum Phase Estimation (QPE) inexact",
    "Quantum Walk", "Random Circuit",
    "Efficient SU2 ansatz with Random Parameters",
    "Real Amplitudes ansatz with Random Parameters", "Two Local ansatz with random parameters",
    "W-State"
], category="auto", is_scalable=True)

_register([
    "Shor's"
], category="auto", is_scalable=False)

PQC_BENCHMARKS = [
    "Efficient SU2 ansatz with Random Parameters",
    "Real Amplitudes ansatz with Random Parameters",
    "Two Local ansatz with random parameters",
    "Quantum Approximation Optimization Algorithm (QAOA)",
    "Quantum Neural Network (QNN)",
]

CLIFFORD_BENCHMARKS = [
    "GHZ State",
    "Graph State",
]

NON_PARAM_NON_CLIFFORD_BENCHMARKS = [
    "W-State",
    "Deutsch-Jozsa",
    "Grover's",
    "Quantum Walk",
    "Shor's",
    "Quantum Fourier Transformation (QFT)",
    "Entangled QFT",
    "Quantum Phase Estimation (QPE) exact",
    "Quantum Phase Estimation (QPE) inexact",
    "Amplitude Estimation (AE)",
]

CONTROL_BENCHMARKS = [
    "Random Circuit"
]

for name in PQC_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "PQC"
for name in CLIFFORD_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Clifford"
for name in NON_PARAM_NON_CLIFFORD_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Non-Parametric Non-Clifford"
for name in CONTROL_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Control"

FIXED_CIRCUIT_SIZES = {
    "shor": 18,
}


def get_mqt_circuit(benchmark_name: str, num_qubits: int = None) -> QuantumCircuit:
    meta = BENCHMARK_REGISTRY.get(benchmark_name, {})
    is_scalable = meta.get("scalable", True)

    name_map = {
        "Grover's": "grover",
        "Quantum Walk": "qwalk",
        "Efficient SU2 ansatz with Random Parameters": "vqe_su2",
        "Real Amplitudes ansatz with Random Parameters": "vqe_real_amp",
        "Two Local ansatz with random parameters": "vqe_two_local",
        "Entangled QFT": "qftentangled",
        "Quantum Fourier Transformation (QFT)": "qft",
        "Quantum Phase Estimation (QPE) exact": "qpeexact",
        "Quantum Phase Estimation (QPE) inexact": "qpeinexact",
        "Amplitude Estimation (AE)": "ae",
        "Quantum Approximation Optimization Algorithm (QAOA)": "qaoa",
        "Quantum Neural Network (QNN)": "qnn",
        "GHZ State": "ghz",
        "Graph State": "graphstate",
        "W-State": "wstate",
        "Deutsch-Jozsa": "dj",
        "Random Circuit": "randomcircuit",
        "Shor's": "shor",
    }

    mqt_id = name_map.get(benchmark_name, benchmark_name.lower().replace(" ", ""))

    try:
        if is_scalable:
            qc = mqt.get_benchmark(mqt_id, BenchmarkLevel.ALG, circuit_size=num_qubits)
        else:
            fixed_size = FIXED_CIRCUIT_SIZES.get(mqt_id, num_qubits)
            qc = mqt.get_benchmark(mqt_id, BenchmarkLevel.ALG, circuit_size=fixed_size)

        if not any(gate.operation.name == "measure" for gate in qc.data):
            qc.measure_all()
        return qc

    except Exception as e:
        raise ValueError(
            f"Greška pri učitavanju kola '{benchmark_name}' (mqt_id='{mqt_id}', n={num_qubits}): {str(e)}"
        ) from e


def plot_benchmark_metrics(bench_data: dict, bench_dir: str):
    qubits = []
    fids_orig, fids_opt = [], []
    tvds_orig, tvds_opt = [], []

    sorted_qubits = sorted([int(k) for k in bench_data["results_per_qubit"].keys()])
    for q in sorted_qubits:
        res = bench_data["results_per_qubit"][str(q)]
        qubits.append(q)
        fids_orig.append(res["original"]["fidelity"])
        tvds_orig.append(res["original"]["tvd"])
        fids_opt.append(res["optimized"]["fidelity"])
        tvds_opt.append(res["optimized"]["tvd"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(qubits, fids_orig, "o--", color="tab:blue", alpha=0.5, label="Orig Fidelity")
    ax1.plot(qubits, fids_opt, "o-", color="tab:blue", label="Opt Fidelity")
    ax2.plot(qubits, tvds_orig, "s--", color="tab:red", alpha=0.5, label="Orig TVD")
    ax2.plot(qubits, tvds_opt, "s-", color="tab:red", label="Opt TVD")

    ax1.set_xlabel("Broj kubita (N)")
    ax1.set_ylabel("Fidelity", color="tab:blue")
    ax2.set_ylabel("TVD", color="tab:red")
    plt.title(f"{bench_data['metadata']['name']} - Optimizacija")
    ax1.grid(True, linestyle=":", alpha=0.6)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    fig.tight_layout()
    plt.savefig(os.path.join(bench_dir, "metrics_vs_qubits_optimized.png"), dpi=200)
    plt.close()


def run_all_benchmarks(
    qubit_range: list,
    shots: int | None,
    max_iterations: int,
    base_results_dir: str
):
    plots_dir = os.path.join(base_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    basis_gates = ["id", "rz", "sx", "x", "cx"]

    print("=" * 70)
    print(f"POKRETANJE EKSPERIMENATA (Qubit opseg: {qubit_range})")
    print("=" * 70)

    for bench_name, meta in BENCHMARK_REGISTRY.items():
        category = meta["category"]
        is_scalable = meta["scalable"]

        cat_dir = os.path.join(base_results_dir, category.replace(" ", "_"))
        bench_dir = os.path.join(cat_dir, bench_name.replace(" ", "_").replace("'", ""))
        circuits_dir = os.path.join(bench_dir, "optimized_circuits")
        os.makedirs(circuits_dir, exist_ok=True)

        metrics_path = os.path.join(bench_dir, "metrics.json")
        bench_data = {
            "metadata": {
                "name": bench_name,
                "category": category,
                "scalable": is_scalable,
                "shots": shots
            },
            "results_per_qubit": {}
        }

        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)
            except json.JSONDecodeError:
                pass

        target_qubits = qubit_range if is_scalable else [qubit_range[0]]
        print(f"\n---> Kolo: {bench_name} [{category}]")

        for n_q in target_qubits:
            q_key = str(n_q)

            if q_key in bench_data["results_per_qubit"] and "original" in bench_data["results_per_qubit"][q_key]:
                print(f"  Qubits: {n_q:2d} | VEĆ ZAVRŠENO, preskačem.")
                continue

            try:
                qc = get_mqt_circuit(bench_name, num_qubits=n_q)
                qc_t = transpile(qc, basis_gates=basis_gates, optimization_level=1)

                print(f"  Qubits: {n_q:2d} | Evaluacija originala...")
                original_metrics = evaluate_circuit(qc_t, shots=shots)

                print(f"  Qubits: {n_q:2d} | UCS optimizacija...")
                best_node_ucs, shortlist = uniform_cost_search(qc_t, max_iterations=max_iterations)

                best_node, optimal_metrics = verify_best_empirically(shortlist, shots=shots)

                improved = False
                if optimal_metrics["fidelity"] < original_metrics["fidelity"]:
                    optimal_metrics = original_metrics
                    saved_node = None
                else:
                    improved = True
                    saved_node = best_node

                qasm_path = None
                if saved_node is not None:
                    qasm_filename = f"n{n_q}_optimized.qasm"
                    qasm_path = os.path.join(circuits_dir, qasm_filename)
                    with open(qasm_path, "w", encoding="utf-8") as qf:
                        qf.write(qasm3.dumps(saved_node.circuit))

                bench_data["results_per_qubit"][q_key] = {
                    "num_qubits": n_q,
                    "improved": improved,
                    "original": original_metrics,
                    "optimized": optimal_metrics,
                    "optimized_gate_count": saved_node.gate_count if saved_node else None,
                    "optimized_qasm_path": qasm_path,
                }

                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(bench_data, f, indent=4)

            except Exception as err:
                print(f"  [GREŠKA] Nije moguće izvršiti {bench_name} za n={n_q}: {err}")

        if is_scalable and len(bench_data["results_per_qubit"]) > 1:
            plot_benchmark_metrics(bench_data, bench_dir)


def main():
    target_dir = "results_quantum" 
    run_all_benchmarks(
        qubit_range=[2, 4, 6, 8],
        shots=None,
        max_iterations=2000,
        base_results_dir=target_dir
    )


if __name__ == "__main__":
    main()