import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import json
import time
from datetime import datetime

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, Kraus, average_gate_fidelity, state_fidelity
from qiskit.synthesis import qs_decomposition
from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import PropertySet

from circuits import get_mqt_circuit
from noise.noise_parametrs import get_sherbrooke_noise_model

# Import both predictors cleanly from their separate files
from predictors.rqc_fidelity import fidelity_from_circuit, guess_architecture
from predictors.epa_fidelity import get_avg_gate_fidelities, predict_fidelity_epa

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


QUBIT_RANGE = [3, 6, 9]

BENCHMARKS = [
    "Quantum Fourier Transformation (QFT)",
    "GHZ State",
    "Quantum Approximation Optimization Algorithm (QAOA)",
    "Real Amplitudes ansatz with Random Parameters",
    "W-State",
    "Graph State",
    "Deutsch-Jozsa",
    "Quantum Walk",
]

MAX_VARIANTS = 30
MAX_BLOCK_SIZE = 3
MAX_BLOCKS_PER_NODE = 12
 
EXPERIMENT_NAME = "exp02_fidelity_comparison"
OUT_DIR = Path.cwd() / "results" / "noise_predictors" / "exp02"
 
 
def get_sherbrooke_coupling_map():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    except ImportError:
        from qiskit.providers.fake_provider import FakeSherbrookeV2 as FakeSherbrooke
    backend = FakeSherbrooke()
    return backend.coupling_map
 
 
def build_alpha_p(noise_model):
    costs_1q, costs_2q = [], []
    for _gate_name, quantum_error in noise_model._default_quantum_errors.items():
        n = quantum_error.num_qubits
        kraus = Kraus(quantum_error)
        fid = average_gate_fidelity(kraus, Operator(np.eye(2 ** n)))
        (costs_1q if n == 1 else costs_2q).append(1.0 - fid)
 
    alpha = (sum(costs_1q) / len(costs_1q)) if costs_1q else 0.0
    p = (sum(costs_2q) / len(costs_2q)) if costs_2q else 0.0
    return alpha, p


 
def extract_circuit_features(qc: QuantumCircuit) -> dict:
    total_depth = qc.depth()
    depth_2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
    ops = qc.count_ops()
    cx_count = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)
    return {
        "total_depth": total_depth,
        "depth_2q": depth_2q,
        "cx_count": cx_count,
    }
 
 
 
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
 
 
def _replace_block(circuit, block_nodes, qubits_in_block, new_block):
    block_op_ids = {id(n.op) for n in block_nodes}
    qmap = {new_block.qubits[i]: q for i, q in enumerate(qubits_in_block)}
 
    new_circuit = circuit.copy_empty_like()
    emitted = False
    for instr in circuit.data:
        if id(instr.operation) in block_op_ids:
            if not emitted:
                for new_instr in new_block.data:
                    new_circuit.append(new_instr.operation, [qmap[q] for q in new_instr.qubits])
                emitted = True
            continue
        new_circuit.append(instr.operation, instr.qubits, instr.clbits)
    return new_circuit
 
 
def get_matrix_decompositions(circuit: QuantumCircuit, max_block_size: int = MAX_BLOCK_SIZE):
    dag = circuit_to_dag(circuit)
 
    collect_pass = CollectMultiQBlocks(max_block_size=max_block_size)
    collect_pass.property_set = PropertySet()
    collect_pass.run(dag)
    blocks = collect_pass.property_set.get("block_list", [])
 
    blocks = [b for b in blocks if len(b) >= 2]
 
    def _sort_key(b):
        two_q_count = sum(1 for n in b if len(n.qargs) == 2)
        qubit_idxs = tuple(sorted(circuit.find_bit(q).index for n in b for q in n.qargs))
        return (-two_q_count, qubit_idxs)
 
    blocks.sort(key=_sort_key)
    blocks = blocks[:MAX_BLOCKS_PER_NODE]
 
    successors = []
    for block_nodes in blocks:
        try:
            unitary, qubits_in_block = _block_to_unitary(block_nodes)
            new_block = qs_decomposition(unitary)
            new_circuit = _replace_block(circuit, block_nodes, qubits_in_block, new_block)
            successors.append(new_circuit)
        except KeyboardInterrupt:
            raise
        except BaseException:
            continue
    return successors
 
 
def collect_circuit_variants(start_circuit: QuantumCircuit, max_variants: int = MAX_VARIANTS) -> list:
    visited_hashes = set()
    variants = []
    queue = [start_circuit]
 
    while queue and len(variants) < max_variants:
        current_qc = queue.pop(0)
        qc_hash = qasm3.dumps(current_qc)
 
        if qc_hash in visited_hashes:
            continue
 
        visited_hashes.add(qc_hash)
        variants.append(current_qc)
 
        for next_qc in get_matrix_decompositions(current_qc):
            if len(variants) + len(queue) < max_variants * 2:
                queue.append(next_qc)
 
    return variants
 
 
def save_progress(experiment_data: dict, json_path: Path):
    tmp_path = json_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(experiment_data, f, indent=4)
    tmp_path.replace(json_path)
 
 
def load_existing_progress(json_path: Path) -> dict:
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if "circuits" in data:
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[!] Could not read existing results file ({e}); starting fresh.", flush=True)
    return {"timestamp": datetime.now().isoformat(), "circuits": []}
 
 
def circuit_key(base_name: str, num_qubits: int) -> str:
    return f"{base_name}__{num_qubits}q"
 
 
 
def run_experiment():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{EXPERIMENT_NAME}.json"
 
    print("=" * 70, flush=True)
    print("STARTING EXP02: EXACT QUANTUM FIDELITY (DENSITY MATRIX)", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Results file: {json_path}", flush=True)
 
    experiment_data = load_existing_progress(json_path)
    completed_keys = {
        circuit_key(c["base_name"], c["num_qubits"]) for c in experiment_data["circuits"]
    }
    if completed_keys:
        print(f"[*] Resuming: {len(completed_keys)} circuit(s) already done, will be skipped.", flush=True)
 
    print("[*] Loading noise model (IBM Sherbrooke)...", flush=True)
    noise_model = get_sherbrooke_noise_model()
    
    alpha, p = build_alpha_p(noise_model)
    f_1q, f_2q = get_avg_gate_fidelities(noise_model)
 
    coupling_map = get_sherbrooke_coupling_map()
    architecture = guess_architecture(coupling_map)
 
    experiment_data["theory_params"] = {
        "alpha": alpha, "p": p, "architecture": architecture, "f_1q": f_1q, "f_2q": f_2q
    }
    print(f"[*] Sherbrooke coupling map -> classified architecture: {architecture}", flush=True)
    print(f"[*] Derived RQC params: alpha={alpha:.6f}, p={p:.6f}", flush=True)
    print(f"[*] Derived EPA params: f_1q={f_1q:.6f}, f_2q={f_2q:.6f}", flush=True)
 
    # Create exact mathematical simulators
    sim_ideal = AerSimulator(method="statevector")
    sim_noisy = AerSimulator(noise_model=noise_model, method="density_matrix")
    
    hardware_basis = ["x", "sx", "rz", "id", "cx", "ecr", "cz"]
 
    base_circuits = []
    print("[*] Building benchmark suite...", flush=True)
    for b_name in BENCHMARKS:
        for n_qubits in QUBIT_RANGE:
            try:
                qc = get_mqt_circuit(b_name, num_qubits=n_qubits)
                # Unroll monolithic blocks (like QFT)
                qc = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
                base_circuits.append((b_name, n_qubits, qc))
                print(f"  + {b_name} ({n_qubits}q)", flush=True)
            except Exception as e:
                print(f"  ! Skipping {b_name} ({n_qubits}q): {e}", flush=True)
 
    start_time_total = time.time()
 
    for idx, (base_name, num_qubits, base_circuit) in enumerate(base_circuits, 1):
        key = circuit_key(base_name, num_qubits)
        if key in completed_keys:
            print(f"\n[{idx}/{len(base_circuits)}] Skipping (done): {base_name}, {num_qubits}q", flush=True)
            continue
 
        print("\n" + "-" * 115, flush=True)
        print(f"[{idx}/{len(base_circuits)}] {base_name}  ({num_qubits} qubits)", flush=True)
 
        t0 = time.time()
        variants = collect_circuit_variants(base_circuit, max_variants=MAX_VARIANTS)
        print(f"    - Generated {len(variants)} variant(s) in {time.time() - t0:.2f}s", flush=True)
 
        circuit_data = {
            "base_name": base_name,
            "num_qubits": num_qubits,
            "variants_tested": [],
        }
 
        for i, qc in enumerate(variants, 1):
            qc_state = qc.remove_final_measurements(inplace=False)
            
            state_ideal = transpile(qc_state, sim_ideal, optimization_level=0)
            state_noisy = transpile(qc_state, sim_noisy, basis_gates=hardware_basis, optimization_level=0)
            
            features = extract_circuit_features(state_noisy)
            
            state_ideal.save_statevector()
            state_noisy.save_density_matrix()

            result_ideal = sim_ideal.run(state_ideal).result()
            result_noisy = sim_noisy.run(state_noisy).result()
            
            ideal_state_result = result_ideal.data()['statevector']
            noisy_state_result = result_noisy.data()['density_matrix']
            
            fidelity_real_quantum = state_fidelity(ideal_state_result, noisy_state_result)
 
            theory = fidelity_from_circuit(state_noisy, alpha=alpha, p=p, architecture=architecture)
            epa_fid = predict_fidelity_epa(state_noisy, f_1q, f_2q)
 
            circuit_data["variants_tested"].append({
                "variant_id": i - 1,
                "qasm": qasm3.dumps(qc),
                "features": features,
                "metrics": {
                    "fidelity_real_quantum": fidelity_real_quantum,
                    "fidelity_theoretical_depth": theory["fidelity_solvable_model_depth"],
                    "fidelity_theoretical_eff": theory["fidelity_solvable_model_eff"],
                    "fidelity_epa": epa_fid,
                    "T_eff": theory["T_eff"]
                },
            })
 
            print(f"      [{i:02d}/{len(variants):02d}] "
                  f"Depth: {features['total_depth']:<3} | "
                  f"2Q Depth: {features['depth_2q']:<3} | "
                  f"Fid(TRUE): {fidelity_real_quantum:.4f} | "
                  f"Fid(EPA): {epa_fid:.4f} | "
                  f"Fid(th_eff): {theory['fidelity_solvable_model_eff']:.4f}", flush=True)
 
        experiment_data["circuits"].append(circuit_data)
        completed_keys.add(key)
 
        save_progress(experiment_data, json_path)
        print(f"    - [saved: {len(experiment_data['circuits'])} circuit(s) written to {json_path}]", flush=True)
 
    total_elapsed = time.time() - start_time_total
    print("\n" + "=" * 70, flush=True)
    print(f"DONE in {total_elapsed:.2f}s (this run). Results at: {json_path}", flush=True)
    print("=" * 70, flush=True)
 
 
if __name__ == "__main__":
    run_experiment()