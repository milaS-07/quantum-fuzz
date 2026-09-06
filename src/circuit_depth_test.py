import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_aer import AerSimulator

from circuits import get_mqt_circuit
from helper_functions import calculate_tvd, calculate_fidelity, calculate_js_divergence
from transformations import get_matrix_decompositions 
from noise_parametrs import get_sherbrooke_noise_model


def calculate_required_shots(num_qubits: int, target_tvd_error: float = 0.05) -> int:
    states = 2 ** num_qubits
    shots = int(states / (2 * (target_tvd_error ** 2)))
    return max(1000, min(shots, 100000))


def extract_circuit_features(qc: QuantumCircuit) -> dict:
    total_depth = qc.depth()
    depth_2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
    ops = qc.count_ops()
    cx_count = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)
    
    return {
        "total_depth": total_depth,
        "depth_2q": depth_2q,
        "cx_count": cx_count
    }


def collect_circuit_variants(start_circuit: QuantumCircuit, max_variants: int = 30) -> list:
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
        
        successors = get_matrix_decompositions(current_qc)
        for next_qc, _ in successors:
            if len(variants) + len(queue) < max_variants * 2:
                queue.append(next_qc)
                
    return variants


def save_progress(experiment_data: dict, json_path: Path):
    """Write current results to disk. Called after every circuit so a crash/interrupt
    never loses more than the circuit currently in progress."""
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=4)


def load_existing_progress(json_path: Path) -> dict:
    """If a results file from a previous (possibly interrupted) run exists, load it
    so already-completed circuits can be skipped instead of redone."""
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


def run_correlation_experiment(benchmark_circuits: list, experiment_name: str = "depth_noise_correlation"):
    project_root = Path.cwd()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{experiment_name}.json"
    
    print("=" * 70, flush=True)
    print("STARTING CORRELATION EXPERIMENT", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Total benchmark circuits loaded: {len(benchmark_circuits)}", flush=True)
    print(f"[*] Results will be saved incrementally to: {json_path}", flush=True)

    # --- Resume logic: pick up any existing progress from a previous run ---
    experiment_data = load_existing_progress(json_path)
    completed_keys = {
        circuit_key(c["base_name"], c["num_qubits"])
        for c in experiment_data["circuits"]
    }
    if completed_keys:
        print(f"[*] Found existing progress: {len(completed_keys)} circuit(s) already completed, will be skipped.", flush=True)

    print("[*] Loading noise model (IBM Sherbrooke)...", flush=True)
    
    start_time_total = time.time()
    noise_model = get_sherbrooke_noise_model()
    
    sim_ideal = AerSimulator()
    sim_noisy = AerSimulator(noise_model=noise_model, method="statevector")
    
    # Rebuild the running metric lists from whatever's already in experiment_data,
    # so the final correlation still includes previously-completed circuits.
    all_depths = []
    all_2q_depths = []
    all_fidelities = []
    all_tvds = []
    for c in experiment_data["circuits"]:
        for v in c["variants_tested"]:
            all_depths.append(v["features"]["total_depth"])
            all_2q_depths.append(v["features"]["depth_2q"])
            all_fidelities.append(v["metrics"]["fidelity"])
            all_tvds.append(v["metrics"]["tvd"])
    
    for idx, base_circuit in enumerate(benchmark_circuits, 1):
        num_qubits = base_circuit.num_qubits
        circ_name = base_circuit.name if base_circuit.name != "circuit" else f"circ_{idx}"

        if circuit_key(circ_name, num_qubits) in completed_keys:
            print(f"\n[{idx}/{len(benchmark_circuits)}] Skipping (already completed): {circ_name}, {num_qubits} qubits", flush=True)
            continue

        shots = calculate_required_shots(num_qubits)
        
        print("\n" + "-" * 50, flush=True)
        print(f"[{idx}/{len(benchmark_circuits)}] Processing Circuit: {circ_name}", flush=True)
        print(f"    - Qubits: {num_qubits}", flush=True)
        print(f"    - Shots per execution: {shots}", flush=True)
        
        t0 = time.time()
        variants = collect_circuit_variants(base_circuit, max_variants=30)
        print(f"    - Generated variants: {len(variants)} (took {time.time() - t0:.2f}s)", flush=True)
        
        circuit_data = {
            "base_name": circ_name,
            "num_qubits": num_qubits,
            "shots_used": shots,
            "variants_tested": []
        }
        
        for i, qc in enumerate(variants, 1):
            features = extract_circuit_features(qc)
            
            meas_circuit = qc.remove_final_measurements(inplace=False)
            meas_circuit.measure_all()
            
            # Prevođenje nepoznatih/kompozitnih kapija uz optimization_level=0 (bez optimizacije kola)
            meas_circuit_ideal = transpile(meas_circuit, sim_ideal, optimization_level=0)
            meas_circuit_noisy = transpile(meas_circuit, sim_noisy, optimization_level=0)
            
            counts_ideal = sim_ideal.run(meas_circuit_ideal, shots=shots).result().get_counts()
            counts_noisy = sim_noisy.run(meas_circuit_noisy, shots=shots).result().get_counts()
            
            fidelity = calculate_fidelity(counts_ideal, counts_noisy, shots)
            tvd = calculate_tvd(counts_ideal, counts_noisy, shots)
            jsd = calculate_js_divergence(counts_ideal, counts_noisy, shots)
            
            all_depths.append(features["total_depth"])
            all_2q_depths.append(features["depth_2q"])
            all_fidelities.append(fidelity)
            all_tvds.append(tvd)
            
            circuit_data["variants_tested"].append({
                "variant_id": i - 1,
                "features": features,
                "metrics": {
                    "fidelity": fidelity,
                    "tvd": tvd,
                    "jsd": jsd
                }
            })
            
            print(f"      [Variant {i:02d}/{len(variants):02d}] "
                  f"Total Depth: {features['total_depth']:<3} | "
                  f"2Q Depth: {features['depth_2q']:<3} | "
                  f"Fidelity: {fidelity:.4f} | "
                  f"TVD: {tvd:.4f}", flush=True)
            
        experiment_data["circuits"].append(circuit_data)
        completed_keys.add(circuit_key(circ_name, num_qubits))

        # Save after every completed circuit, not just at the very end.
        save_progress(experiment_data, json_path)
        print(f"    - [saved progress: {len(experiment_data['circuits'])} circuit(s) written to {json_path}]", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("CALCULATING PEARSON CORRELATIONS", flush=True)
    print("=" * 70, flush=True)

    pearson_depth_fid, p_val1 = stats.pearsonr(all_depths, all_fidelities)
    pearson_2q_fid, p_val2 = stats.pearsonr(all_2q_depths, all_fidelities)
    pearson_2q_tvd, p_val3 = stats.pearsonr(all_2q_depths, all_tvds)
    
    print(f"Total Depth vs Fidelity: r = {pearson_depth_fid:.4f} (p-value: {p_val1:.4e})", flush=True)
    print(f"2-Qubit Depth vs Fidelity: r = {pearson_2q_fid:.4f} (p-value: {p_val2:.4e})", flush=True)
    print(f"2-Qubit Depth vs TVD:      r = {pearson_2q_tvd:.4f} (p-value: {p_val3:.4e})", flush=True)

    experiment_data["statistical_analysis"] = {
        "pearson_total_depth_vs_fidelity": {"r": float(pearson_depth_fid), "p_value": float(p_val1)},
        "pearson_2q_depth_vs_fidelity": {"r": float(pearson_2q_fid), "p_value": float(p_val2)},
        "pearson_2q_depth_vs_tvd": {"r": float(pearson_2q_tvd), "p_value": float(p_val3)}
    }
    
    save_progress(experiment_data, json_path)
        
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(all_2q_depths, all_fidelities, alpha=0.7, color='blue')
    plt.title("2-Qubit Depth vs Fidelity")
    plt.xlabel("2-Qubit Circuit Depth")
    plt.ylabel("Bhattacharyya Fidelity")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(all_2q_depths, all_tvds, alpha=0.7, color='red')
    plt.title("2-Qubit Depth vs TVD")
    plt.xlabel("2-Qubit Circuit Depth")
    plt.ylabel("Total Variation Distance")
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = results_dir / f"{experiment_name}_plots.png"
    plt.savefig(plot_path)
    plt.close()

    total_elapsed = time.time() - start_time_total
    print("\n" + "=" * 70, flush=True)
    print(f"EXPERIMENT COMPLETE in {total_elapsed:.2f} seconds (this run)", flush=True)
    print(f"Results saved to:\n  - {json_path}\n  - {plot_path}", flush=True)
    print("=" * 70, flush=True)

    return experiment_data


if __name__ == "__main__":
    scalable_benchmarks = [
        "Quantum Fourier Transformation (QFT)",
        "Quantum Approximation Optimization Algorithm (QAOA)",
        "GHZ State",
        "Real Amplitudes ansatz with Random Parameters"
    ]
    
    test_circuits = []
    
    print("Building benchmark suite...", flush=True)
    for b_name in scalable_benchmarks:
        for n_qubits in [3, 6, 9, 12, 15]:
            try:
                qc = get_mqt_circuit(b_name, num_qubits=n_qubits)
                test_circuits.append(qc)
                print(f"  + Added: {b_name} ({n_qubits} qubits)", flush=True)
            except Exception as e:
                print(f"  ! Skipping {b_name} with {n_qubits} qubits: {e}", flush=True)
                
    run_correlation_experiment(test_circuits, experiment_name="depth_noise_correlation")