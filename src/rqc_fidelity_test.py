import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import time
import math
from datetime import datetime
from typing import Literal, Union
from qiskit import QuantumCircuit, qasm3

from circuits import get_mqt_circuit
from transformations import get_matrix_decompositions 

# =====================================================================
# THEORETICAL FIDELITY FORMULAS (From RQC Paper)
# =====================================================================

def f4_exact(alpha: float) -> float:
    a2 = alpha * alpha
    poly = (-(a2**5) + 12.5 * a2**4 - 64 * a2**3 + 138 * a2**2 - 144 * a2 + 36)
    return math.exp(-a2) * poly / 36.0

def Delta_factor(L: int, alpha: float) -> float:
    f4 = f4_exact(alpha)
    return (2.0 ** L) * (3.0 * f4 + 1.0) ** (L / 2.0)

EULER_GAMMA = 0.5772156649015329

def delta_full(L: int, p: float) -> float:
    if L <= 1: return 4.0 ** L
    return (4.0 ** L) * math.exp(-0.75 * p * (L - math.log(L) - EULER_GAMMA))

def delta_1d(L: int, p: float) -> float:
    return (4.0 ** L) * math.exp(-0.1875 * p * L * (L - 1))

def delta_dD(L: int, p: float, d: int) -> float:
    if d == 1: return delta_1d(L, p)
    return (4.0 ** L) * math.exp(-0.75 * (d - 0.5) * p * (L ** (1.0 + 1.0 / d)))

Architecture = Literal["full", "1d", "2d", "3d"]

def average_fidelity_theoretical(
    L: int,
    T: Union[int, float],
    p: float = 0.0,
    alpha: float = 0.0,
    architecture: Architecture = "1d",
) -> float:
    if L < 1 or T < 0: return 1.0
    if p <= 0:
        delta = 4.0 ** L
    elif architecture == "full": delta = delta_full(L, p)
    elif architecture == "1d":   delta = delta_1d(L, p)
    elif architecture == "2d":   delta = delta_dD(L, p, 2)
    elif architecture == "3d":   delta = delta_dD(L, p, 3)
    else: delta = delta_full(L, p)

    d_val = Delta_factor(L, alpha)
    base = 4.0 ** L - 1.0
    bracket = (delta - 1.0) * (d_val - 1.0) / (base * base)
    
    floor = 1.0 / (2.0 ** L)
    return (1.0 - floor) * (bracket ** T) + floor


# =====================================================================
# CIRCUIT GENERATION & FEATURE EXTRACTION
# =====================================================================

def extract_circuit_features(qc: QuantumCircuit) -> dict:
    total_depth = qc.depth()
    depth_2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
    ops = qc.count_ops()
    cx_count = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)
    
    L = qc.num_qubits
    expected_gates_per_layer = L / 2.0
    
    # Calculate effective depth for the sparse algorithm assumption
    effective_2q_depth = (cx_count / expected_gates_per_layer) if expected_gates_per_layer > 0 else 0
    
    return {
        "total_depth": total_depth,
        "depth_2q": depth_2q,
        "cx_count": cx_count,
        "effective_2q_depth": effective_2q_depth
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
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=4)

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


# =====================================================================
# MAIN EXPERIMENT RUNNER
# =====================================================================

def run_theoretical_experiment(benchmark_circuits: list, experiment_name: str = "theoretical_fidelities"):
    project_root = Path.cwd()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{experiment_name}.json"
    
    print("=" * 70, flush=True)
    print("STARTING THEORETICAL FIDELITY CALCULATIONS", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Total benchmark circuits loaded: {len(benchmark_circuits)}", flush=True)
    
    experiment_data = load_existing_progress(json_path)
    completed_keys = {
        circuit_key(c["base_name"], c["num_qubits"])
        for c in experiment_data["circuits"]
    }
    
    THEORETICAL_ALPHA = 0.07   # Roughly 0.5% - 0.7% gate error
    THEORETICAL_P = 0.001      # Swap/Routing omission error
    THEORETICAL_ARCH = "1d"    # Heavy hex is essentially sparse/1D
    
    start_time_total = time.time()
    
    for idx, base_circuit in enumerate(benchmark_circuits, 1):
        num_qubits = base_circuit.num_qubits
        circ_name = base_circuit.name if base_circuit.name != "circuit" else f"circ_{idx}"

        if circuit_key(circ_name, num_qubits) in completed_keys:
            print(f"[{idx}/{len(benchmark_circuits)}] Skipping (already completed): {circ_name}, {num_qubits} qubits", flush=True)
            continue
        
        print("\n" + "-" * 50, flush=True)
        print(f"[{idx}/{len(benchmark_circuits)}] Processing Circuit: {circ_name}", flush=True)
        print(f"    - Qubits: {num_qubits}", flush=True)
        
        t0 = time.time()
        variants = collect_circuit_variants(base_circuit, max_variants=30)
        print(f"    - Generated {len(variants)} variants in {time.time() - t0:.2f}s", flush=True)
        
        circuit_data = {
            "base_name": circ_name,
            "num_qubits": num_qubits,
            "variants_tested": []
        }
        
        for i, qc in enumerate(variants, 1):
            features = extract_circuit_features(qc)
            
            raw_T = features["depth_2q"]
            eff_T = features["effective_2q_depth"]
            
            # Calculate theoretical fidelity using the formula
            th_fid_raw = average_fidelity_theoretical(
                L=num_qubits, T=raw_T, p=THEORETICAL_P, alpha=THEORETICAL_ALPHA, architecture=THEORETICAL_ARCH
            )
            th_fid_eff = average_fidelity_theoretical(
                L=num_qubits, T=eff_T, p=THEORETICAL_P, alpha=THEORETICAL_ALPHA, architecture=THEORETICAL_ARCH
            )
            
            circuit_data["variants_tested"].append({
                "variant_id": i - 1,
                "features": features,
                "theoretical_metrics": {
                    "fidelity_raw": th_fid_raw,
                    "fidelity_eff": th_fid_eff
                }
            })
            
            print(f"      [Variant {i-1:02d}] 2Q Depth: {raw_T:<3} (Eff: {eff_T:5.1f}) | "
                  f"Th. Fid (Raw): {th_fid_raw:.4f} | Th. Fid (Eff): {th_fid_eff:.4f}")
            
        experiment_data["circuits"].append(circuit_data)
        completed_keys.add(circuit_key(circ_name, num_qubits))
        save_progress(experiment_data, json_path)

    total_elapsed = time.time() - start_time_total
    print("\n" + "=" * 70, flush=True)
    print(f"ALL CALCULATIONS COMPLETE in {total_elapsed:.2f} seconds")
    print(f"Results saved to: {json_path}")
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
        # Added 18 to the list!
        for n_qubits in [3, 6, 9, 12, 15, 18]:
            try:
                qc = get_mqt_circuit(b_name, num_qubits=n_qubits)
                test_circuits.append(qc)
                print(f"  + Added: {b_name} ({n_qubits} qubits)", flush=True)
            except Exception as e:
                print(f"  ! Skipping {b_name} with {n_qubits} qubits: {e}", flush=True)
                
    # Run the theoretical extraction
    run_theoretical_experiment(test_circuits, experiment_name="theoretical_fidelities")