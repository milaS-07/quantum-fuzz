import sys
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Kraus, Operator

from src.noise_parametrs import get_sherbrooke_noise_model
from src.circuits import get_mqt_circuit

file_path = Path(__file__).parent / "2mc-obppp.py"
if not file_path.exists():
    file_path = PROJECT_ROOT / "2mc-obppp.py"

spec = importlib.util.spec_from_file_location("twomc", file_path)
twomc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twomc)

def qiskit_error_to_ptm(quantum_error, k: int) -> np.ndarray:
    kraus_ops = Kraus(quantum_error).data
    return twomc.local_ptm_row_matrix(kraus_ops, k)

def qiskit_to_pqc(qc, ptm_1q: np.ndarray, ptm_2q: np.ndarray):
    qc_clean = qc.remove_final_measurements(inplace=False)
    
    # CRITICAL FIX: Include Cliffords in basis gates so they aren't destroyed/randomized
    basis = ["rx", "ry", "rz", "rxx", "rzz", "cx", "cz", "h", "s", "x", "y", "z"]
    qc_transpiled = transpile(qc_clean, basis_gates=basis, optimization_level=1)

    gates = []
    param_map = {}
    param_counter = 0

    for instruction in qc_transpiled.data:
        gname = instruction.operation.name.lower()
        q_indices = tuple(qc_transpiled.find_bit(q).index for q in instruction.qubits)

        # Handle fixed Cliffords by calculating their exact PTM
        if gname in ["cx", "cz", "h", "s", "x", "y", "z"]:
            op_mat = Operator(instruction.operation).data
            ptm = twomc.local_ptm_row_matrix([op_mat], len(q_indices))
            gates.append(twomc.Gate(qubits=q_indices, clifford_ptm=ptm))
            continue

        generator = None
        if gname in ["rx", "rxx"]:
            generator = tuple([twomc.X_] * len(q_indices))
        elif gname in ["ry", "ryy"]:
            generator = tuple([twomc.Y_] * len(q_indices))
        elif gname in ["rz", "p", "rzz"]:
            generator = tuple([twomc.Z_] * len(q_indices))

        if generator is not None:
            params = instruction.operation.params
            # Extract Qiskit Parameter names properly if it's a symbolic parameter
            if len(params) > 0 and hasattr(params[0], "parameters") and len(params[0].parameters) > 0:
                p_name = list(params[0].parameters)[0].name
                if p_name not in param_map:
                    param_map[p_name] = param_counter
                    param_counter += 1
                pidx = param_map[p_name]
            else:
                pidx = param_counter
                param_counter += 1

            gates.append(
                twomc.Gate(qubits=q_indices, generator=generator, param_idx=pidx)
            )

    pqc = twomc.PQC(
        n_qubits=qc_transpiled.num_qubits,
        gates=gates,
        noise_1q=ptm_1q,
        noise_2q=ptm_2q,
    )
    return pqc, max(param_counter, 1)

def run_quick_test():
    print("=" * 60)
    print("TESTING 2MC-OBPPP WITH MAIN NOISE MODEL & MQT BENCH")
    print("=" * 60)

    noise_model = get_sherbrooke_noise_model()
    err_1q = noise_model._default_quantum_errors["x"]
    err_2q = noise_model._default_quantum_errors["cx"]

    ptm_1q = qiskit_error_to_ptm(err_1q, k=1)
    ptm_2q = qiskit_error_to_ptm(err_2q, k=2)

    qc = get_mqt_circuit("qaoa", num_qubits=3)
    pqc, n_params = qiskit_to_pqc(qc, ptm_1q, ptm_2q)

    rng = np.random.default_rng(42)

    print(f"Circuit: {qc.name} (Qubits: {pqc.n_qubits}, Total Gates: {len(pqc.gates)}, Params: {n_params})\n")

    # 2. Expressibility (Eq. 5 / Fig 4)
    print("--- 1. Expressibility ---")
    expr_ideal = twomc.estimate_expressibility(
        pqc, n_params=n_params, n_theta_samples=15, n_pauli_pairs=10, apply_noise=False, rng=rng
    )
    expr_noisy = twomc.estimate_expressibility(
        pqc, n_params=n_params, n_theta_samples=15, n_pauli_pairs=10, apply_noise=True, rng=rng
    )
    print(f"Ideal Expressibility (deviation from 2-design): {expr_ideal:.6f}")
    print(f"Noisy Expressibility (deviation from 2-design): {expr_noisy:.6f}\n")

    z_word = twomc.PauliWord(np.zeros(pqc.n_qubits, dtype=np.int8))
    z_word.labels[0] = twomc.Z_
    obs_terms = [(1.0, z_word)]

    print("--- 2. Noise Robustness (MSE) ---")
    mse = twomc.estimate_mse(
        pqc, obs_terms, n_params, 
        n_theta_samples=15, n_inner_samples=10, rng=rng
    )
    print(f"Observable: Z on Qubit 0")
    print(f"Mean Squared Error (MSE): {mse:.6f}\n")

    # 4. Trainability / Gradient Variance (Eq. 4 / Fig 3)
    print("--- 3. Trainability (Gradient Variance) ---")
    trainability_arr = twomc.estimate_trainability(
        pqc, obs_terms, n_params, 
        n_theta_samples=15, n_inner_samples=10, rng=rng
    )
    total_var = np.sum(trainability_arr)
    print(f"Total Gradient Variance: {total_var:.6f}")
    print(f"(Exponentially small values indicate a Barren Plateau)\n")

    print("--- 4. Noise Bottleneck Map ---")
    def depol_kraus(p):
        return [
            np.sqrt(1 - 0.75 * p) * np.eye(2),
            np.sqrt(p / 4) * np.array([[0, 1], [1, 0]]),
            np.sqrt(p / 4) * np.array([[0, -1j], [1j, 0]]),
            np.sqrt(p / 4) * np.array([[1, 0], [0, -1]])
        ]
        
    bottleneck_dict = twomc.estimate_noise_bottleneck(
        pqc, obs_terms, n_params, 
        kraus_fn=depol_kraus, base_param=0.01, delta=0.001,
        n_theta_samples=15, n_inner_samples=10, rng=rng
    )
    
    print("Sensitivity per Qubit:")
    for q, val in bottleneck_dict.items():
        print(f"  Qubit {q}: {val:.6f}")
        
    print("\nRendering Heatmap...")
    twomc.plot_heatmap(
        pqc.n_qubits, 
        bottleneck_dict, 
        "Noise Bottleneck Map (Sensitivity)", 
        grid_shape=(1, pqc.n_qubits)
    )

if __name__ == "__main__":
    run_quick_test()