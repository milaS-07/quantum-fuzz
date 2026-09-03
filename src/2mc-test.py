import json
import os
import sys
import importlib.util
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from qiskit import transpile
from qiskit.quantum_info import Kraus, Operator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.circuits import BENCHMARK_REGISTRY, get_mqt_circuit
from src.noise_parametrs import get_sherbrooke_noise_model

file_path = Path(__file__).resolve().parent / "2mc-obppp.py"

spec = importlib.util.spec_from_file_location("twomc", file_path)
twomc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(twomc)

def qiskit_error_to_ptm(quantum_error, k: int) -> np.ndarray:
    kraus_ops = Kraus(quantum_error).data
    return twomc.local_ptm_row_matrix(kraus_ops, k)

def qiskit_to_pqc(qc, ptm_1q: np.ndarray, ptm_2q: np.ndarray):
    qc_clean = qc.remove_final_measurements(inplace=False)
    basis = ["rx", "ry", "rz", "rxx", "rzz", "cx", "cz", "h", "s", "x", "y", "z"]
    qc_transpiled = transpile(qc_clean, basis_gates=basis, optimization_level=1)

    gates = []
    param_map = {}
    param_counter = 0

    for instruction in qc_transpiled.data:
        gname = instruction.operation.name.lower()
        q_indices = tuple(qc_transpiled.find_bit(q).index for q in instruction.qubits)

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
            if len(params) > 0 and hasattr(params[0], "parameters") and len(params[0].parameters) > 0:
                p_name = list(params[0].parameters)[0].name
                if p_name not in param_map:
                    param_map[p_name] = param_counter
                    param_counter += 1
                pidx = param_map[p_name]
            else:
                pidx = param_counter
                param_counter += 1

            gates.append(twomc.Gate(qubits=q_indices, generator=generator, param_idx=pidx))

    pqc = twomc.PQC(
        n_qubits=qc_transpiled.num_qubits,
        gates=gates,
        noise_1q=ptm_1q,
        noise_2q=ptm_2q,
    )
    return pqc, param_counter  # Return actual parameter count, 0 if none!

def run_all_2mc_benchmarks(
    qubit_range: list = [2, 4, 6, 8, 10, 12],
    n_theta_samples: int = 15,
    n_inner_samples: int = 10,
    base_results_dir: str = "results_2mc"
):
    plots_dir = os.path.join(base_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    noise_model = get_sherbrooke_noise_model()
    ptm_1q = qiskit_error_to_ptm(noise_model._default_quantum_errors["x"], k=1)
    ptm_2q = qiskit_error_to_ptm(noise_model._default_quantum_errors["cx"], k=2)

    rng = np.random.default_rng(42)

    print("=" * 70)
    print(f"POKRETANJE 2MC-OBPPP EKSPERIMENATA")
    print("=" * 70)

    for bench_name, meta in BENCHMARK_REGISTRY.items():
        category = meta["category"]
        is_scalable = meta["scalable"]

        cat_dir = os.path.join(base_results_dir, category.replace(" ", "_"))
        bench_dir = os.path.join(cat_dir, bench_name.replace(" ", "_").replace("'", ""))
        os.makedirs(bench_dir, exist_ok=True)

        metrics_path = os.path.join(bench_dir, "metrics_2mc.json")
        target_qubits = qubit_range if is_scalable else [qubit_range[0]]

        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            with open(metrics_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
            done_qubits = [int(q) for q in bench_data.get("results_per_qubit", {}).keys()]
            if all(q in done_qubits for q in target_qubits):
                print(f"\n---> Kolo: {bench_name} [{category}] — SVI KUBITI ZAVRŠENI.")
                continue
            else:
                print(f"\n---> Kolo: {bench_name} [{category}] — Nastavljam...")
        else:
            bench_data = {"metadata": {"name": bench_name}, "results_per_qubit": {}}

        expr_ideals, expr_noisys, mses, trainabilities, actual_qubit_list = [], [], [], [], []

        for n_q in target_qubits:
            q_key = str(n_q)
            
            if q_key in bench_data["results_per_qubit"]:
                res = bench_data["results_per_qubit"][q_key]
                expr_ideals.append(res["expressibility_ideal"])
                expr_noisys.append(res["expressibility_noisy"])
                mses.append(res["mse"])
                trainabilities.append(res["trainability_total_var"])
                actual_qubit_list.append(n_q)
                continue

            try:
                try:
                    qc = get_mqt_circuit(bench_name, num_qubits=n_q)
                except Exception:
                    try:
                        qc = get_mqt_circuit(bench_name)
                    except Exception as e:
                        print(f"  [GREŠKA] Neuspešno generisanje kola za n={n_q}. Preskačem.")
                        continue

                real_n = qc.num_qubits
                if str(real_n) in bench_data["results_per_qubit"]:
                    continue

                pqc, n_params = qiskit_to_pqc(qc, ptm_1q, ptm_2q)

                z_word = twomc.PauliWord(np.zeros(pqc.n_qubits, dtype=np.int8))
                z_word.labels[0] = twomc.Z_
                if pqc.n_qubits > 1:
                    z_word.labels[1] = twomc.Z_
                obs_terms = [(1.0, z_word)]

                # 1. Expressibility (Only for N <= 8 to prevent 4^n blowup and save time!)
                if real_n <= 8:
                    expr_ideal = max(0.0, twomc.estimate_expressibility(pqc, max(n_params, 1), n_theta_samples, n_pauli_pairs=10, apply_noise=False, rng=rng))
                    expr_noisy = max(0.0, twomc.estimate_expressibility(pqc, max(n_params, 1), n_theta_samples, n_pauli_pairs=10, apply_noise=True, rng=rng))
                else:
                    expr_ideal, expr_noisy = 0.0, 0.0 # Omitted for N > 8

                # 2. MSE (Robustness)
                mse = max(0.0, twomc.estimate_mse(pqc, obs_terms, max(n_params, 1), n_theta_samples, n_inner_samples, rng=rng))

                # 3. Trainability (Skip completely if no parameters, saving massive time)
                if n_params > 0:
                    t_arr = twomc.estimate_trainability(pqc, obs_terms, n_params, n_theta_samples, n_inner_samples, rng=rng)
                    train_var = max(0.0, float(np.sum(t_arr)))
                else:
                    train_var = 0.0

                expr_ideals.append(expr_ideal)
                expr_noisys.append(expr_noisy)
                mses.append(mse)
                trainabilities.append(train_var)
                actual_qubit_list.append(real_n)

                bench_data["results_per_qubit"][str(real_n)] = {
                    "num_qubits": real_n,
                    "num_params": n_params,
                    "expressibility_ideal": expr_ideal,
                    "expressibility_noisy": expr_noisy,
                    "mse": mse,
                    "trainability_total_var": train_var
                }

                expr_str = f"{expr_ideal:.3f}/{expr_noisy:.3f}" if real_n <= 8 else "N/A (N>8)"
                print(f"  Qubits: {real_n:2d} | Params: {n_params:3d} | Expr(I/N): {expr_str} | MSE: {mse:.4f} | TrainVar: {train_var:.6f}")

                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(bench_data, f, indent=4)

            except Exception as err:
                print(f"  [GREŠKA] Nije moguće izvršiti {bench_name} za n={n_q}: {err}")

        # Plotting
        if is_scalable and len(actual_qubit_list) > 1:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
            
            valid_idx = [i for i, q in enumerate(actual_qubit_list) if q <= 8]
            if valid_idx:
                ax1.plot([actual_qubit_list[i] for i in valid_idx], [expr_ideals[i] for i in valid_idx], "o-", color="tab:blue", label="Ideal")
                ax1.plot([actual_qubit_list[i] for i in valid_idx], [expr_noisys[i] for i in valid_idx], "s--", color="tab:red", label="Noisy")
            ax1.set_ylabel("Expressibility\n(M2^2 deviation)")
            ax1.set_title(f"{bench_name} ({category}) - 2MC-OBPPP Metrics")
            ax1.legend()
            ax1.grid(True, linestyle=":", alpha=0.6)

            ax2.plot(actual_qubit_list, mses, "d-", color="purple")
            ax2.set_ylabel("Noise Robustness\n(MSE)")
            ax2.grid(True, linestyle=":", alpha=0.6)

            ax3.plot(actual_qubit_list, trainabilities, "^-", color="green")
            ax3.set_xlabel("Broj kubita (N)")
            ax3.set_ylabel("Trainability\n(Total Gradient Variance)")
            ax3.set_yscale("symlog", linthresh=1e-5)
            ax3.grid(True, linestyle=":", alpha=0.6)

            fig.tight_layout()
            plt.savefig(os.path.join(bench_dir, "metrics_2mc_vs_qubits.png"), dpi=200)
            plt.close()

def main():
    target_dir = os.path.join(PROJECT_ROOT, "results_2mc")
    run_all_2mc_benchmarks(
        qubit_range=[2, 4, 6, 8, 10, 12], 
        n_theta_samples=15,
        n_inner_samples=10,
        base_results_dir=target_dir
    )

if __name__ == "__main__":
    main()