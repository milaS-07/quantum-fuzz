import json
import os
import statistics
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

from qiskit import transpile
from qiskit_aer import AerSimulator

from src.circuits import BENCHMARK_REGISTRY, get_mqt_circuit
from src.helper_functions import (
    calculate_tvd,
    calculate_fidelity,
    calculate_js_divergence
)

from src.noise_parametrs import get_sherbrooke_noise_model




def run_all_benchmarks(
    qubit_range: list = [2, 3, 4, 5, 6, 7, 8],
    shots: int = 1000,
    base_results_dir: str = "results"
):
    plots_dir = os.path.join(base_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    noise_model = get_sherbrooke_noise_model()
    sim_ideal = AerSimulator(max_parallel_threads=0)
    sim_noisy = AerSimulator(noise_model=noise_model, max_parallel_threads=0)

    basis_gates = ["id", "rz", "sx", "x", "cx"]

    print("=" * 70)
    print(f"POKRETANJE EKSPERIMENATA (Shots: {shots}, Qubit opseg: {qubit_range})")
    print("=" * 70)

    for bench_name, meta in BENCHMARK_REGISTRY.items():
        category = meta["category"]
        is_scalable = meta["scalable"]

        cat_dir = os.path.join(base_results_dir, category.replace(" ", "_"))
        bench_dir = os.path.join(cat_dir, bench_name.replace(" ", "_").replace("'", ""))
        os.makedirs(bench_dir, exist_ok=True)

        metrics_path = os.path.join(bench_dir, "metrics.json")

        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            print(f"\n---> Kolo: {bench_name} [{category}] — VEĆ ZAVRŠENO, preskačem.")
            continue

        target_qubits = qubit_range if is_scalable else [qubit_range[0]]

        bench_data = {
            "metadata": {
                "name": bench_name,
                "category": category,
                "scalable": is_scalable,
                "shots": shots
            },
            "results_per_qubit": {}
        }

        fidelities, tvds, jsds, actual_qubit_list = [], [], [], []
        zero_fidelity_count = 0

        print(f"\n---> Kolo: {bench_name} [{category}] (Skalabilno: {is_scalable})")

        for n_q in target_qubits:
            try:
                if zero_fidelity_count >= 2:
                    print(f"  Qubits: {n_q:2d} | [PRESKOČENO - Signal u potpunom šumu] Fidelity: 0.0000 | TVD: 1.0000 | JSD: 1.0000")
                    fid, tvd_val, jsd_val = 0.0, 1.0, 1.0
                    real_n = n_q

                    fidelities.append(fid)
                    tvds.append(tvd_val)
                    jsds.append(jsd_val)
                    actual_qubit_list.append(real_n)

                    q_key = str(real_n)
                    bench_data["results_per_qubit"][q_key] = {
                        "num_qubits": real_n,
                        "depth": -1,
                        "gate_count": {},
                        "fidelity": fid,
                        "tvd": tvd_val,
                        "js_divergence": jsd_val,
                        "counts_ideal": {},
                        "counts_noisy": {}
                    }
                    continue

                try:
                    qc = get_mqt_circuit(bench_name, num_qubits=n_q)
                except Exception:
                    qc = get_mqt_circuit(bench_name)

                real_n = qc.num_qubits

                qc_t = transpile(qc, basis_gates=basis_gates, optimization_level=1)

                counts_ideal = sim_ideal.run(qc_t, shots=shots).result().get_counts()
                counts_noisy = sim_noisy.run(qc_t, shots=shots).result().get_counts()

                fid = calculate_fidelity(counts_ideal, counts_noisy, shots)
                tvd_val = calculate_tvd(counts_ideal, counts_noisy, shots)
                jsd_val = calculate_js_divergence(counts_ideal, counts_noisy, shots)

                if fid == 0.0:
                    zero_fidelity_count += 1
                else:
                    zero_fidelity_count = 0

                fidelities.append(fid)
                tvds.append(tvd_val)
                jsds.append(jsd_val)
                actual_qubit_list.append(real_n)

                q_key = str(real_n)
                bench_data["results_per_qubit"][q_key] = {
                    "num_qubits": real_n,
                    "depth": qc_t.depth(),
                    "gate_count": dict(qc_t.count_ops()),
                    "fidelity": fid,
                    "tvd": tvd_val,
                    "js_divergence": jsd_val,
                    "counts_ideal": counts_ideal,
                    "counts_noisy": counts_noisy
                }

                print(f"  Qubits: {real_n:2d} | Fidelity: {fid:.4f} | TVD: {tvd_val:.4f} | JSD: {jsd_val:.4f}")

            except Exception as err:
                print(f"  [GREŠKA] Nije moguće izvršiti {bench_name} za n={n_q}: {err}")

        if not fidelities:
            continue

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(bench_data, f, indent=4)

        if is_scalable and len(actual_qubit_list) > 1:
            fig, ax1 = plt.subplots(figsize=(7, 4.5))
            ax2 = ax1.twinx()

            ax1.plot(actual_qubit_list, fidelities, "o-", color="tab:blue", label="Fidelity")
            ax2.plot(actual_qubit_list, tvds, "s--", color="tab:red", label="TVD")

            ax1.set_xlabel("Broj kubita (N)")
            ax1.set_ylabel("Classical Fidelity", color="tab:blue")
            ax2.set_ylabel("TVD", color="tab:red")
            plt.title(f"{bench_name} ({category})")
            ax1.grid(True, linestyle=":", alpha=0.6)

            fig.tight_layout()
            plt.savefig(os.path.join(bench_dir, "metrics_vs_qubits.png"), dpi=200)
            plt.close()

    all_raw_results = {}
    grouped_metrics = {}

    for bench_name, meta in BENCHMARK_REGISTRY.items():
        category = meta["category"]
        cat_dir = os.path.join(base_results_dir, category.replace(" ", "_"))
        bench_dir = os.path.join(cat_dir, bench_name.replace(" ", "_").replace("'", ""))
        metrics_path = os.path.join(bench_dir, "metrics.json")

        if not (os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0):
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        all_raw_results[bench_name] = bench_data

        per_q = bench_data["results_per_qubit"].values()
        fids = [r["fidelity"] for r in per_q]
        tvds_ = [r["tvd"] for r in per_q]
        jsds_ = [r["js_divergence"] for r in per_q]

        if category not in grouped_metrics:
            grouped_metrics[category] = {"fidelities": [], "tvds": [], "jsds": []}
        grouped_metrics[category]["fidelities"].extend(fids)
        grouped_metrics[category]["tvds"].extend(tvds_)
        grouped_metrics[category]["jsds"].extend(jsds_)

    with open(os.path.join(base_results_dir, "all_experiments_raw.json"), "w", encoding="utf-8") as f:
        json.dump(all_raw_results, f, indent=4)

    summary_grouped = {}
    for cat, vals in grouped_metrics.items():
        if not vals["fidelities"]:
            continue
        summary_grouped[cat] = {
            "mean_fidelity": statistics.mean(vals["fidelities"]),
            "mean_tvd": statistics.mean(vals["tvds"]),
            "mean_jsd": statistics.mean(vals["jsds"]),
            "min_fidelity": min(vals["fidelities"]),
            "max_tvd": max(vals["tvds"])
        }

    with open(os.path.join(base_results_dir, "summary_grouped.json"), "w", encoding="utf-8") as f:
        json.dump(summary_grouped, f, indent=4)

    categories = list(summary_grouped.keys())
    if categories:
        avg_fids = [summary_grouped[c]["mean_fidelity"] for c in categories]
        avg_tvds = [summary_grouped[c]["mean_tvd"] for c in categories]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(categories))
        width = 0.35

        ax.bar([p - width / 2 for p in x], avg_fids, width, label="Prosečni Fidelity", color="navy")
        ax.bar([p + width / 2 for p in x], avg_tvds, width, label="Prosečni TVD", color="crimson")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha="right")
        ax.set_ylabel("Vrednost metrike")
        ax.set_title("Poređenje performansi po algebarskim kategorijama kola")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "overall_category_comparison.png"), dpi=300)
        plt.close()

    print("\n" + "=" * 70)
    print("EKSPERIMENTI USPEŠNO ZAVRŠENI!")
    print(f"Rezultati i grafici su sačuvani u folderu: '{base_results_dir}/'")
    print("=" * 70)


def main():
    target_dir = r"D:\results_quantum"
    run_all_benchmarks(
        qubit_range=[2, 4, 6, 8, 10, 12, 14, 16, 18],
        shots=1000,
        base_results_dir=target_dir
    )


if __name__ == "__main__":
    main()