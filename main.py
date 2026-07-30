import json
import os
import matplotlib.pyplot as plt
import statistics

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from src.circuits import *
from src.noise_models import *
from src.helper_functions import *


def run_experiment(
    circuit_fn,
    noise_fn,
    exp_name,
    qubit_range=range(2, 11),
    shots: int = 10000,
):
    exp_dir = os.path.join("results", exp_name)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sim_ideal = AerSimulator()

    experiment_data = {
        "metadata": {
            "experiment_name": exp_name,
            "shots": shots,
            "qubit_range": list(qubit_range),
        },
        "results_per_qubit": {},
    }

    fidelities_ideal = []
    fidelities_noisy = []
    tvds = []

    for n_qubits in qubit_range:
        qc = circuit_fn(num_qubits=n_qubits)
        noise_model = noise_fn(qc)

        sim_noisy = AerSimulator(noise_model=noise_model)

        #bez šuma
        qc_ideal = transpile(qc, sim_ideal)
        counts_ideal = sim_ideal.run(qc_ideal, shots=shots).result().get_counts()

        #sa šumom
        qc_noisy = transpile(qc, sim_noisy)
        counts_noisy = sim_noisy.run(qc_noisy, shots=shots).result().get_counts()

        fid_ideal = calculate_ghz_fidelity(counts_ideal, shots, n_qubits)
        fid_noisy = calculate_ghz_fidelity(counts_noisy, shots, n_qubits)
        tvd_val = calculate_tvd(counts_ideal, counts_noisy, shots)

        fidelities_ideal.append(fid_ideal)
        fidelities_noisy.append(fid_noisy)
        tvds.append(tvd_val)

        experiment_data["results_per_qubit"][n_qubits] = {
            "fidelity_ideal": fid_ideal,
            "fidelity_noisy": fid_noisy,
            "tvd": tvd_val,
            "counts_ideal": counts_ideal,
            "counts_noisy": counts_noisy,
        }

        print(f"N={n_qubits:2d} | Fidelity Noisy: {fid_noisy:.4f} | TVD: {tvd_val:.4f}")

        if n_qubits == qubit_range[-1]:
            fig = plot_histogram(
                [counts_ideal, counts_noisy],
                legend=["Bez šuma", "Sa šumom"],
                sort="value_desc",
                number_to_keep=10,
                title=f"Histogram za N={n_qubits} kubita",
            )
            fig.savefig(
                os.path.join(plots_dir, f"histogram_top10_N{n_qubits}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.plot(qubit_range, fidelities_noisy, "o-", label="Sa šumom")
    plt.xlabel("Broj kubita (N)")
    plt.ylabel("Fidelity")
    plt.title(f"Eksperiment: {exp_name}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(qubit_range)
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "fidelity_vs_qubits.png"), bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(qubit_range, tvds, "s-", color="tab:red", label="TVD")
    plt.xlabel("Broj kubita (N)")
    plt.ylabel("Total Variation Distance (TVD)")
    plt.title(f"TVD vs. Broj kubita ({exp_name})")
    plt.xticks(qubit_range)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "tvd_vs_qubits.png"), bbox_inches="tight"
    )
    plt.close()


    json_path = os.path.join(exp_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4)

    mean_fidelity = statistics.mean(fidelities_noisy)
    mean_tvd = statistics.mean(tvds)
    min_fidelity = min(fidelities_noisy)
    max_fidelity = max(fidelities_noisy)
    fidelity_drop_per_qubit = (fidelities_noisy[0] - fidelities_noisy[-1]) / (list(qubit_range)[-1] - list(qubit_range)[0])

    txt_path = os.path.join(exp_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Opseg kubita: N = {list(qubit_range)[0]} do {list(qubit_range)[-1]}\n")
        f.write(f"Broj merenja (shots): {shots}\n\n")
        f.write("--- FIDELITY METRIKE ---\n")
        f.write(f"Prosečni Fidelity: {mean_fidelity:.4f}\n")
        f.write(f"Maksimalni Fidelity (N={list(qubit_range)[0]}): {max_fidelity:.4f}\n")
        f.write(f"Minimalni Fidelity (N={list(qubit_range)[-1]}): {min_fidelity:.4f}\n")
        f.write(f"Prosečan pad fideliteta po kubitu: {fidelity_drop_per_qubit:.4f}\n\n")
        f.write("--- TVD METRIKE ---\n")
        f.write(f"Prosečni TVD: {mean_tvd:.4f}\n")
        f.write(f"Minimalni TVD (N={list(qubit_range)[0]}): {min(tvds):.4f}\n")
        f.write(f"Maksimalni TVD (N={list(qubit_range)[-1]}): {max(tvds):.4f}\n")


def main():
    run_experiment(
        circuit_fn=get_ghz_circuit,
        noise_fn=pauli_noise_model,
        exp_name="exp_01_ghz_pauli",
        qubit_range=range(2, 21),
        shots=10000,
    )

if __name__ == "__main__":
    main()