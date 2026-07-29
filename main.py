import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from src.circuits import get_test_circuit
from src.noise_models import pauli_noise_model


def main():
    qc = get_test_circuit(num_qubits=10)

    noise_model = pauli_noise_model(qc)

    shots = 10000

    #bez šuma
    sim_ideal = AerSimulator()
    qc_ideal = transpile(qc, sim_ideal)
    result_ideal = sim_ideal.run(qc_ideal, shots=shots).result()
    counts_ideal = result_ideal.get_counts()

    #sa šumom
    sim_noisy = AerSimulator(noise_model=noise_model)
    qc_noisy = transpile(qc, sim_noisy)
    result_noisy = sim_noisy.run(qc_noisy, shots=shots).result()
    counts_noisy = result_noisy.get_counts()

    sorted_ideal = dict(sorted(counts_ideal.items(), key=lambda item: item[1], reverse=True))
    sorted_noisy = dict(sorted(counts_noisy.items(), key=lambda item: item[1], reverse=True))

    print("rezultati bez šuma:", sorted_ideal)
    print("rezultati sa šumom:", sorted_noisy)

    plot_histogram([counts_ideal, counts_noisy], legend=['bez šuma', 'sa šumom'], sort='value_desc', number_to_keep=10)
    plt.show()


if __name__ == "__main__":
    main()