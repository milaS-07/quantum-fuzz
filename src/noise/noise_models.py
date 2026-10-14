import math
import warnings

from qiskit import QuantumCircuit
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer.noise import (
    NoiseModel,
    pauli_error,
    thermal_relaxation_error,
    depolarizing_error,
)


def pauli_noise_model(qc: QuantumCircuit, p_x: float = 0.01, p_y: float = 0.01, p_z: float = 0.01) -> NoiseModel:
    p_identity = 1 - (p_x + p_y + p_z)

    pauli_channel_one_qubit = pauli_error([
        ('I', p_identity),
        ('X', p_x),
        ('Y', p_y),
        ('Z', p_z)
    ])

    pauli_channel_two_qubit = pauli_channel_one_qubit.tensor(pauli_channel_one_qubit)

    one_qubit_gates = list({
        inst.operation.name for inst in qc.data
        if inst.operation.num_qubits == 1 and inst.operation.name != 'measure'
    })

    two_qubit_gates = list({
        inst.operation.name for inst in qc.data
        if inst.operation.num_qubits == 2
    })

    noise_model = NoiseModel()

    if one_qubit_gates:
        noise_model.add_all_qubit_quantum_error(pauli_channel_one_qubit, one_qubit_gates)

    if two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(pauli_channel_two_qubit, two_qubit_gates)

    return noise_model


def _thermal_process_fidelity(t1: float, t2: float, duration: float, n_qubits: int) -> float:
    single = (1 + math.exp(-duration / t1) + 2 * math.exp(-duration / t2)) / 4
    return single ** n_qubits


def _depolarizing_param(avg_gate_error: float, t1: float, t2: float, duration: float, n_qubits: int) -> float:
    d = 2 ** n_qubits
    f_target = 1 - (d + 1) / d * avg_gate_error
    f_thermal = _thermal_process_fidelity(t1, t2, duration, n_qubits)
    lam = (f_thermal - f_target) / (f_thermal - 1 / d ** 2)
    if lam < 0:
        warnings.warn(
            f"{n_qubits}Q: thermal relaxation alone exceeds the calibration error "
            f"({avg_gate_error:.3e}); depolarizing parameter set to 0."
        )
        return 0.0
    return lam


def build_gate_errors(t1, t2, time_1q, time_2q, err_1q, err_2q):
    t2 = min(t2, 2 * t1)

    thermal_1q = thermal_relaxation_error(t1, t2, time_1q)
    lam_1q = _depolarizing_param(err_1q, t1, t2, time_1q, 1)
    error_1q = thermal_1q.compose(depolarizing_error(lam_1q, 1))

    thermal_qubit = thermal_relaxation_error(t1, t2, time_2q)
    thermal_2q = thermal_qubit.tensor(thermal_qubit)
    lam_2q = _depolarizing_param(err_2q, t1, t2, time_2q, 2)
    error_2q = thermal_2q.compose(depolarizing_error(lam_2q, 2))

    return error_1q, error_2q, lam_1q, lam_2q


def depolarizing_thermal_noise_model(
    t1: float,
    t2: float,
    time_1q: float,
    time_2q: float,
    err_1q: float,
    err_2q: float,
) -> NoiseModel:
    error_1q, error_2q, _, _ = build_gate_errors(t1, t2, time_1q, time_2q, err_1q, err_2q)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1q, ["sx", "x", "id"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "ecr", "cz"])

    return noise_model


def check_gate_errors(t1, t2, time_1q, time_2q, err_1q, err_2q):
    error_1q, error_2q, lam_1q, lam_2q = build_gate_errors(t1, t2, time_1q, time_2q, err_1q, err_2q)
    r_1q = 1 - average_gate_fidelity(error_1q.to_quantumchannel())
    r_2q = 1 - average_gate_fidelity(error_2q.to_quantumchannel())
    print(f"1Q: model {r_1q * 100:.4f}%  calibration {err_1q * 100:.4f}%  (lambda = {lam_1q * 100:.4f}%)")
    print(f"2Q: model {r_2q * 100:.4f}%  calibration {err_2q * 100:.4f}%  (lambda = {lam_2q * 100:.4f}%)")