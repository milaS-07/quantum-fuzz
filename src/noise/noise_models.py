from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import QuantumCircuit
from qiskit_aer.noise import (
    NoiseModel, 
    thermal_relaxation_error, 
    depolarizing_error
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

    
    one_qubit_gates = list({inst.operation.name for inst in qc.data 
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

def depolarizing_thermal_noise_model(
    t1: float,
    t2: float,
    time_1q: float,
    time_2q: float,
    depol_1q: float,
    depol_2q: float
) -> NoiseModel:
    if t2 > 2 * t1:
        t2 = 2 * t1

    noise_model = NoiseModel()

    thermal_1q = thermal_relaxation_error(t1, t2, time_1q)
    depol_1q_err = depolarizing_error(depol_1q, 1)
    combined_1q = thermal_1q.compose(depol_1q_err)

    thermal_2q_q0 = thermal_relaxation_error(t1, t2, time_2q)
    thermal_2q_q1 = thermal_relaxation_error(t1, t2, time_2q)
    thermal_2q = thermal_2q_q0.tensor(thermal_2q_q1)
    
    depol_2q_err = depolarizing_error(depol_2q, 2)
    combined_2q = thermal_2q.compose(depol_2q_err)

    noise_model.add_all_qubit_quantum_error(combined_1q, ["sx", "x", "id"])
    noise_model.add_all_qubit_quantum_error(combined_2q, ["cx", "ecr", "cz"])

    return noise_model