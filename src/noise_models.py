from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import QuantumCircuit

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