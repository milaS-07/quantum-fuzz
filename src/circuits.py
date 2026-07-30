from qiskit import QuantumCircuit

def get_empty_circuit(num_qubits: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    qc.measure_all()
    return qc

def get_test_circuit(num_qubits: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.rx(0.5, 1)
    qc.cx(0, 1)
    qc.rz(0.2, 2)
    qc.measure_all()
    return qc

def get_ghz_circuit(num_qubits: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    
    qc.h(0)
    
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        
    qc.measure_all()
    
    return qc