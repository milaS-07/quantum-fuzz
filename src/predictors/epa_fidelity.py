import numpy as np
from qiskit.quantum_info import Operator, Kraus, average_gate_fidelity

VIRTUAL_GATES = frozenset({"rz", "z", "s", "sdg", "t", "tdg", "p", "u1", "id"})
IGNORED_OPS = frozenset({"barrier", "measure", "delay"})


def get_avg_gate_fidelities(noise_model):
    fids_1q, fids_2q = [], []
    for gate_name, quantum_error in noise_model._default_quantum_errors.items():
        if gate_name in VIRTUAL_GATES:
            continue
        n = quantum_error.num_qubits
        fid = average_gate_fidelity(Kraus(quantum_error), Operator(np.eye(2 ** n)))
        if n == 1:
            fids_1q.append(fid)
        elif n == 2:
            fids_2q.append(fid)
    f_1q = sum(fids_1q) / len(fids_1q) if fids_1q else 1.0
    f_2q = sum(fids_2q) / len(fids_2q) if fids_2q else 1.0
    return f_1q, f_2q


def predict_fidelity_epa(circuit, f_1q, f_2q):
    n_1q = n_2q = 0
    for instr in circuit.data:
        op = instr.operation
        if op.name in IGNORED_OPS or op.name in VIRTUAL_GATES:
            continue
        if op.num_qubits == 1:
            n_1q += 1
        elif op.num_qubits == 2:
            n_2q += 1
        else:
            n_2q += 2 * op.num_qubits
    return (f_1q ** n_1q) * (f_2q ** n_2q)