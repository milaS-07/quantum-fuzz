import numpy as np
from qiskit.quantum_info import Operator, Kraus, average_gate_fidelity

def get_avg_gate_fidelities(noise_model) -> tuple[float, float]:
    fids_1q, fids_2q = [], []
    for _gate_name, quantum_error in noise_model._default_quantum_errors.items():
        n = quantum_error.num_qubits
        kraus = Kraus(quantum_error)
        fid = average_gate_fidelity(kraus, Operator(np.eye(2 ** n)))
        if n == 1:
            fids_1q.append(fid)
        elif n == 2:
            fids_2q.append(fid)
    
    f_1q = (sum(fids_1q) / len(fids_1q)) if fids_1q else 1.0
    f_2q = (sum(fids_2q) / len(fids_2q)) if fids_2q else 1.0
    return f_1q, f_2q

def predict_fidelity_epa(circuit, f_1q: float, f_2q: float) -> float:
    n_1q = 0
    n_2q = 0
    for instr in getattr(circuit, "data", circuit):
        op = getattr(instr, "operation", instr[0])
        
        if op.name in ["barrier", "measure", "rz", "z"]:
            continue
            
        num_qubits = getattr(op, "num_qubits", None)
        if num_qubits == 1:
            n_1q += 1
        elif num_qubits == 2:
            n_2q += 1
            
    return (f_1q ** n_1q) * (f_2q ** n_2q)