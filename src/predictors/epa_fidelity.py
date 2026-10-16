from qiskit.quantum_info import average_gate_fidelity

from src.noise.noise_models import GATES_1Q, GATES_2Q, build_gate_errors
from src.noise.noise_parameters import SHERBROOKE_PARAMS


def gate_errors_from_params(params=SHERBROOKE_PARAMS):
    error_1q, error_2q, _, _ = build_gate_errors(**params)
    eps_1q = 1.0 - average_gate_fidelity(error_1q.to_quantumchannel())
    eps_2q = 1.0 - average_gate_fidelity(error_2q.to_quantumchannel())
    gate_errors = {name: eps_1q for name in GATES_1Q}
    gate_errors.update({name: eps_2q for name in GATES_2Q})
    return gate_errors


def predict_fidelity_esp(circuit, gate_errors):
    value = 1.0
    for name, count in circuit.count_ops().items():
        value *= (1.0 - gate_errors.get(name, 0.0)) ** count
    return value