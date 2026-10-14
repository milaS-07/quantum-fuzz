import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


def extract_chip_parameters(backend) -> dict:
    props = backend.properties()
    n = backend.num_qubits

    t1_list = [props.t1(i) for i in range(n) if props.t1(i)]
    t2_list = [props.t2(i) for i in range(n) if props.t2(i)]
    t1 = float(np.median(t1_list))
    t2 = min(float(np.median(t2_list)), 2 * t1)

    ONE_Q = {"sx", "x"}
    TWO_Q = {"ecr", "cx", "cz"}

    times_1q, errors_1q = [], []
    times_2q, errors_2q = [], []

    for g in props.gates:
        try:
            duration = props.gate_length(g.gate, g.qubits)
            error = props.gate_error(g.gate, g.qubits)
        except Exception:
            continue

        if g.gate in ONE_Q:
            times_1q.append(duration)
            errors_1q.append(error)
        elif g.gate in TWO_Q:
            times_2q.append(duration)
            errors_2q.append(error)

    return {
        "t1": t1,
        "t2": t2,
        "time_1q": float(np.median(times_1q)),
        "time_2q": float(np.median(times_2q)),
        "err_1q": float(np.median(errors_1q)),
        "err_2q": float(np.median(errors_2q)),
        "mean_t1": float(np.mean(t1_list)),
        "mean_t2": float(np.mean(t2_list)),
        "mean_err_1q": float(np.mean(errors_1q)),
        "mean_err_2q": float(np.mean(errors_2q)),
        "n_1q": len(errors_1q),
        "n_2q": len(errors_2q),
    }

backend = FakeSherbrooke()
izlaz = extract_chip_parameters(backend)

print("Izvučeni parametri za model šuma (medijane):")
print(f"Datum kalibracije: {backend.properties().last_update_date}")
print(f"T1: {izlaz['t1'] * 1e6:.4f} us   (prosek {izlaz['mean_t1'] * 1e6:.4f})")
print(f"T2: {izlaz['t2'] * 1e6:.4f} us   (prosek {izlaz['mean_t2'] * 1e6:.4f})")
print(f"Trajanje 1Q kapije: {izlaz['time_1q'] * 1e9:.4f} ns")
print(f"Trajanje 2Q kapije: {izlaz['time_2q'] * 1e9:.4f} ns")
print(f"Greška 1Q kapije: {izlaz['err_1q'] * 100:.5f}%   (prosek {izlaz['mean_err_1q'] * 100:.5f}%)")
print(f"Greška 2Q kapije: {izlaz['err_2q'] * 100:.5f}%   (prosek {izlaz['mean_err_2q'] * 100:.5f}%)")

print("\nVrednosti za model (pune preciznosti):")
for key in ["t1", "t2", "time_1q", "time_2q", "err_1q", "err_2q"]:
    print(f"{key}={izlaz[key]!r},")