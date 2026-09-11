import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


def extract_chip_averages(backend) -> dict:
    props = backend.properties()
    num_qubits = backend.num_qubits

    t1_list = [props.t1(i) for i in range(num_qubits) if props.t1(i) is not None]
    t2_list = [props.t2(i) for i in range(num_qubits) if props.t2(i) is not None]

    avg_t1 = float(np.mean(t1_list))
    avg_t2 = float(np.mean(t2_list))

    if avg_t2 > 2 * avg_t1:
        avg_t2 = 2 * avg_t1

    times_1q, times_2q = [], []
    errors_1q, errors_2q = [], [] 

    for g in props.gates:
        try:
            duration = props.gate_length(g.gate, g.qubits)
            error = props.gate_error(g.gate, g.qubits)

            if len(g.qubits) == 1:
                times_1q.append(duration)
                errors_1q.append(error)                
            elif len(g.qubits) == 2:
                times_2q.append(duration)
                errors_2q.append(error)                 
        except Exception:
            continue

    avg_time_1q = float(np.mean(times_1q)) if times_1q else -1
    avg_time_2q = float(np.mean(times_2q)) if times_2q else -1

    avg_depol_1q = float(np.mean(errors_1q)) if errors_1q else -1
    avg_depol_2q = float(np.mean(errors_2q)) if errors_2q else -1

    return {
        "t1": avg_t1,
        "t2": avg_t2,
        "time_1q": avg_time_1q,
        "time_2q": avg_time_2q,
        "depol_1q": avg_depol_1q,
        "depol_2q": avg_depol_2q,
    }


backend_device = FakeSherbrooke()

izlaz = extract_chip_averages(backend_device)

print("Izvučeni parametri za model šuma:")
print(f"T1 prosek: {izlaz['t1'] * 1e6:.2f} us")
print(f"T2 prosek: {izlaz['t2'] * 1e6:.2f} us")
print(f"Trajanje 1Q kapije: {izlaz['time_1q'] * 1e9:.2f} ns")
print(f"Trajanje 2Q kapije: {izlaz['time_2q'] * 1e9:.2f} ns")
print(f"Greška 1Q kapije: {izlaz['depol_1q'] * 100:.3f}%")
print(f"Greška 2Q kapije: {izlaz['depol_2q'] * 100:.3f}%")