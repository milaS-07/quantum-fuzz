from qiskit import QuantumCircuit
import numpy as np
import mqt.bench as mqt
from typing import Dict, List, Any

from mqt.bench import BenchmarkLevel


def get_empty_circuit(num_qubits: int = 10) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    qc.measure_all()
    return qc

def get_rnd_circuit_1(num_qubits: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(3)
    qc.s(0)
    qc.t(1)
    qc.cx(0, 2)
    qc.cz(1, 3)
    qc.measure_all()
    return qc

def get_rnd_circuit_2(num_qubits: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        if i > 0:
            qc.cx(i - 1, i)
        qc.x(i)
    qc.measure_all()
    return qc

def get_rnd_circuit_3(num_qubits: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.rx(1.57, 0)
    qc.ry(0.78, 1)
    qc.rz(3.14, 2)
    
    qc.ccx(0, 1, 2)
    
    qc.cx(3, 4)
    
    qc.measure_all()
    return qc

def get_rnd_circuit_4(num_qubits: int = 8) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    
    for i in range(num_qubits):
        qc.h(i)
        
    for i in range(num_qubits):
        qc.cx(i, (i + 1) % num_qubits)
        
    qc.swap(0, 4)
    qc.swap(2, 6)
    
    qc.measure_all()
    return qc

FIXED_CIRCUIT_SIZES = {
    "shor": 18,
}


def get_mqt_circuit(benchmark_name: str, num_qubits: int = None) -> QuantumCircuit:
    meta = BENCHMARK_REGISTRY.get(benchmark_name, {})
    is_scalable = meta.get("scalable", True)

    name_map = {
        "Grover's": "grover",
        "Quantum Walk": "qwalk",
        "Efficient SU2 ansatz with Random Parameters": "vqe_su2",
        "Real Amplitudes ansatz with Random Parameters": "vqe_real_amp",
        "Two Local ansatz with random parameters": "vqe_two_local",
        "Entangled QFT": "qftentangled",
        "Quantum Fourier Transformation (QFT)": "qft",
        "Quantum Phase Estimation (QPE) exact": "qpeexact",
        "Quantum Phase Estimation (QPE) inexact": "qpeinexact",
        "Amplitude Estimation (AE)": "ae",
        "Quantum Approximation Optimization Algorithm (QAOA)": "qaoa",
        "Quantum Neural Network (QNN)": "qnn",
        "GHZ State": "ghz",
        "Graph State": "graphstate",
        "W-State": "wstate",
        "Deutsch-Jozsa": "dj",
        "Random Circuit": "randomcircuit",
        "Shor's": "shor",
    }

    mqt_id = name_map.get(benchmark_name, benchmark_name.lower().replace(" ", ""))

    try:
        if is_scalable:
            qc = mqt.get_benchmark(mqt_id, BenchmarkLevel.ALG, circuit_size=num_qubits)
        else:
            fixed_size = FIXED_CIRCUIT_SIZES.get(mqt_id, num_qubits)
            qc = mqt.get_benchmark(mqt_id, BenchmarkLevel.ALG, circuit_size=fixed_size)

        if not any(gate.operation.name == "measure" for gate in qc.data):
            qc.measure_all()
        return qc

    except Exception as e:
        raise ValueError(
            f"Greška pri učitavanju kola '{benchmark_name}' (mqt_id='{mqt_id}', n={num_qubits}): {str(e)}"
        ) from e


PQC_BENCHMARKS = [
    "Efficient SU2 ansatz with Random Parameters",
    "Real Amplitudes ansatz with Random Parameters",
    "Two Local ansatz with random parameters",
    "Quantum Approximation Optimization Algorithm (QAOA)",
    "Quantum Neural Network (QNN)",
]

CLIFFORD_BENCHMARKS = [
    "GHZ State",
    "Graph State",
]

NON_PARAM_NON_CLIFFORD_BENCHMARKS = [
    "W-State",
    "Deutsch-Jozsa",
    "Grover's",
    "Quantum Walk",
    "Shor's",
    "Quantum Fourier Transformation (QFT)",
    "Entangled QFT",
    "Quantum Phase Estimation (QPE) exact",
    "Quantum Phase Estimation (QPE) inexact",
    "Amplitude Estimation (AE)",
]

CONTROL_BENCHMARKS = [
    "Random Circuit"
]

BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register(benchmarks: List[str], category: str, is_scalable: bool):
    for name in benchmarks:
        BENCHMARK_REGISTRY[name] = {
            "category": category,
            "scalable": is_scalable
        }


_register([
    "Amplitude Estimation (AE)", "Deutsch-Jozsa", "Graph State", "GHZ State",
    "Grover's", "Quantum Approximation Optimization Algorithm (QAOA)",
    "Quantum Fourier Transformation (QFT)", "Entangled QFT", "Quantum Neural Network (QNN)",
    "Quantum Phase Estimation (QPE) exact", "Quantum Phase Estimation (QPE) inexact",
    "Quantum Walk", "Random Circuit",
    "Efficient SU2 ansatz with Random Parameters",
    "Real Amplitudes ansatz with Random Parameters", "Two Local ansatz with random parameters",
    "W-State"
], category="auto", is_scalable=True)

_register([
    "Shor's"
], category="auto", is_scalable=False)


for name in PQC_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "PQC"
for name in CLIFFORD_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Clifford"
for name in NON_PARAM_NON_CLIFFORD_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Non-Parametric Non-Clifford"
for name in CONTROL_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Control"

FIXED_CIRCUIT_SIZES = {
    "shor": 18,
}

PQC_BENCHMARKS = [
    "Efficient SU2 ansatz with Random Parameters",
    "Real Amplitudes ansatz with Random Parameters",
    "Two Local ansatz with random parameters",
    "Quantum Approximation Optimization Algorithm (QAOA)",
    "Quantum Neural Network (QNN)",
]

CLIFFORD_BENCHMARKS = [
    "GHZ State",
    "Graph State",
]

NON_PARAM_NON_CLIFFORD_BENCHMARKS = [
    "W-State",
    "Deutsch-Jozsa",
    "Grover's",
    "Quantum Walk",
    "Shor's",
    "Quantum Fourier Transformation (QFT)",
    "Entangled QFT",
    "Quantum Phase Estimation (QPE) exact",
    "Quantum Phase Estimation (QPE) inexact",
    "Amplitude Estimation (AE)",
]

CONTROL_BENCHMARKS = [
    "Random Circuit"
]

BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register(benchmarks: List[str], category: str, is_scalable: bool):
    for name in benchmarks:
        BENCHMARK_REGISTRY[name] = {
            "category": category,
            "scalable": is_scalable
        }


_register([
    "Amplitude Estimation (AE)", "Deutsch-Jozsa", "Graph State", "GHZ State",
    "Grover's", "Quantum Approximation Optimization Algorithm (QAOA)",
    "Quantum Fourier Transformation (QFT)", "Entangled QFT", "Quantum Neural Network (QNN)",
    "Quantum Phase Estimation (QPE) exact", "Quantum Phase Estimation (QPE) inexact",
    "Quantum Walk", "Random Circuit",
    "Efficient SU2 ansatz with Random Parameters",
    "Real Amplitudes ansatz with Random Parameters", "Two Local ansatz with random parameters",
    "W-State"
], category="auto", is_scalable=True)

_register([
    "Shor's"
], category="auto", is_scalable=False)


for name in PQC_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "PQC"
for name in CLIFFORD_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Clifford"
for name in NON_PARAM_NON_CLIFFORD_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Non-Parametric Non-Clifford"
for name in CONTROL_BENCHMARKS:
    BENCHMARK_REGISTRY[name]["category"] = "Control"