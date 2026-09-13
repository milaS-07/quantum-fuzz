from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Union


def f4_exact(alpha: float) -> float:
    a2 = alpha * alpha
    poly = (-(a2**5) + 12.5 * a2**4 - 64 * a2**3 + 138 * a2**2 - 144 * a2 + 36)
    return math.exp(-a2) * poly / 36.0


def f4_approx(alpha: float) -> float:
    return math.exp(-5.0 * alpha * alpha)


def Delta(L: int, alpha: float, exact: bool = True) -> float:
    f4 = f4_exact(alpha) if exact else f4_approx(alpha)
    return (2.0 ** L) * (3.0 * f4 + 1.0) ** (L / 2.0)


EULER_GAMMA = 0.5772156649015329


def delta_full(L: int, p: float) -> float:
    if L <= 1:
        return 4.0 ** L
    exponent = -0.75 * p * (L - math.log(L) - EULER_GAMMA)
    return (4.0 ** L) * math.exp(exponent)


def delta_1d(L: int, p: float) -> float:
    exponent = -0.1875 * p * L * (L - 1)
    return (4.0 ** L) * math.exp(exponent)


def delta_dD(L: int, p: float, d: int) -> float:
    if d == 1:
        return delta_1d(L, p)
    mu = d - 0.5
    exponent = -0.75 * mu * p * (L ** (1.0 + 1.0 / d))
    return (4.0 ** L) * math.exp(exponent)


Architecture = Literal["full", "1d", "2d", "3d"]


def delta_for_architecture(L: int, p: float, architecture: Architecture) -> float:
    if p <= 0:
        return 4.0 ** L
    if architecture == "full":
        return delta_full(L, p)
    if architecture == "1d":
        return delta_1d(L, p)
    if architecture == "2d":
        return delta_dD(L, p, 2)
    if architecture == "3d":
        return delta_dD(L, p, 3)
    raise ValueError(f"Unknown architecture {architecture!r}")


def average_fidelity(
    L: int,
    T: float, # Changed to float to support T_eff
    p: float = 0.0,
    alpha: float = 0.0,
    architecture: Architecture = "full",
    exact_f4: bool = True,
) -> float:
    if L < 1:
        raise ValueError("L must be >= 1")
    if T < 0:
        raise ValueError("T must be >= 0")

    d = Delta(L, alpha, exact=exact_f4)
    delta = delta_for_architecture(L, p, architecture)

    base = 4.0 ** L - 1.0
    bracket = (delta - 1.0) * (d - 1.0) / (base * base)

    floor = 1.0 / (2.0 ** L)
    return (1.0 - floor) * (bracket ** T) + floor


def asymptotic_fidelity(
    L: int,
    T: float, # Changed to float to support T_eff
    p: float = 0.0,
    alpha: float = 0.0,
    architecture: Architecture = "full",
) -> float:
    nu = 15.0 / 8.0
    perm_term = 0.0
    if p > 0:
        if architecture == "full":
            perm_term = 0.75 * p * T * L
        elif architecture == "1d":
            perm_term = (3.0 / 16.0) * p * T * (L ** 2)
        elif architecture == "2d":
            perm_term = (9.0 / 8.0) * p * T * (L ** 1.5)
        elif architecture == "3d":
            perm_term = 0.75 * 2.5 * p * T * (L ** (4.0 / 3.0))
        else:
            raise ValueError(f"Unknown architecture {architecture!r}")
    gate_term = nu * (alpha ** 2) * L * T
    return math.exp(-(gate_term + perm_term))


@dataclass
class CircuitRQCParams:
    L: int
    T: int
    architecture: Optional[Architecture] = None


def extract_circuit_metrics(circuit) -> Tuple[int, int, float]:
    """
    Extracts L (qubits), T_depth (original critical path depth for 2Q gates),
    and T_eff (density-based depth for sparse circuits).
    """
    L = circuit.num_qubits

    def _is_two_qubit(instr) -> bool:
        op = getattr(instr, "operation", None)
        if op is None:
            op = instr[0]
        return getattr(op, "num_qubits", None) == 2

    # 1. Calculate T_depth (Original method)
    try:
        T_depth = circuit.depth(filter_function=_is_two_qubit)
    except TypeError:
        qubit_layer = {q: 0 for q in circuit.qubits}
        max_layer = 0
        for instr in getattr(circuit, "data", circuit):
            op = getattr(instr, "operation", instr[0])
            qargs = getattr(instr, "qubits", None)
            if qargs is None:
                qargs = instr[1]
            if getattr(op, "num_qubits", None) != 2:
                continue
            layer = max(qubit_layer.get(q, 0) for q in qargs) + 1
            for q in qargs:
                qubit_layer[q] = layer
            max_layer = max(max_layer, layer)
        T_depth = max_layer

    # 2. Calculate T_eff (New sparse method)
    try:
        num_2q_gates = sum(1 for instr in circuit.data if _is_two_qubit(instr))
    except AttributeError:
        num_2q_gates = sum(1 for instr in circuit if _is_two_qubit(instr))
        
    gates_per_full_layer = L / 2.0
    T_eff = num_2q_gates / gates_per_full_layer if gates_per_full_layer > 0 else 0.0

    return L, int(T_depth), float(T_eff)


def guess_architecture(coupling_map) -> Architecture:
    if coupling_map is None:
        return "full"

    edges = list(coupling_map.get_edges()) if hasattr(coupling_map, "get_edges") else list(coupling_map)
    n = coupling_map.size() if hasattr(coupling_map, "size") else (max(max(e) for e in edges) + 1)

    undirected = {frozenset(e) for e in edges}
    num_edges = len(undirected)
    max_edges_full = n * (n - 1) // 2

    if num_edges >= 0.9 * max_edges_full:
        return "full"

    deg = {}
    for e in undirected:
        for q in e:
            deg[q] = deg.get(q, 0) + 1
    avg_deg = sum(deg.values()) / max(len(deg), 1)

    if avg_deg <= 2.5:
        return "1d"
    elif avg_deg <= 4.5:
        return "2d"
    else:
        return "3d"


def fidelity_from_circuit(
    circuit,
    alpha: float,
    p: float = 0.0,
    architecture: Optional[Architecture] = None,
    coupling_map=None,
    exact_f4: bool = True,
) -> dict:
    # Now extracts both metrics
    L, T_depth, T_eff = extract_circuit_metrics(circuit)
    
    if architecture is None:
        architecture = guess_architecture(coupling_map)

    # Theoretical fidelity based on regular Depth (Original)
    F_exact_depth = average_fidelity(L, T_depth, p=p, alpha=alpha,
                                     architecture=architecture, exact_f4=exact_f4)
    F_asym_depth = asymptotic_fidelity(L, T_depth, p=p, alpha=alpha, architecture=architecture)

    # Theoretical fidelity based on Effective Depth (New)
    F_exact_eff = average_fidelity(L, T_eff, p=p, alpha=alpha,
                                   architecture=architecture, exact_f4=exact_f4)
    F_asym_eff = asymptotic_fidelity(L, T_eff, p=p, alpha=alpha, architecture=architecture)

    return {
        "L": L,
        "T_depth": T_depth,
        "T_eff": T_eff,
        "architecture": architecture,
        "alpha": alpha,
        "p": p,
        "fidelity_solvable_model_depth": F_exact_depth,
        "fidelity_asymptotic_depth": F_asym_depth,
        "fidelity_solvable_model_eff": F_exact_eff,
        "fidelity_asymptotic_eff": F_asym_eff,
    }


HARDWARE_PRESETS = {
    "ibm_sherbrooke": dict(
        t1=289.55e-6, t2=186.01e-6,
        time_1q=42.67e-9, time_2q=539.90e-9,
        depol_1q=0.00042, depol_2q=0.07200,
    ),
}


def F_avg_from_alpha(alpha: float) -> float:
    # Corrected formula for d=4 extracting exact entanglement fidelity relation
    return (3.0 * f4_exact(alpha) + 2.0) / 5.0


def alpha_from_F_avg(F_avg_target: float, tol: float = 1e-12) -> float:
    if not (0.0 < F_avg_target <= 1.0):
        raise ValueError("F_avg_target must be in (0, 1]")
    lo, hi = 0.0, 3.0
    f_lo = F_avg_from_alpha(lo) - F_avg_target
    f_hi = F_avg_from_alpha(hi) - F_avg_target
    if f_lo * f_hi > 0:
        raise ValueError(f"F_avg_target={F_avg_target} out of range for this model")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = F_avg_from_alpha(mid) - F_avg_target
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def alpha_from_2q_channel(t1: float, t2: float, time_2q: float, depol_2q: float) -> float:
    from qiskit.quantum_info import average_gate_fidelity, Kraus
    from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error

    if t2 > 2 * t1:
        t2 = 2 * t1

    thermal_2q = thermal_relaxation_error(t1, t2, time_2q).tensor(
        thermal_relaxation_error(t1, t2, time_2q)
    )
    combined_2q = thermal_2q.compose(depolarizing_error(depol_2q, 2))

    chan = Kraus(combined_2q.to_quantumchannel())
    F_avg = average_gate_fidelity(chan)

    return alpha_from_F_avg(F_avg)


def alpha_for_hardware(name: str) -> float:
    if name not in HARDWARE_PRESETS:
        raise KeyError(f"No preset for {name!r}. Add it to HARDWARE_PRESETS, "
                        f"or call alpha_from_2q_channel(...) directly.")
    p = HARDWARE_PRESETS[name]
    return alpha_from_2q_channel(t1=p["t1"], t2=p["t2"], time_2q=p["time_2q"], depol_2q=p["depol_2q"])


def fidelity_with_real_topology(circuit, coupling_map, t1, t2, time_2q, depol_2q, p=0.0):
    alpha = alpha_from_2q_channel(t1, t2, time_2q, depol_2q)
    return fidelity_from_circuit(circuit, alpha=alpha, p=p, coupling_map=coupling_map)


if __name__ == "__main__":
    L, T = 6, 20
    # A quick dry-run test
    print(f"L={L} T={T} alpha=0.05 p=0.0 -> Original Formula = {average_fidelity(L, T, alpha=0.05, architecture='1d'):.4f}")