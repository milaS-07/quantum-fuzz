from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple


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
    raise ValueError(f"Unknown architecture {architecture!r} "
                      "(use 'full', '1d', '2d', or '3d')")


def average_fidelity(
    L: int,
    T: int,
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
    T: int,
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


def extract_L_T(circuit) -> Tuple[int, int]:
    L = circuit.num_qubits

    def _is_two_qubit(instr) -> bool:
        op = getattr(instr, "operation", None)
        if op is None:
            op = instr[0]
        return getattr(op, "num_qubits", None) == 2

    try:
        T = circuit.depth(filter_function=_is_two_qubit)
    except TypeError:
        qubit_layer = {q: 0 for q in circuit.qubits}
        max_layer = 0
        for instr in circuit.data:
            op = getattr(instr, "operation", instr[0])
            qargs = getattr(instr, "qubits", None) or instr[1]
            if getattr(op, "num_qubits", None) != 2:
                continue
            layer = max(qubit_layer[q] for q in qargs) + 1
            for q in qargs:
                qubit_layer[q] = layer
            max_layer = max(max_layer, layer)
        T = max_layer

    return L, T


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
    L, T = extract_L_T(circuit)
    if architecture is None:
        architecture = guess_architecture(coupling_map)

    F_exact = average_fidelity(L, T, p=p, alpha=alpha,
                                architecture=architecture, exact_f4=exact_f4)
    F_asym = asymptotic_fidelity(L, T, p=p, alpha=alpha, architecture=architecture)

    return {
        "L": L,
        "T": T,
        "architecture": architecture,
        "alpha": alpha,
        "p": p,
        "fidelity_solvable_model": F_exact,
        "fidelity_asymptotic": F_asym,
    }





if __name__ == "__main__":
    L, T = 6, 20
    for p in (0.0, 0.001, 0.005):
        Fv = average_fidelity(L, T, p=p, alpha=0.05, architecture="1d")
        print(f"L={L} T={T} p={p:<6} alpha=0.05  ->  F = {Fv:.4f}")