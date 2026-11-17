import hashlib
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit.library import CXGate
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import PropertySet
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, state_fidelity
from qiskit.synthesis import TwoQubitBasisDecomposer, qs_decomposition
from qiskit.transpiler.passes import CollectMultiQBlocks

from src.predictors.esp_fidelity import predict_fidelity_esp

MAX_BLOCK_SIZE = 3
MAX_BLOCKS_PER_NODE = 12
DEFAULT_HARDWARE_BASIS = ("x", "sx", "rz", "id", "cx", "ecr", "cz")
DEFAULT_SEED = 1234
KAK_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())

_IDX_PREFIX = "__i"

_MOVE_ERRORS = Counter()


def _note_error(move, exc):
    _MOVE_ERRORS[(move, type(exc).__name__, str(exc).replace("\n", " ")[:120])] += 1


def _unitary_key(unitary):
    m = np.asarray(unitary, dtype=complex)
    return tuple((np.round(m.real, 8) + 1j * np.round(m.imag, 8)).flat)


_QSD_CACHE = {}
SYNTH_USED = Counter()

_QSD_STRATEGIES = (
    ("qsd", {}),
    ("qsd-no-a2", {"opt_a2": False}),
    ("qsd-no-a1-a2", {"opt_a1": False, "opt_a2": False}),
)


def _where(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return ""
    f = frames[-1]
    return f" [{Path(f.filename).name}:{f.lineno} in {f.name}]"


def _exactify(circ, target):
    v = Operator(circ).data
    tr = np.vdot(v, target)
    if abs(tr) < 1e-9:
        raise ValueError("synthesized circuit is unrelated to the target unitary")
    err = float(np.max(np.abs(target - (tr / abs(tr)) * v)))
    if err > 1e-8:
        raise ValueError(f"synthesized circuit differs from the target by {err:.1e}")
    out = circ.copy()
    out.global_phase += float(np.angle(tr))
    return out


def _isometry_fallback(m, n_qubits):
    try:
        from qiskit.circuit.library import IsometryGate as Iso
    except ImportError:
        from qiskit.circuit.library import Isometry as Iso
    qc = QuantumCircuit(n_qubits)
    qc.append(Iso(m, 0, 0), list(range(n_qubits)))
    return qc.decompose(reps=10)


def _cached_qs_decomposition(unitary):
    m = np.asarray(unitary, dtype=complex)
    key = _unitary_key(m)
    hit = _QSD_CACHE.get(key)
    if hit is not None:
        return hit

    n_qubits = int(round(np.log2(m.shape[0])))
    errors = []
    result = None

    attempts = [(label, (lambda kw=kw: qs_decomposition(m, **kw))) for label, kw in _QSD_STRATEGIES]
    attempts.append(("isometry", lambda: _isometry_fallback(m, n_qubits)))

    for label, run in attempts:
        try:
            result = _exactify(run(), m)
            SYNTH_USED[label] += 1
            break
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:100]}{_where(exc)}")

    if result is None:
        raise RuntimeError(" | ".join(errors))

    if len(_QSD_CACHE) >= 4096:
        _QSD_CACHE.clear()
    _QSD_CACHE[key] = result
    return result


def _qubit_index(circuit, qubit):
    return circuit.find_bit(qubit).index


def _tag_circuit(circuit):
    tagged = circuit.copy_empty_like()
    for i, instr in enumerate(circuit.data):
        op = instr.operation
        op = op.to_mutable() if hasattr(op, "to_mutable") else op.copy()
        op.label = f"{_IDX_PREFIX}{i}"
        tagged.append(op, instr.qubits, instr.clbits)
    return tagged


def _node_index(node):
    return int(node.op.label[len(_IDX_PREFIX):])


def _block_to_unitary(block_nodes):
    qubits = []
    for node in block_nodes:
        for q in node.qargs:
            if q not in qubits:
                qubits.append(q)

    sub = QuantumCircuit(len(qubits))
    qmap = {q: i for i, q in enumerate(qubits)}

    for node in block_nodes:
        sub.append(node.op, [qmap[q] for q in node.qargs], [])

    return Operator(sub).data, qubits


def _emit_index(circuit, block_indices):
    idx = sorted(block_indices)
    lo, hi = idx[0], idx[-1]
    block_set = set(idx)
    between = [i for i in range(lo + 1, hi) if i not in block_set]
    if not between:
        return lo, "contiguous"

    block_q = {i: set(circuit.data[i].qubits) for i in idx}
    can_lo = can_hi = True
    for g in between:
        gq = set(circuit.data[g].qubits)
        later = set().union(*(block_q[i] for i in idx if i > g))
        earlier = set().union(*(block_q[i] for i in idx if i < g))
        if gq & later:
            can_lo = False
        if gq & earlier:
            can_hi = False

    if can_lo:
        return lo, "first"
    if can_hi:
        return hi, "last"
    raise ValueError("block is not contiguous on its qubits")


def _replace_block(circuit, block_indices, qubits, new_block, emit_at):
    block_set = set(block_indices)
    qmap = {new_block.qubits[i]: q for i, q in enumerate(qubits)}

    new_circuit = circuit.copy_empty_like()
    new_circuit.global_phase += new_block.global_phase

    for i, instr in enumerate(circuit.data):
        if i in block_set:
            if i == emit_at:
                for new_instr in new_block.data:
                    new_circuit.append(
                        new_instr.operation,
                        [qmap[q] for q in new_instr.qubits],
                        new_instr.clbits
                    )
            continue
        new_circuit.append(instr.operation, instr.qubits, instr.clbits)

    return new_circuit


def _synthesize_unitary(unitary, n_qubits):
    m = np.asarray(unitary, dtype=complex)
    if n_qubits == 2:
        return _exactify(KAK_DECOMPOSER(Operator(m)), m)
    return _cached_qs_decomposition(m)


def _replace_window(circuit, start, end, new_block, sub_qubits):
    new_circuit = circuit.copy_empty_like()
    new_circuit.global_phase += new_block.global_phase
    qmap = {new_block.qubits[i]: sub_qubits[i] for i in range(len(sub_qubits))}

    for i, instr in enumerate(circuit.data):
        if i == start:
            for new_instr in new_block.data:
                new_circuit.append(
                    new_instr.operation,
                    [qmap[q] for q in new_instr.qubits],
                    new_instr.clbits
                )
        elif start < i <= end:
            continue
        else:
            new_circuit.append(instr.operation, instr.qubits, instr.clbits)

    return new_circuit


def _peephole_successors(circuit):
    successors = []

    for i in range(len(circuit.data) - 1):
        a = circuit.data[i]
        b = circuit.data[i + 1]

        if a.operation.num_qubits != b.operation.num_qubits:
            continue

        if a.qubits != b.qubits or a.clbits != b.clbits:
            continue

        if a.operation.name in {"x", "y", "z", "h", "cx", "cz"}:
            if b.operation.name == a.operation.name:
                new_circuit = circuit.copy_empty_like()

                for j, instr in enumerate(circuit.data):
                    if j == i:
                        continue
                    if j == i + 1:
                        continue
                    new_circuit.append(instr.operation, instr.qubits, instr.clbits)

                successors.append(new_circuit)

    return successors


def get_matrix_decompositions(circuit, max_block_size=MAX_BLOCK_SIZE, tags=None):
    successors = []

    dag = circuit_to_dag(_tag_circuit(circuit))
    collect = CollectMultiQBlocks(max_block_size=max_block_size)
    collect.property_set = PropertySet()
    collect.run(dag)

    blocks = [
        b for b in collect.property_set.get("block_list", [])
        if len(b) >= 2
    ]

    def sort_key(block):
        two_q = sum(len(n.qargs) == 2 for n in block)
        qubits = tuple(
            sorted(
                _qubit_index(circuit, q)
                for n in block
                for q in n.qargs
            )
        )
        return (-two_q, qubits)

    blocks.sort(key=sort_key)

    for block in blocks[:MAX_BLOCKS_PER_NODE]:
        try:
            indices = [_node_index(n) for n in block]
            emit_at, kind = _emit_index(circuit, indices)
            unitary, qubits = _block_to_unitary(block)
            new_block = _synthesize_unitary(unitary, len(qubits))
            successors.append(_replace_block(circuit, indices, qubits, new_block, emit_at))
            if tags is not None:
                suffix = {"contiguous": "", "first": "+jump-back", "last": "+jump-fwd"}[kind]
                tags.append(f"block-{len(qubits)}q{suffix}")
        except Exception as exc:
            _note_error("block", exc)
            continue

    two_q_indices = [
        i for i, inst in enumerate(circuit.data)
        if inst.operation.num_qubits == 2
    ]

    for idx in range(len(two_q_indices) - 1):
        i1 = two_q_indices[idx]
        i2 = two_q_indices[idx + 1]

        inst1 = circuit.data[i1]
        inst2 = circuit.data[i2]

        q1 = set(inst1.qubits)
        q2 = set(inst2.qubits)

        if not q1.intersection(q2) or i2 - i1 > 8:
            continue

        try:
            sub_qubits = sorted(q1 | q2, key=lambda q: _qubit_index(circuit, q))
            sub_qc = QuantumCircuit(len(sub_qubits))
            qmap = {q: i for i, q in enumerate(sub_qubits)}

            for instr in circuit.data[i1:i2 + 1]:
                sub_qc.append(
                    instr.operation,
                    [qmap[q] for q in instr.qubits],
                    instr.clbits
                )

            unitary = Operator(sub_qc).data
            new_block = _synthesize_unitary(unitary, len(sub_qubits))

            successors.append(
                _replace_window(
                    circuit,
                    i1,
                    i2,
                    new_block,
                    sub_qubits
                )
            )
            if tags is not None:
                tags.append(f"window-{len(sub_qubits)}q")
        except Exception as exc:
            _note_error("window", exc)
            continue

    peephole = _peephole_successors(circuit)
    successors.extend(peephole)
    if tags is not None:
        tags.extend(["peephole"] * len(peephole))

    return successors


def _num_key(x):
    try:
        return round(float(x), 10)
    except (TypeError, ValueError):
        return str(x)


def _param_key(p):
    if isinstance(p, np.ndarray):
        a = np.asarray(p, dtype=complex)
        return ("nd", a.shape, tuple(np.round(a.real, 10).ravel().tolist()),
                tuple(np.round(a.imag, 10).ravel().tolist()))
    return _num_key(p)


def _structural_key(circuit):
    qidx = {q: i for i, q in enumerate(circuit.qubits)}
    cidx = {c: i for i, c in enumerate(circuit.clbits)}
    parts = [_num_key(circuit.global_phase)]
    for ins in circuit.data:
        op = ins.operation
        has_matrix = any(isinstance(p, np.ndarray) for p in op.params)
        definition = None
        module = getattr(type(op), "__module__", None) or ""
        if not has_matrix and not module.startswith("qiskit.circuit.library.standard_gates"):
            definition = getattr(op, "definition", None)
        parts.append((
            op.name,
            tuple(qidx[q] for q in ins.qubits),
            tuple(cidx[c] for c in ins.clbits),
            tuple(_param_key(p) for p in op.params),
            _structural_key(definition) if isinstance(definition, QuantumCircuit) else None,
        ))
    return "structural:" + hashlib.sha1(repr(parts).encode()).hexdigest()


def _circuit_key(circuit):
    try:
        return qasm3.dumps(circuit)
    except Exception:
        return _structural_key(circuit)


def hardware_transpile(circuit, coupling_map=None, hardware_basis=DEFAULT_HARDWARE_BASIS,
                       seed=DEFAULT_SEED, optimization_level=0):
    return transpile(
        circuit,
        basis_gates=list(hardware_basis),
        coupling_map=coupling_map,
        optimization_level=optimization_level,
        seed_transpiler=seed
    )


class CandidateScorer:

    def __init__(self, gate_errors, coupling_map=None, hardware_basis=DEFAULT_HARDWARE_BASIS,
                 seed=DEFAULT_SEED):
        self.gate_errors = gate_errors
        self.coupling_map = coupling_map
        self.hardware_basis = hardware_basis
        self.seed = seed
        self.cache = {}
        self.transpile_seconds = 0.0
        self.transpile_calls = 0

    def __call__(self, circuit):
        key = _circuit_key(circuit)
        hit = self.cache.get(key)
        if hit is None:
            t = time.perf_counter()
            routed = hardware_transpile(circuit, self.coupling_map, self.hardware_basis, self.seed)
            hit = (predict_fidelity_esp(routed, self.gate_errors), routed)
            self.transpile_seconds += time.perf_counter() - t
            self.transpile_calls += 1
            self.cache[key] = hit
        return hit


def collect_circuit_variants_beam(
    start_circuit,
    max_evaluations,
    scorer,
    beam_width=10,
    seed=DEFAULT_SEED,
    patience=5,
    min_improvement=1e-9
):
    rng = np.random.default_rng(seed)

    start_esp, _ = scorer(start_circuit)
    visited = {_circuit_key(start_circuit)}
    candidates = [dict(circuit=start_circuit, esp=start_esp, depth=0, parent=None)]
    frontier = [(start_esp, 0)]
    expansions = 0
    generations = 0
    best_esp = start_esp
    stale = 0
    plateau = False

    while frontier and len(candidates) < max_evaluations:
        batch = frontier[:beam_width]
        frontier = frontier[beam_width:]

        next_generation = []

        for _, parent_idx in batch:
            parent = candidates[parent_idx]
            successors = get_matrix_decompositions(parent["circuit"])
            expansions += 1

            for i in rng.permutation(len(successors)):
                successor = successors[int(i)]
                key = _circuit_key(successor)

                if key in visited:
                    continue

                visited.add(key)

                esp, _ = scorer(successor)

                candidates.append(dict(
                    circuit=successor,
                    esp=esp,
                    depth=parent["depth"] + 1,
                    parent=parent_idx
                ))
                next_generation.append((esp, len(candidates) - 1))

                if len(candidates) >= max_evaluations:
                    break

            if len(candidates) >= max_evaluations:
                break

        generations += 1

        generation_best = max((e for e, _ in next_generation), default=best_esp)
        if generation_best > best_esp + min_improvement:
            best_esp = generation_best
            stale = 0
        else:
            stale += 1

        next_generation.sort(key=lambda x: x[0], reverse=True)
        frontier.extend(next_generation)
        frontier.sort(key=lambda x: x[0], reverse=True)

        frontier = frontier[:max_evaluations * 2]

        if patience is not None and stale >= patience:
            plateau = True
            break

    if plateau:
        stop_reason = "plateau"
    elif len(candidates) >= max_evaluations:
        stop_reason = "budget"
    else:
        stop_reason = "exhausted"

    return candidates, dict(
        expansions=expansions,
        generations=generations,
        frontier_left=len(frontier),
        stop_reason=stop_reason,
        best_esp_found=best_esp
    )


def verify_true_fidelity(routed_circuits, noise_model, seed=DEFAULT_SEED):
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        seed_simulator=seed
    )

    results = []

    for routed in routed_circuits:
        clean = routed.remove_final_measurements(inplace=False)
        ideal = Statevector.from_instruction(clean)

        sim_circuit = clean.copy()
        sim_circuit.save_density_matrix()
        result = simulator.run(sim_circuit, seed_simulator=seed).result()
        noisy = DensityMatrix(result.data(0)["density_matrix"])

        results.append({
            "fidelity": float(state_fidelity(ideal, noisy)),
            "p_ideal": np.asarray(ideal.probabilities(), dtype=float),
            "p_noisy": np.asarray(noisy.probabilities(), dtype=float)
        })

    return results


def optimize_circuit(
    circuit,
    noise_model,
    gate_errors,
    max_evaluations=1000,
    beam_width=10,
    top_k=5,
    patience=5,
    min_improvement=1e-9,
    coupling_map=None,
    hardware_basis=DEFAULT_HARDWARE_BASIS,
    seed=DEFAULT_SEED,
    beam_seed=None,
    sim_seed=None,
    **kwargs
):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    beam_seed = seed if beam_seed is None else beam_seed
    sim_seed = seed if sim_seed is None else sim_seed

    scorer = CandidateScorer(gate_errors, coupling_map, hardware_basis, seed)

    t = time.perf_counter()
    candidates, search_info = collect_circuit_variants_beam(
        circuit,
        max_evaluations,
        scorer,
        beam_width,
        beam_seed,
        patience,
        min_improvement
    )
    search_seconds = time.perf_counter() - t

    ranked = sorted(range(len(candidates)), key=lambda i: candidates[i]["esp"], reverse=True)
    esp_rank = [0] * len(candidates)
    for position, i in enumerate(ranked):
        esp_rank[i] = position

    chosen = [0] + [i for i in ranked[:top_k] if i != 0]

    t = time.perf_counter()
    routed_chosen = [scorer(candidates[i]["circuit"])[1] for i in chosen]
    sims = verify_true_fidelity(routed_chosen, noise_model, sim_seed)
    simulation_seconds = time.perf_counter() - t

    verified = []
    for i, routed, sim in zip(chosen, routed_chosen, sims):
        verified.append(dict(index=i, routed=routed, **sim))

    original = verified[0]
    best = max(verified, key=lambda v: v["fidelity"])
    esp_selected = next(v for v in verified if v["index"] == ranked[0])

    return {
        "best_circuit": candidates[best["index"]]["circuit"],
        "best_routed": best["routed"],
        "best_index": best["index"],
        "best_true_fidelity": best["fidelity"],
        "original_true_fidelity": original["fidelity"],
        "best_esp_score": candidates[best["index"]]["esp"],
        "improvement": best["fidelity"] - original["fidelity"],
        "esp_selected_index": esp_selected["index"],
        "esp_selected_true_fidelity": esp_selected["fidelity"],
        "seed": seed,
        "beam_seed": beam_seed,
        "sim_seed": sim_seed,
        "candidates": candidates,
        "esp_rank": esp_rank,
        "verified": verified,
        "search_stats": {
            "nodes_expanded": search_info["expansions"],
            "generations": search_info["generations"],
            "nodes_evaluated": len(candidates),
            "frontier_left": search_info["frontier_left"],
            "verified_count": len(verified),
            "transpile_calls": scorer.transpile_calls,
            "stop_reason": search_info["stop_reason"],
            "stopped_early": search_info["stop_reason"] != "budget",
            "max_evaluations": max_evaluations,
            "patience": patience,
            "best_candidate_index": best["index"]
        },
        "timing": {
            "search_seconds": search_seconds,
            "transpile_seconds": scorer.transpile_seconds,
            "simulation_seconds": simulation_seconds
        }
    }