from functools import lru_cache
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import PropertySet
from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.quantum_info import Operator, DensityMatrix, Statevector, state_fidelity
from qiskit.synthesis import qs_decomposition, TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate

MAX_BLOCK_SIZE = 3
MAX_BLOCKS_PER_NODE = 12
DEFAULT_HARDWARE_BASIS = ("x", "sx", "rz", "id", "cx", "ecr", "cz")
KAK_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())


@lru_cache(maxsize=4096)
def _cached_qs_decomposition(key):
    return qs_decomposition(np.array(key, dtype=complex))


def _unitary_key(unitary):
    m = np.asarray(unitary, dtype=complex)
    return tuple((np.round(m.real, 8) + 1j * np.round(m.imag, 8)).flat)


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


def _replace_block(circuit, block_nodes, qubits, new_block):
    block_ids = {id(n.op) for n in block_nodes}
    qmap = {new_block.qubits[i]: q for i, q in enumerate(qubits)}

    new_circuit = circuit.copy_empty_like()
    emitted = False

    for instr in circuit.data:
        if id(instr.operation) in block_ids:
            if not emitted:
                for new_instr in new_block.data:
                    new_circuit.append(
                        new_instr.operation,
                        [qmap[q] for q in new_instr.qubits],
                        new_instr.clbits
                    )
                emitted = True
        else:
            new_circuit.append(instr.operation, instr.qubits, instr.clbits)

    return new_circuit


def compute_epa_score(circuit, f_1q, f_2q):
    n_1q = n_2q = 0

    for instr in circuit.data:
        op = instr.operation

        if op.name in ("barrier", "measure", "delay"):
            continue

        if op.num_qubits == 1:
            n_1q += 1
        elif op.num_qubits == 2:
            n_2q += 1
        else:
            n_2q += op.num_qubits * 2

    return (f_1q ** n_1q) * (f_2q ** n_2q)


def _synthesize_unitary(unitary, n_qubits):
    if n_qubits == 2:
        return KAK_DECOMPOSER(Operator(unitary))
    return _cached_qs_decomposition(_unitary_key(unitary))


def _replace_window(circuit, start, end, new_block, sub_qubits):
    new_circuit = circuit.copy_empty_like()
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


def get_matrix_decompositions(circuit, max_block_size=MAX_BLOCK_SIZE):
    successors = []

    dag = circuit_to_dag(circuit)
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
                circuit.find_bit(q).index
                for n in block
                for q in n.qargs
            )
        )
        return (-two_q, qubits)

    blocks.sort(key=sort_key)

    for block in blocks[:MAX_BLOCKS_PER_NODE]:
        try:
            unitary, qubits = _block_to_unitary(block)
            new_block = _synthesize_unitary(unitary, len(qubits))
            successors.append(_replace_block(circuit, block, qubits, new_block))
        except Exception:
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
            sub_qubits = list(q1 | q2)
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
        except Exception:
            continue

    successors.extend(_peephole_successors(circuit))

    return successors


def _circuit_key(circuit):
    return qasm3.dumps(circuit)


def collect_circuit_variants_beam(
    start_circuit,
    max_evaluations,
    f_1q,
    f_2q,
    beam_width=10
):
    visited = set()
    all_candidates = []

    start_key = _circuit_key(start_circuit)
    start_epa = compute_epa_score(start_circuit, f_1q, f_2q)

    visited.add(start_key)
    all_candidates.append((start_circuit, start_epa))
    frontier = [(start_epa, start_circuit)]

    while frontier and len(all_candidates) < max_evaluations:
        current_batch = frontier[:beam_width]
        frontier = frontier[beam_width:]

        next_generation = []

        for _, current_circuit in current_batch:
            for successor in get_matrix_decompositions(current_circuit):
                key = _circuit_key(successor)

                if key in visited:
                    continue

                visited.add(key)

                epa = compute_epa_score(
                    successor,
                    f_1q,
                    f_2q
                )

                all_candidates.append((successor, epa))
                next_generation.append((epa, successor))

                if len(all_candidates) >= max_evaluations:
                    break

            if len(all_candidates) >= max_evaluations:
                break

        next_generation.sort(key=lambda x: x[0], reverse=True)
        frontier.extend(next_generation)
        frontier.sort(key=lambda x: x[0], reverse=True)

        frontier = frontier[:max_evaluations * 2]

    return all_candidates


def verify_true_fidelity(
    circuits,
    noise_model,
    coupling_map,
    hardware_basis=DEFAULT_HARDWARE_BASIS
):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model
    )

    results = []

    for circuit in circuits:
        routed = transpile(
            circuit,
            basis_gates=list(hardware_basis),
            coupling_map=coupling_map,
            optimization_level=0
        )

        clean = routed.remove_final_measurements(inplace=False)
        ideal = Statevector.from_instruction(clean)

        clean.save_density_matrix()
        result = simulator.run(clean).result()
        noisy = DensityMatrix(
            result.data(0)["density_matrix"]
        )

        results.append({
            "circuit": circuit,
            "routed": routed,
            "fidelity": float(
                state_fidelity(ideal, noisy)
            )
        })

    return results


def optimize_circuit(
    circuit,
    noise_model,
    f_1q,
    f_2q,
    max_evaluations=250,
    beam_width=10,
    coupling_map=None,
    hardware_basis=DEFAULT_HARDWARE_BASIS,
    **kwargs
):
    if coupling_map is None:
        raise ValueError("coupling_map is required")

    candidates = collect_circuit_variants_beam(
        circuit,
        max_evaluations,
        f_1q,
        f_2q,
        beam_width
    )

    candidates.sort(key=lambda x: x[1], reverse=True)

    top_candidates = [
        circuit for circuit, _ in candidates[:max_evaluations]
    ]

    original_key = _circuit_key(circuit)

    if not any(_circuit_key(c) == original_key for c in top_candidates):
        top_candidates.append(circuit)

    verified = verify_true_fidelity(
        top_candidates,
        noise_model,
        coupling_map,
        hardware_basis
    )

    original = next(
        x for x in verified
        if _circuit_key(x["circuit"]) == original_key
    )

    best = max(
        verified,
        key=lambda x: x["fidelity"]
    )

    best_key = _circuit_key(best["circuit"])

    best_epa = next(
        epa for c, epa in candidates
        if _circuit_key(c) == best_key
    )

    best_index = next(
        i for i, (c, _) in enumerate(candidates)
        if _circuit_key(c) == best_key
    )

    return {
        "best_circuit": best["circuit"],
        "best_true_fidelity": best["fidelity"],
        "original_true_fidelity": original["fidelity"],
        "best_epa_score": best_epa,
        "improvement": best["fidelity"] - original["fidelity"],
        "search_stats": {
            "nodes_expanded": max(0, len(candidates) - 1),
            "nodes_evaluated": len(candidates),
            "verified_count": len(verified),
            "stopped_early": len(candidates) < max_evaluations,
            "best_candidate_index": best_index
        }
    }