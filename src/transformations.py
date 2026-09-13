import sys
import os
import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import qs_decomposition
from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import PropertySet

warnings.filterwarnings("ignore", category=UserWarning)

MAX_BLOCK_SIZE = 3
MAX_BLOCKS_PER_NODE = 12


class SuppressRustPanic:
    def __enter__(self):
        sys.stderr.flush()
        try:
            self.fd = sys.stderr.fileno()
            self.old_stderr = os.dup(self.fd)
            self.devnull = open(os.devnull, 'w')
            os.dup2(self.devnull.fileno(), self.fd)
        except Exception:
            self.fd = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            sys.stderr.flush()
            os.dup2(self.old_stderr, self.fd)
            self.devnull.close()
            os.close(self.old_stderr)


def _block_to_unitary(block_nodes):
    qubits_in_block = []
    for node in block_nodes:
        for q in node.qargs:
            if q not in qubits_in_block:
                qubits_in_block.append(q)
    qmap = {q: i for i, q in enumerate(qubits_in_block)}
    sub = QuantumCircuit(len(qubits_in_block))
    for node in block_nodes:
        sub.append(node.op, [qmap[q] for q in node.qargs])
    return Operator(sub).data, qubits_in_block


def _replace_block(circuit, block_nodes, qubits_in_block, new_block):
    """Rebuilds `circuit` with `block_nodes` replaced by `new_block`.

    Matches block membership via id(instr.operation) against circuit.data,
    NOT via DAGNode identity — DAGNode wrapper objects returned by
    dag.topological_op_nodes() are not guaranteed to keep a stable id()
    across separate calls/traversals in current Qiskit (rustworkx-backed
    DAGs re-wrap nodes on access). The underlying `.operation` / `.op`
    payload object, however, IS the same object shared between
    circuit.data and the DAG's node.op, so we key off that instead.
    """
    block_op_ids = {id(n.op) for n in block_nodes}
    qmap = {new_block.qubits[i]: q for i, q in enumerate(qubits_in_block)}

    new_circuit = circuit.copy_empty_like()
    emitted = False
    for instr in circuit.data:
        if id(instr.operation) in block_op_ids:
            if not emitted:
                for new_instr in new_block.data:
                    new_circuit.append(new_instr.operation, [qmap[q] for q in new_instr.qubits])
                emitted = True
            continue
        new_circuit.append(instr.operation, instr.qubits, instr.clbits)
    return new_circuit


def _round_unitary_key(unitary: np.ndarray) -> bytes:
    return np.round(unitary, 6).tobytes()


_DECOMP_CACHE: dict = {}


def _cached_qs_decomposition(unitary: np.ndarray):
    key = _round_unitary_key(unitary)
    if key not in _DECOMP_CACHE:
        with SuppressRustPanic():
            _DECOMP_CACHE[key] = qs_decomposition(unitary)
    return _DECOMP_CACHE[key]


def get_matrix_decompositions(circuit: QuantumCircuit, max_block_size: int = MAX_BLOCK_SIZE):
    dag = circuit_to_dag(circuit)

    collect_pass = CollectMultiQBlocks(max_block_size=max_block_size)
    collect_pass.property_set = PropertySet()
    collect_pass.run(dag)
    blocks = collect_pass.property_set.get("block_list", [])

    blocks = [b for b in blocks if len(b) >= 2]
    blocks.sort(key=lambda b: sum(1 for n in b if len(n.qargs) == 2), reverse=True)
    blocks = blocks[:MAX_BLOCKS_PER_NODE]

    successors = []
    for block_nodes in blocks:
        try:
            unitary, qubits_in_block = _block_to_unitary(block_nodes)
            orig_2q = sum(1 for n in block_nodes if len(n.qargs) == 2)
            new_block = _cached_qs_decomposition(unitary)
            new_2q = sum(1 for instr in new_block.data if len(instr.qubits) == 2)
            if new_2q > orig_2q:
                continue
            new_circuit = _replace_block(circuit, block_nodes, qubits_in_block, new_block)
            successors.append((new_circuit, ""))
        except KeyboardInterrupt:
            raise
        except BaseException:
            continue
    return successors