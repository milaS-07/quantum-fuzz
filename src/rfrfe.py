"""
Standalone diagnostic: run from your project root (same place you run the
real experiment) so `circuits` and `transformations` import correctly.

Prints, for one circuit:
  - how many blocks CollectMultiQBlocks actually found
  - for each block: its size, orig_2q count, whether qs_decomposition
    succeeded, new_2q count, and whether it was accepted/skipped/errored
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import PropertySet

from circuits import get_mqt_circuit
import transformations as T

# ---- change these to whichever circuit is collapsing on you ----
BENCHMARK_NAME = "GHZ State"
NUM_QUBITS = 3
MAX_BLOCK_SIZE = getattr(T, "MAX_BLOCK_SIZE", 3)
MAX_BLOCKS_PER_NODE = getattr(T, "MAX_BLOCKS_PER_NODE", 12)
# ------------------------------------------------------------------

qc = get_mqt_circuit(BENCHMARK_NAME, num_qubits=NUM_QUBITS)
print(f"Circuit: {BENCHMARK_NAME}, {NUM_QUBITS}q, gates: {qc.count_ops()}, depth: {qc.depth()}")

dag = circuit_to_dag(qc)
collect_pass = CollectMultiQBlocks(max_block_size=MAX_BLOCK_SIZE)
collect_pass.property_set = PropertySet()
collect_pass.run(dag)
blocks = collect_pass.property_set.get("block_list", [])
print(f"Raw blocks found by CollectMultiQBlocks: {len(blocks)}")

blocks = [b for b in blocks if len(b) >= 2]
print(f"Blocks with len >= 2: {len(blocks)}")

blocks.sort(key=lambda b: sum(1 for n in b if len(n.qargs) == 2), reverse=True)
blocks = blocks[:MAX_BLOCKS_PER_NODE]
print(f"Blocks after MAX_BLOCKS_PER_NODE={MAX_BLOCKS_PER_NODE} cap: {len(blocks)}\n")

accepted, skipped_worse, errored = 0, 0, 0
for i, block_nodes in enumerate(blocks):
    names = [n.op.name for n in block_nodes]
    try:
        unitary, qubits_in_block = T._block_to_unitary(block_nodes)
        orig_2q = sum(1 for n in block_nodes if len(n.qargs) == 2)
        new_block = T._cached_qs_decomposition(unitary)
        new_2q = sum(1 for instr in new_block.data if len(instr.qubits) == 2)
        if new_2q > orig_2q:
            print(f"  Block {i}: ops={names} orig_2q={orig_2q} new_2q={new_2q} -> SKIPPED (worse)")
            skipped_worse += 1
        else:
            print(f"  Block {i}: ops={names} orig_2q={orig_2q} new_2q={new_2q} -> ACCEPTED")
            accepted += 1
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        print(f"  Block {i}: ops={names} -> ERRORED: {type(e).__name__}: {e}")
        errored += 1

print(f"\nTotals: accepted={accepted}, skipped_worse={skipped_worse}, errored={errored}")