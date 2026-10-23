import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
import zlib

current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(
    os.path.join(os.path.dirname(current_file_path), "..", "..")
)
if root_dir not in sys.path:
  sys.path.insert(0, root_dir)

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.quantum_info import Kraus, Operator, average_gate_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator

from src.circuits import get_benchmarks, get_mqt_circuit
from src.noise.noise_parameters import (
    SHERBROOKE_PARAMS,
    depolarizing_thermal_noise_model,
)
from src.optimization_search import _circuit_key, get_matrix_decompositions
from src.predictors.esp_fidelity import (
    gate_errors_from_params,
    predict_fidelity_esp,
)
from src.predictors.rqc_fidelity import average_fidelity, guess_architecture

warnings.filterwarnings("ignore", category=UserWarning)

EXP_DIR = Path(root_dir) / "results" / "noise_predictors" / "exp03"

NUM_WORKERS = min(4, os.cpu_count() or 1)

TIERS = [
    {
        "method": "density_matrix",
        "qubits": [3, 4, 5, 6, 7, 8, 9],
        "levels": [0, 1, 2, 3],
        "max_gates": 6000,
    },
    {
        "method": "trajectory",
        "qubits": [10, 12, 14, 16],
        "levels": [1, 3],
        "max_gates": 4000,
    },
]

SMOKE_TIERS = [
    {
        "method": "density_matrix",
        "qubits": [3, 4],
        "levels": [0, 1, 2, 3],
        "max_gates": 6000,
    },
    {
        "method": "trajectory",
        "qubits": [5],
        "levels": [1, 3],
        "max_gates": 4000,
    },
]

TARGET_SE = 0.015
MIN_SHOTS = 200
MAX_SHOTS = 1500
BATCH_SHOTS = 50
BATCH_BYTES = 256 * 2**20
DM_PRECISION = "double"
SIM_SEED = 12345
TRANSPILE_SEED = 0
STORE_QASM = True

START_LEVEL = 0
SAVE_EVERY = 5
DEFAULT_TARGET_VARIANTS = 30
DEFAULT_MAX_CANDIDATES = 300

HARDWARE_BASIS = ["rz", "sx", "x", "id", "cx"]

BENCHMARKS = get_benchmarks(scalable_only=True)

REMOVE_BARRIERS = PassManager([RemoveBarriers()])


def get_sherbrooke_coupling_map():
  try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
  except ImportError:
    from qiskit.providers.fake_provider import FakeSherbrookeV2 as FakeSherbrooke
  backend = FakeSherbrooke()
  return backend.coupling_map


def build_alpha_p(noise_model):
  costs_1q, costs_2q = [], []
  for _gate_name, quantum_error in noise_model._default_quantum_errors.items():
    n = quantum_error.num_qubits
    kraus = Kraus(quantum_error)
    fid = average_gate_fidelity(kraus, Operator(np.eye(2**n)))
    (costs_1q if n == 1 else costs_2q).append(1.0 - fid)

  alpha = (sum(costs_1q) / len(costs_1q)) if costs_1q else 0.0
  p = (sum(costs_2q) / len(costs_2q)) if costs_2q else 0.0
  return alpha, p


@functools.lru_cache(maxsize=1)
def get_context():
  noise_model = depolarizing_thermal_noise_model(**SHERBROOKE_PARAMS)
  gate_errors = gate_errors_from_params(SHERBROOKE_PARAMS)
  alpha, p = build_alpha_p(noise_model)
  architecture = guess_architecture(get_sherbrooke_coupling_map())
  return {
      "noise_model": noise_model,
      "gate_errors": gate_errors,
      "alpha": alpha,
      "p": p,
      "architecture": architecture,
  }


def extract_circuit_features(qc: QuantumCircuit) -> dict:
  total_depth = qc.depth()
  depth_2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
  ops = qc.count_ops()
  cx_count = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)
  return {
      "total_depth": int(total_depth),
      "depth_2q": int(depth_2q),
      "cx_count": int(cx_count),
  }


def stable_seed(key: str, index: int) -> int:
  return zlib.crc32(f"{SIM_SEED}|{key}|{index}".encode()) % (2**31 - 1)


def circuit_fingerprint(qc: QuantumCircuit) -> str:
  digest = hashlib.sha1()
  for inst in qc.data:
    op = inst.operation
    qubits = ",".join(str(qc.find_bit(q).index) for q in inst.qubits)
    params = ",".join(f"{float(p) + 0.0:.6f}" for p in op.params)
    digest.update(f"{op.name}|{qubits}|{params};".encode())
  return digest.hexdigest()


def build_base_circuit(base_name: str, num_qubits: int) -> QuantumCircuit:
  qc = get_mqt_circuit(base_name, num_qubits=num_qubits)
  qc = qc.remove_final_measurements(inplace=False)
  return REMOVE_BARRIERS.run(qc)


def check_basis(compiled: QuantumCircuit):
  unexpected = set(compiled.count_ops()) - set(HARDWARE_BASIS)
  if unexpected:
    raise ValueError(f"unexpected operations {sorted(unexpected)}")


def compile_to_hardware(base: QuantumCircuit, level: int) -> QuantumCircuit:
  compiled = transpile(
      base,
      basis_gates=HARDWARE_BASIS,
      optimization_level=level,
      seed_transpiler=TRANSPILE_SEED,
  )
  compiled = REMOVE_BARRIERS.run(compiled)
  check_basis(compiled)
  return compiled


def bfs_candidates(start: QuantumCircuit, limit: int):
  visited = {_circuit_key(start)}
  queue = deque([start])
  produced = 0
  while queue and produced < limit:
    current = queue.popleft()
    for successor in get_matrix_decompositions(current):
      key = _circuit_key(successor)
      if key in visited:
        continue
      visited.add(key)
      queue.append(successor)
      produced += 1
      yield successor
      if produced >= limit:
        return


def _as_rows(states):
  if isinstance(states, (list, tuple)):
    return [np.asarray(s) for s in states]
  arr = np.asarray(states)
  return [arr] if arr.ndim == 1 else list(arr)


def _standard_error(total: float, total_sq: float, count: int) -> float:
  if count < 2:
    return float("inf")
  mean = total / count
  variance = max(total_sq / count - mean * mean, 0.0) * count / (count - 1)
  return (variance / count) ** 0.5


def ideal_statevector(circuit: QuantumCircuit, seed: int) -> np.ndarray:
  sim = AerSimulator(method="statevector", max_parallel_threads=1)
  ideal = circuit.copy()
  ideal.save_statevector()
  result = sim.run(ideal, seed_simulator=seed).result()
  return np.asarray(result.data()["statevector"])


def exact_fidelity(circuit, noise_model, psi, seed) -> float:
  sim = AerSimulator(
      noise_model=noise_model,
      method="density_matrix",
      precision=DM_PRECISION,
      max_parallel_threads=1,
  )
  noisy = circuit.copy()
  noisy.save_density_matrix()
  result = sim.run(noisy, seed_simulator=seed).result()
  rho = np.asarray(result.data()["density_matrix"])
  return float(np.real(np.vdot(psi, rho @ psi)))


def trajectory_fidelity(circuit, noise_model, psi, key):
  sim = AerSimulator(
      noise_model=noise_model,
      method="statevector",
      max_parallel_threads=1,
  )
  noisy = circuit.copy()
  noisy.save_statevector(pershot=True)

  state_bytes = 16 * 2 ** max(circuit.num_qubits, 1)
  batch = int(max(1, min(BATCH_SHOTS, BATCH_BYTES // state_bytes)))

  total = 0.0
  total_sq = 0.0
  count = 0
  index = 0
  while count < MAX_SHOTS:
    result = sim.run(
        noisy, shots=batch, seed_simulator=stable_seed(key, index + 1)
    ).result()
    for row in _as_rows(result.data()["statevector"]):
      value = float(abs(np.vdot(psi, row)) ** 2)
      total += value
      total_sq += value * value
      count += 1
    index += 1
    if (
        count >= MIN_SHOTS
        and _standard_error(total, total_sq, count) <= TARGET_SE
    ):
      break
  return total / count, _standard_error(total, total_sq, count), count


def config_signature(tiers) -> str:
  payload = {
      "tiers": [
          {k: v for k, v in t.items() if k != "qubits"} for t in tiers
      ],
      "target_se": TARGET_SE,
      "min_shots": MIN_SHOTS,
      "max_shots": MAX_SHOTS,
      "batch_shots": BATCH_SHOTS,
      "basis": HARDWARE_BASIS,
      "precision": DM_PRECISION,
      "sim_seed": SIM_SEED,
      "transpile_seed": TRANSPILE_SEED,
      "store_qasm": STORE_QASM,
      "noise": SHERBROOKE_PARAMS,
  }
  text = json.dumps(payload, sort_keys=True, default=str)
  return hashlib.sha1(text.encode()).hexdigest()[:10]


def circuit_key(base_name: str, num_qubits: int) -> str:
  return f"{base_name}__{num_qubits}q"


def task_path(task_dir: str, base_name: str, num_qubits: int) -> Path:
  return Path(task_dir) / f"{circuit_key(base_name, num_qubits)}.json"


def write_json_atomic(data: dict, path: Path):
  tmp_path = path.with_suffix(".json.tmp")
  with open(tmp_path, "w") as f:
    json.dump(data, f, indent=1)
  tmp_path.replace(path)


def load_task_result(path: Path):
  try:
    with open(path, "r") as f:
      return json.load(f)
  except Exception:
    return None


def finish_task(record: dict, task: dict, started: float) -> dict:
  elapsed = time.monotonic() - started
  record["elapsed_s"] = round(record.get("elapsed_s", 0.0) + elapsed, 1)
  path = task_path(task["task_dir"], task["base_name"], task["num_qubits"])
  write_json_atomic(record, path)
  return record


def out_of_time(task: dict) -> bool:
  stop_at = task.get("stop_at")
  return stop_at is not None and time.time() >= stop_at


def evaluate_variant(
    hardware, fingerprint, opt_levels, method, base_name, num_qubits, ctx
) -> dict:
  features = extract_circuit_features(hardware)
  width = hardware.num_qubits
  sim_key = f"{base_name}|{num_qubits}|{fingerprint}"

  sim_started = time.monotonic()
  psi = ideal_statevector(hardware, stable_seed(sim_key, 0))
  if method == "density_matrix":
    fidelity_real = exact_fidelity(
        hardware, ctx["noise_model"], psi, stable_seed(sim_key, 1)
    )
    fidelity_se = 0.0
    shots = 0
  else:
    fidelity_real, fidelity_se, shots = trajectory_fidelity(
        hardware, ctx["noise_model"], psi, sim_key
    )
  sim_seconds = time.monotonic() - sim_started

  fid_rqc = float(
      average_fidelity(
          L=width,
          T=features["depth_2q"],
          p=ctx["p"],
          alpha=ctx["alpha"],
          architecture=ctx["architecture"],
      )
  )
  fid_esp = float(predict_fidelity_esp(hardware, ctx["gate_errors"]))

  qasm = None
  if STORE_QASM:
    try:
      qasm = qasm3.dumps(hardware)
    except Exception:
      qasm = None

  return {
      "opt_levels": opt_levels,
      "fingerprint": fingerprint,
      "qasm": qasm,
      "features": features,
      "metrics": {
          "fidelity_real": fidelity_real,
          "fidelity_esp": fid_esp,
          "fidelity_rqc": fid_rqc,
          "total_depth": features["total_depth"],
          "depth_2q": features["depth_2q"],
      },
      "simulation": {
          "method": method,
          "fidelity_real_se": fidelity_se,
          "shots": shots,
          "seconds": round(sim_seconds, 2),
      },
  }


def run_task(task: dict) -> dict:
  started = time.monotonic()
  base_name = task["base_name"]
  num_qubits = task["num_qubits"]
  method = task["method"]
  ctx = get_context()
  path = task_path(task["task_dir"], base_name, num_qubits)

  existing = load_task_result(path) if path.exists() else None
  is_extension = existing is not None and existing.get("status") == "ok"
  if is_extension:
    record = existing
    record.setdefault("skipped_oversize", [])
    record.setdefault("errors", [])
  else:
    record = {
        "base_name": base_name,
        "num_qubits": num_qubits,
        "method": method,
        "status": "ok",
        "variants_tested": [],
        "skipped_oversize": [],
        "errors": [],
    }

  try:
    base = build_base_circuit(base_name, num_qubits)
  except Exception as exc:
    if not is_extension:
      record["status"] = "build_failed"
    record["errors"].append(repr(exc))
    return finish_task(record, task, started)

  record["circuit_qubits"] = base.num_qubits
  if not is_extension and base.num_qubits > task["max_width"]:
    record["status"] = "too_wide"
    record["errors"].append(f"circuit has {base.num_qubits} qubits")
    return finish_task(record, task, started)

  variants = record["variants_tested"]
  seen = {v["fingerprint"]: v for v in variants}

  if not is_extension:
    for level in task["levels"]:
      try:
        hardware = compile_to_hardware(base, level)
        fingerprint = circuit_fingerprint(hardware)

        if fingerprint in seen:
          if seen[fingerprint] is not None:
            seen[fingerprint]["opt_levels"].append(level)
          continue
        seen[fingerprint] = None

        if len(hardware.data) > task["max_gates"]:
          record["skipped_oversize"].append(
              {"opt_level": level, "gates": len(hardware.data)}
          )
          continue

        variant = evaluate_variant(
            hardware, fingerprint, [level], method, base_name, num_qubits, ctx
        )
        variant["variant_id"] = len(variants)
        seen[fingerprint] = variant
        variants.append(variant)
      except Exception as exc:
        if len(record["errors"]) < 5:
          record["errors"].append(f"level {level}: {exc!r}")

  target = task["target_variants"]
  max_candidates = task["max_candidates"]
  generated = 0
  stopped = False
  unsaved = 0

  if len(variants) < target:
    try:
      start = compile_to_hardware(base, START_LEVEL)
    except Exception as exc:
      start = None
      if len(record["errors"]) < 5:
        record["errors"].append(f"bfs start: {exc!r}")

    if start is not None:
      for candidate in bfs_candidates(start, max_candidates):
        generated += 1
        if out_of_time(task):
          stopped = True
          break
        try:
          hardware = compile_to_hardware_candidate(candidate)
          fingerprint = circuit_fingerprint(hardware)
          if fingerprint in seen:
            continue
          seen[fingerprint] = None

          if len(hardware.data) > task["max_gates"]:
            continue

          variant = evaluate_variant(
              hardware, fingerprint, [], method, base_name, num_qubits, ctx
          )
          variant["variant_id"] = len(variants)
          variant["origin"] = "bfs"
          variant["bfs_index"] = generated
          seen[fingerprint] = variant
          variants.append(variant)
          unsaved += 1
          if unsaved >= SAVE_EVERY:
            finish_task(record, task, started)
            started = time.monotonic()
            unsaved = 0
        except Exception as exc:
          if len(record["errors"]) < 5:
            record["errors"].append(f"bfs candidate {generated}: {exc!r}")
        if len(variants) >= target:
          break

      if not stopped and len(variants) < target:
        if generated >= max_candidates:
          record["bfs_cap"] = max_candidates
        else:
          record["bfs_exhausted"] = True

  if not variants:
    record["status"] = "no_variants"
  return finish_task(record, task, started)


def compile_to_hardware_candidate(candidate: QuantumCircuit) -> QuantumCircuit:
  compiled = transpile(
      candidate,
      basis_gates=HARDWARE_BASIS,
      optimization_level=0,
      seed_transpiler=TRANSPILE_SEED,
  )
  compiled = REMOVE_BARRIERS.run(compiled)
  check_basis(compiled)
  return compiled


def build_tasks(tiers, benchmarks, task_dir: Path) -> list:
  tasks = []
  for tier in tiers:
    max_width = max(tier["qubits"]) + 1
    for num_qubits in tier["qubits"]:
      for base_name in benchmarks:
        tasks.append({
            "base_name": base_name,
            "num_qubits": num_qubits,
            "method": tier["method"],
            "levels": tier["levels"],
            "max_gates": tier["max_gates"],
            "max_width": max_width,
            "task_dir": str(task_dir),
        })
  tasks.sort(key=lambda t: t["num_qubits"])
  return tasks


def needs_work(existing, target: int, max_candidates: int) -> bool:
  if existing is None:
    return True
  if existing.get("status") != "ok":
    return False
  if len(existing.get("variants_tested", [])) >= target:
    return False
  if existing.get("bfs_exhausted"):
    return False
  return existing.get("bfs_cap", 0) < max_candidates


def write_merged(records: dict, theory_params: dict, tiers, path: Path):
  circuits = [r for r in records.values() if r["status"] == "ok"]
  circuits.sort(key=lambda r: (r["num_qubits"], r["base_name"]))
  skipped = [
      {
          "base_name": r["base_name"],
          "num_qubits": r["num_qubits"],
          "status": r["status"],
          "errors": r.get("errors", []),
      }
      for r in records.values()
      if r["status"] != "ok"
  ]
  data = {
      "timestamp": datetime.now().isoformat(),
      "theory_params": theory_params,
      "tiers": tiers,
      "config": {
          "target_se": TARGET_SE,
          "min_shots": MIN_SHOTS,
          "max_shots": MAX_SHOTS,
          "precision": DM_PRECISION,
          "basis": HARDWARE_BASIS,
      },
      "circuits": circuits,
      "skipped": skipped,
  }
  write_json_atomic(data, path)


def run_pool(pending: list, workers: int, deadline, on_result):
  with ProcessPoolExecutor(max_workers=workers) as executor:
    queue = iter(pending)
    in_flight = {}
    exhausted = False
    hit_deadline = False
    while True:
      while not exhausted and len(in_flight) < workers * 2:
        task = next(queue, None)
        if task is None:
          exhausted = True
          break
        if deadline is not None and time.monotonic() >= deadline:
          hit_deadline = True
          exhausted = True
          break
        in_flight[executor.submit(run_task, task)] = task
      if not in_flight:
        break
      finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
      for future in finished:
        task = in_flight.pop(future)
        try:
          result = future.result()
        except Exception as exc:
          result = {
              "base_name": task["base_name"],
              "num_qubits": task["num_qubits"],
              "method": task["method"],
              "status": "worker_crashed",
              "variants_tested": [],
              "errors": [repr(exc)],
          }
        on_result(task, result)
  return hit_deadline


def print_summary(records: dict):
  print("\nname  qubits  levels  real  se  esp  rqc  seconds")
  for record in sorted(
      records.values(), key=lambda r: (r["num_qubits"], r["base_name"])
  ):
    for v in record["variants_tested"]:
      m = v["metrics"]
      s = v["simulation"]
      print(
          f"{record['base_name']:<18} {record['num_qubits']:>3} "
          f"{str(v['opt_levels']):<14} real={m['fidelity_real']:.4f} "
          f"se={s['fidelity_real_se']:.4f} esp={m['fidelity_esp']:.4f} "
          f"rqc={m['fidelity_rqc']:.4f} cx={v['features']['cx_count']} "
          f"t={s['seconds']}s"
      )


def crosscheck(benchmarks, num_qubits: int = 6, level: int = 1, limit: int = 4):
  ctx = get_context()
  print(f"exact vs trajectory at {num_qubits}q, opt level {level}")
  for name in benchmarks[:limit]:
    try:
      hardware = compile_to_hardware(
          build_base_circuit(name, num_qubits), level
      )
      key = f"crosscheck|{name}|{num_qubits}"
      psi = ideal_statevector(hardware, stable_seed(key, 0))
      exact = exact_fidelity(
          hardware, ctx["noise_model"], psi, stable_seed(key, 1)
      )
      mean, se, shots = trajectory_fidelity(
          hardware, ctx["noise_model"], psi, key
      )
      z = (mean - exact) / se if se > 0 else float("nan")
      print(
          f"{name:<18} exact={exact:.4f} traj={mean:.4f} se={se:.4f} "
          f"shots={shots} z={z:+.2f}"
      )
    except Exception as exc:
      print(f"{name:<18} failed: {exc!r}")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--crosscheck", action="store_true")
  parser.add_argument("--workers", type=int, default=NUM_WORKERS)
  parser.add_argument("--hours", type=float, default=None)
  parser.add_argument("--target", type=int, default=DEFAULT_TARGET_VARIANTS)
  parser.add_argument(
      "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES
  )
  parser.add_argument(
      "--method", choices=["density_matrix", "trajectory"], default=None
  )
  return parser.parse_args()


def main():
  args = parse_args()
  benchmarks = list(BENCHMARKS)

  if args.crosscheck:
    crosscheck(benchmarks)
    return

  if args.smoke:
    tiers = SMOKE_TIERS
    benchmarks = benchmarks[:3]
    out_dir = EXP_DIR / "smoke"
  else:
    tiers = TIERS
    out_dir = EXP_DIR

  task_dir = out_dir / f"tasks_{config_signature(tiers)}"
  merged_path = out_dir / "fidelity_comparison.json"
  task_dir.mkdir(parents=True, exist_ok=True)

  ctx = get_context()
  theory_params = {
      "alpha": ctx["alpha"],
      "p": ctx["p"],
      "architecture": ctx["architecture"],
      "gate_errors": ctx["gate_errors"],
  }

  tasks = build_tasks(tiers, benchmarks, task_dir)
  total = len(tasks)

  stop_at = None
  if args.hours is not None:
    stop_at = time.time() + args.hours * 3600

  records = {}
  pending = []
  for task in tasks:
    path = task_path(task["task_dir"], task["base_name"], task["num_qubits"])
    existing = load_task_result(path) if path.exists() else None
    if existing is not None:
      records[circuit_key(task["base_name"], task["num_qubits"])] = existing
    if args.method is not None and task["method"] != args.method:
      continue
    if needs_work(existing, args.target, args.max_candidates):
      task["target_variants"] = args.target
      task["max_candidates"] = args.max_candidates
      task["stop_at"] = stop_at
      pending.append(task)

  variants_now = sum(
      len(r.get("variants_tested", [])) for r in records.values()
  )
  print(
      f"[*] {total} tasks, {len(records)} with results "
      f"({variants_now} variants), {len(pending)} to extend or run "
      f"on {args.workers} workers, target {args.target} variants each",
      flush=True,
  )

  started = time.monotonic()
  deadline = None
  if args.hours is not None:
    deadline = started + args.hours * 3600

  def on_result(task, result):
    records[circuit_key(task["base_name"], task["num_qubits"])] = result
    write_merged(records, theory_params, tiers, merged_path)
    slowest = max(
        (v["simulation"]["seconds"] for v in result["variants_tested"]),
        default=0.0,
    )
    print(
        f"[+] {task['base_name']} {task['num_qubits']}q {task['method']} "
        f"status={result['status']} "
        f"variants={len(result['variants_tested'])} slowest_sim={slowest}s "
        f"elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )

  stopped_early = run_pool(pending, args.workers, deadline, on_result)
  write_merged(records, theory_params, tiers, merged_path)

  if stopped_early or (stop_at is not None and time.time() >= stop_at):
    print("[*] Time limit reached. Run again to continue where it stopped.")
  else:
    print(f"[*] Finished. Results: {merged_path}")

  if args.smoke:
    print_summary(records)


if __name__ == "__main__":
  main()