import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import qiskit
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import CZGate
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, random_statevector, random_unitary

MODE = "full"
EXP_NAME = "exp02"
SEEDS = "0-29"
JOBS = 4
REUSE_ONLY = {0}

PRESETS = {
    "full": {},
    "crash_check": dict(only="Amplitude Estimation", circuits=0, no_mutation=True, no_determinism=True,
                        seeds="1"),
    "quick": dict(circuits=30, variants=10, walks=1, depth=4, bench_variants=35, bench_walks=1,
                  bench_depth=4, mqt_sizes="3,4", min_circuits=30, min_per_size=2, min_per_family=5,
                  min_per_move=10, mutation_circuits=10, e2e_generated=2, e2e_evaluations=8, seeds="2"),
}

SETTING_EXCLUDE = {"seed", "seeds", "jobs", "rerun", "aggregate_only", "optimizer_file", "results_dir",
                   "exp", "no_determinism", "src", "max_dumps", "child_pipeline", "print_digest"}

FAMILIES = ["random2q", "staggered", "redundant", "random3q", "structured"]
REQUIRED_MOVES = ["block-2q", "block-3q", "window-2q", "window-3q", "peephole"]
REQUIRED_JUMPS = ["+jump-back", "+jump-fwd"]
CONTROL_FLOW = {"if_else", "while_loop", "for_loop", "switch_case"}


def ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def qft(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / 2 ** (k - j), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def hea(n, layers=3, seed=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
            qc.rz(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def trotter(n, steps=2, seed=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for q in range(n - 1):
            t = float(rng.uniform(0.1, 1.0))
            qc.rxx(t, q, q + 1)
            qc.ryy(t, q, q + 1)
            qc.rzz(t, q, q + 1)
    return qc


ONE_Q = ["h", "x", "y", "z", "s", "t", "sx"]
ROT_1Q = ["rx", "ry", "rz"]


def _add_random_1q(qc, q, rng):
    if rng.random() < 0.5:
        getattr(qc, ONE_Q[int(rng.integers(len(ONE_Q)))])(q)
    else:
        getattr(qc, ROT_1Q[int(rng.integers(len(ROT_1Q)))])(float(rng.uniform(0, 2 * np.pi)), q)


def _add_random_2q(qc, a, b, rng):
    k = int(rng.integers(5))
    if k == 0:
        qc.cx(a, b)
    elif k == 1:
        qc.cz(a, b)
    elif k == 2:
        qc.cx(b, a)
    elif k == 3:
        qc.rzz(float(rng.uniform(0, 2 * np.pi)), a, b)
    else:
        qc.cp(float(rng.uniform(0, 2 * np.pi)), a, b)


def staggered(n, n_gates, seed, p1=0.5):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(n_gates):
        if rng.random() < p1:
            _add_random_1q(qc, int(rng.integers(n)), rng)
        else:
            a = int(rng.integers(n - 1))
            _add_random_2q(qc, a, a + 1, rng)
    return qc


def with_redundant_pairs(qc, seed, k):
    rng = np.random.default_rng(seed)
    out = QuantumCircuit(qc.num_qubits)
    where = Counter(int(x) for x in rng.integers(0, len(qc.data) + 1, size=k))
    for i in range(len(qc.data) + 1):
        for _ in range(where.get(i, 0)):
            name = ["h", "x", "y", "z", "cx", "cz"][int(rng.integers(6))]
            if name in ("cx", "cz"):
                a, b = (int(v) for v in rng.choice(qc.num_qubits, size=2, replace=False))
                args = (a, b)
            else:
                args = (int(rng.integers(qc.num_qubits)),)
            for _ in range(2):
                getattr(out, name)(*args)
        if i < len(qc.data):
            ins = qc.data[i]
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def prepare(circ):
    c = circ.remove_final_measurements(inplace=False)
    if any(i.operation.name in ("measure", "reset") for i in c.data):
        return None
    return c


def build_suite(n_circuits, min_q, max_q, seed):
    rng = np.random.default_rng(seed)
    suite = []
    for i in range(n_circuits):
        fam = FAMILIES[i % len(FAMILIES)]
        k = i // len(FAMILIES)
        n = min_q + k % (max_q - min_q + 1)
        s = int(rng.integers(2 ** 31 - 1))
        label = fam
        if fam == "random2q":
            qc = random_circuit(n, int(rng.integers(3, 9)), max_operands=2, seed=s)
        elif fam == "random3q":
            qc = random_circuit(n, int(rng.integers(3, 8)), max_operands=3, seed=s)
        elif fam == "staggered":
            qc = staggered(n, int(rng.integers(8, 30)), s)
        elif fam == "redundant":
            qc = with_redundant_pairs(staggered(n, int(rng.integers(6, 20)), s), s + 1,
                                      int(rng.integers(1, 4)))
        else:
            kind = ["ghz", "qft", "hea", "trotter"][(k + k // 6) % 4]
            label = f"structured-{kind}"
            qc = {"ghz": lambda: ghz(n), "qft": lambda: qft(n),
                  "hea": lambda: hea(n, 2, s % 1000),
                  "trotter": lambda: trotter(n, 1, s % 1000)}[kind]()
        qc = prepare(qc)
        if qc is None or len(qc.data) == 0:
            continue
        suite.append(dict(name=f"{label}_{i:03d}_n{n}", family=fam, n=n, seed=s, circuit=qc))
    return suite


def _digest(text):
    return hashlib.sha1(text.encode()).hexdigest()


def prepare_benchmark(name, circ, cfg, native=False):
    note = ""
    try:
        if len(circ.parameters):
            r = np.random.default_rng(int(_digest(name)[:8], 16) + cfg.seed)
            circ = circ.assign_parameters({p: float(r.uniform(0, 2 * np.pi)) for p in circ.parameters})
            note = "parameters bound to random values"
        c = prepare(circ)
        if c is None:
            return None, "mid-circuit measurement or reset"
        if any(i.operation.name in CONTROL_FLOW for i in c.data):
            return None, "contains classical control flow"
        if native:
            from qiskit import transpile
            c = transpile(c, basis_gates=["x", "sx", "rz", "cx"], optimization_level=0,
                          seed_transpiler=cfg.seed)
        if len(c.data) == 0:
            return None, "no gates"
        if c.num_qubits > cfg.bench_max_qubits:
            return None, f"{c.num_qubits} qubits > --bench-max-qubits {cfg.bench_max_qubits}"
        if len(c.data) > cfg.bench_max_gates:
            return None, f"{len(c.data)} gates > --bench-max-gates {cfg.bench_max_gates}"
        Checker(c.num_qubits, cfg.max_unitary_qubits, cfg.states, cfg.seed).sig(c)
        return c, note
    except Exception as exc:
        return None, f"cannot evaluate: {type(exc).__name__}: {str(exc)[:100]}"


def load_benchmarks(cfg, bench_mod):
    forms = [f.strip() for f in cfg.mqt_forms.split(",") if f.strip()]
    sizes = [int(x) for x in cfg.mqt_sizes.split(",") if x.strip()]
    raw, skipped = [], []

    def add(label, qc):
        if "alg" in forms:
            raw.append((f"{label} [alg]", qc, False))
        if "native" in forms:
            raw.append((f"{label} [native]", qc, True))

    for fn_name in sorted(dir(bench_mod)):
        if fn_name == "get_empty_circuit" or fn_name.startswith("get_rnd_circuit_"):
            try:
                add(f"circuits.{fn_name}", getattr(bench_mod, fn_name)())
            except Exception as exc:
                skipped.append((f"circuits.{fn_name}", f"cannot build: {str(exc)[:100]}"))
    for bname in bench_mod.get_benchmarks():
        scalable = bench_mod.BENCHMARK_REGISTRY.get(bname, {}).get("scalable", True)
        for n in (sizes if scalable else [None]):
            label = f"mqt:{bname}" + (f" n={n}" if n else "")
            try:
                add(label, bench_mod.get_mqt_circuit(bname, n))
            except Exception as exc:
                skipped.append((label, f"cannot build: {str(exc)[:110]}"))

    items, seen = [], set()
    for name, circ, native in raw:
        c, note = prepare_benchmark(name, circ, cfg, native)
        if c is None:
            skipped.append((name, note))
            continue
        try:
            d = _digest(qasm3.dumps(c))
        except Exception:
            d = name
        if d in seen:
            skipped.append((name, "duplicate of another benchmark circuit"))
            continue
        seen.add(d)
        items.append(dict(name=name, family="benchmark", n=c.num_qubits, seed=0, circuit=c, note=note))

    if cfg.only:
        terms = [t.strip().lower() for t in cfg.only.split(",") if t.strip()]
        items = [it for it in items if any(t in it["name"].lower() for t in terms)]
        skipped = [x for x in skipped if any(t in x[0].lower() for t in terms)]
    return items, skipped


def _phase_dist(a, b):
    tr = np.vdot(a, b)
    if abs(tr) < 1e-12:
        return float("inf"), 0.0
    ph = tr / abs(tr)
    return float(np.max(np.abs(b - ph * a))), float(np.angle(tr))


class Checker:

    def __init__(self, n, max_unitary_qubits, n_states, seed):
        self.use_unitary = n <= max_unitary_qubits
        rng = np.random.default_rng(seed)
        self.psis = [] if self.use_unitary else [
            random_statevector(2 ** n, seed=int(rng.integers(2 ** 31 - 1))) for _ in range(n_states)]

    def sig(self, circ):
        if self.use_unitary:
            return Operator(circ).data
        return [p.evolve(circ).data for p in self.psis]

    def dist(self, sa, sb):
        if self.use_unitary:
            return _phase_dist(sa, sb)
        results = [_phase_dist(a, b) for a, b in zip(sa, sb)]
        return max(e for e, _ in results), results[0][1]


def verify_pair(original, optimized, tol=1e-7, max_unitary_qubits=6, states=3, seed=0):
    a, b = prepare(original), prepare(optimized)
    chk = Checker(a.num_qubits, max_unitary_qubits, states, seed)
    err, ph = chk.dist(chk.sig(a), chk.sig(b))
    return err <= tol, err, ph


def selftest(tol):
    base = qft(4)
    wrong = base.copy()
    wrong.rz(1e-4, 0)
    phased = base.copy()
    phased.global_phase += 0.3
    for max_u, mode in ((10, "unitary"), (0, "statevector")):
        chk = Checker(4, max_u, 3, 0)
        s0 = chk.sig(base)
        e_wrong, _ = chk.dist(s0, chk.sig(wrong))
        e_ph, ph = chk.dist(s0, chk.sig(phased))
        ok = e_wrong > tol and e_ph < tol and abs(ph - 0.3) < 1e-6
        print(f"  selftest [{mode}]: wrong-circuit err={e_wrong:.2e}, phase-only err={e_ph:.2e}, "
              f"phase={ph:.3f}  ->  {'OK' if ok else 'BROKEN'}")
        if not ok:
            return False
    return True


def synthesis_probe(opt):
    fn = getattr(opt, "_synthesize_unitary", None)
    if fn is None:
        print("  synthesis probe: skipped (no _synthesize_unitary in the optimizer)")
        return True
    ok_all = True
    for n in (2, 3):
        U = random_unitary(2 ** n, seed=7 + n).data
        try:
            err, _ = _phase_dist(U, Operator(fn(U, n)).data)
            ok, msg = err < 1e-7, f"err={err:.1e}"
        except Exception as exc:
            ok, msg = False, f"raised {type(exc).__name__}: {str(exc)[:700]}"
        print(f"  synthesis probe [{n} qubits]: {msg}  ->  {'OK' if ok else 'BROKEN'}")
        ok_all = ok_all and ok
    return ok_all


class Metrics:
    def __init__(self):
        self.per_move = defaultdict(lambda: dict(tests=0, fails=0, phase=0, max_err=0.0))
        self.per_family = defaultdict(lambda: dict(circuits=0, tests=0, fails=0))
        self.per_size = defaultdict(lambda: dict(circuits=0, tests=0, fails=0))
        self.per_circuit, self.failures = [], []
        self.distinct = set()
        self.tests = self.fails = self.phase = self.noops = self.failed_circuits = 0
        self.gen_s = self.chk_s = self.max_err = 0.0

    def to_dict(self):
        d = dict(vars(self))
        d["distinct"] = sorted(self.distinct)
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls()
        for key, val in d.items():
            cur = getattr(m, key)
            if isinstance(cur, defaultdict):
                cur.update({(int(k) if key == "per_size" else k): v for k, v in val.items()})
            elif isinstance(cur, set):
                cur.update(val)
            else:
                setattr(m, key, val)
        return m


def save_checkpoint(path, done, m, rng_state):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(dict(done=sorted(done), metrics=m.to_dict(), rng_state=rng_state),
                              default=str), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(d["done"]), Metrics.from_dict(d["metrics"]), d["rng_state"]


def cp_upper(k, n, alpha=0.05):
    if n <= 0 or k >= n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha, k + 1, n - k))
    except ImportError:
        z, p = 1.645, k / n
        centre = p + z * z / (2 * n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return float((centre + half) / (1 + z * z / n))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _crash_where(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    return f" [{Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name}]" if frames else ""


def dump_failure(fail_dir, bench, kind, label, rname, ref, cand):
    d = Path(fail_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{bench}_{kind}_{label}_vs_{rname}"
    try:
        (d / f"{stem}_ref.qasm").write_text(qasm3.dumps(ref))
        (d / f"{stem}_cand.qasm").write_text(qasm3.dumps(cand))
    except Exception as exc:
        print(f"    (could not dump {stem}: {exc})")


LAST_INPUT = {}


def dump_crash(fail_dir, item, exc):
    d = Path(fail_dir)
    stem = _safe(item["name"])
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}_crash_traceback.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
        (d / f"{stem}_crash_item.qasm").write_text(qasm3.dumps(item["circuit"]))
        last = LAST_INPUT.get("circ")
        if last is not None:
            (d / f"{stem}_crash_optimizer_input.qasm").write_text(qasm3.dumps(last))
    except Exception as err:
        print(f"    (could not dump crash files for {stem}: {err})")


def check_circuit(item, opt, cfg, rng, m, key_fn, fail_dir=None):
    circ, name = item["circuit"], item["name"]
    LAST_INPUT["circ"] = None
    chk = Checker(circ.num_qubits, cfg.max_unitary_qubits, cfg.states, cfg.seed)
    t = time.perf_counter()
    ref = chk.sig(circ)
    m.chk_s += time.perf_counter() - t
    fam, size = m.per_family[item["family"]], m.per_size[circ.num_qubits]
    fam["circuits"] += 1
    size["circuits"] += 1
    state = dict(tests=0, fails=0, failed=False, max_err=0.0)
    seen = set()

    def judge(tag, label, cand, refs):
        t0 = time.perf_counter()
        try:
            s = chk.sig(cand)
        except Exception as exc:
            m.fails += 1
            fam["fails"] += 1
            size["fails"] += 1
            state["failed"] = True
            m.failures.append(dict(circuit=name, tag=tag, label=label, ref="-", err=None,
                                   note=f"cannot evaluate candidate: {exc}"))
            m.chk_s += time.perf_counter() - t0
            return None
        for rname, rcirc, rsig in refs:
            err, ph = chk.dist(rsig, s)
            phase_off = err <= cfg.tol and abs(np.angle(np.exp(1j * ph))) > cfg.tol
            bad = err > cfg.tol
            key = "beam" if tag == "beam" else (tag if rname == "parent" else "cumulative")
            b = m.per_move[key]
            for st in (b, fam, size):
                st["tests"] += 1
            m.tests += 1
            state["tests"] += 1
            b["max_err"] = max(b["max_err"], err)
            m.max_err = max(m.max_err, err)
            state["max_err"] = max(state["max_err"], err)
            if bad or (phase_off and cfg.strict_phase):
                for st in (b, fam, size):
                    st["fails"] += 1
                m.fails += 1
                state["fails"] += 1
                state["failed"] = True
                m.failures.append(dict(circuit=name, tag=tag, label=label, ref=rname, err=err, phase=ph))
                if fail_dir and len(m.failures) <= cfg.max_dumps:
                    dump_failure(fail_dir, _safe(name), _safe(key), label, rname, rcirc, cand)
            elif phase_off:
                b["phase"] += 1
                m.phase += 1
        m.chk_s += time.perf_counter() - t0
        return s

    def record(cand):
        dg = _digest(key_fn(cand))
        m.distinct.add(dg)
        seen.add(dg)

    LAST_INPUT["circ"] = circ
    t = time.perf_counter()
    cands = opt.collect_circuit_variants_beam(circ, cfg.variants, cfg.f1q, cfg.f2q, cfg.beam_width)
    m.gen_s += time.perf_counter() - t
    for idx, (cand, _e) in enumerate(cands):
        if idx == 0:
            continue
        record(cand)
        judge("beam", f"b{idx}", cand, [("orig", circ, ref)])

    for w in range(cfg.walks):
        cur, sig_cur = circ, ref
        for step in range(cfg.depth):
            tags = []
            LAST_INPUT["circ"] = cur
            t = time.perf_counter()
            try:
                succ = opt.get_matrix_decompositions(cur, tags=tags)
            except TypeError:
                tags, succ = None, opt.get_matrix_decompositions(cur)
            m.gen_s += time.perf_counter() - t
            if not succ:
                break
            if tags is not None and len(tags) != len(succ):
                tags = None
            cur_key = key_fn(cur)
            chosen = chosen_sig = None
            for j, i in enumerate(rng.permutation(len(succ))[: cfg.succ_cap]):
                cand = succ[int(i)]
                if key_fn(cand) == cur_key:
                    m.noops += 1
                    continue
                record(cand)
                refs = [("parent", cur, sig_cur)]
                if chosen is None:
                    refs.append(("orig", circ, ref))
                s = judge(tags[int(i)] if tags else "unknown", f"w{w}s{step}c{j}", cand, refs)
                if chosen is None and s is not None:
                    chosen, chosen_sig = cand, s
            if chosen is None:
                break
            cur, sig_cur = chosen, chosen_sig

    if state["failed"]:
        m.failed_circuits += 1
    m.per_circuit.append(dict(
        name=name, family=item["family"], n=circ.num_qubits, gates=len(circ.data),
        mode="unitary" if chk.use_unitary else "states", transformed=len(seen),
        beam_variants=max(0, len(cands) - 1), exhausted=len(cands) < cfg.variants,
        tests=state["tests"], fails=state["fails"], max_err=state["max_err"]))


@contextmanager
def patched(mod, name, fn):
    orig = getattr(mod, name)
    setattr(mod, name, fn)
    try:
        yield
    finally:
        setattr(mod, name, orig)


def _rebuild(c, transform):
    out = c.copy_empty_like()
    for i, ins in enumerate(c.data):
        op, qs = transform(i, ins)
        out.append(op, qs, ins.clbits)
    return out


def m_drop_gate(c):
    if len(c.data) < 2:
        return None
    k = len(c.data) // 2
    out = c.copy_empty_like()
    for i, ins in enumerate(c.data):
        if i != k:
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def m_extra_rotation(c):
    out = c.copy()
    out.rz(1e-3, 0)
    return out


def m_swap_cx_qubits(c):
    for i, ins in enumerate(c.data):
        if ins.operation.name == "cx":
            return _rebuild(c, lambda j, x: (x.operation, tuple(reversed(x.qubits)) if j == i else x.qubits))
    return None


def m_cx_to_cz(c):
    for i, ins in enumerate(c.data):
        if ins.operation.name == "cx":
            return _rebuild(c, lambda j, x: (CZGate() if j == i else x.operation, x.qubits))
    return None


def m_phase_only(c):
    out = c.copy()
    out.global_phase += 0.4
    return out


OUTPUT_MUTANTS = [
    ("drop-a-gate", "one gate disappears from every successor", m_drop_gate, "fail"),
    ("extra-rotation", "successor gets an extra rz(1e-3)", m_extra_rotation, "fail"),
    ("reversed-cx", "control/target of the first cx swapped", m_swap_cx_qubits, "fail"),
    ("cx-becomes-cz", "first cx replaced by cz", m_cx_to_cz, "fail"),
    ("phase-only", "global phase shifted by 0.4 (harmless, must show up as 'phase')", m_phase_only, "phase"),
]


def _post_wrapper(orig_fn, mutate, counter):
    def wrapper(circuit, *a, **kw):
        out = []
        for c in orig_fn(circuit, *a, **kw):
            c2 = mutate(c)
            if c2 is None:
                out.append(c)
            else:
                counter["applied"] += 1
                out.append(c2)
        return out
    return wrapper


def internal_mutants(opt):
    muts = []
    if hasattr(opt, "_emit_index"):
        def emit_bug(counter):
            orig = opt._emit_index

            def bug(circuit, block_indices):
                idx = min(block_indices)
                try:
                    true_idx, _kind = orig(circuit, block_indices)
                    if true_idx != idx:
                        counter["applied"] += 1
                except ValueError:
                    counter["applied"] += 1
                return idx, "first"
            return "_emit_index", bug
        muts.append(("emit-ignores-dependencies",
                     "block always emitted at its first gate, gates on shared qubits are jumped over",
                     emit_bug))
    if hasattr(opt, "_replace_window"):
        def window_bug(counter):
            orig = opt._replace_window

            def bug(circuit, start, end, new_block, sub_qubits):
                counter["applied"] += 1
                return orig(circuit, start, end, new_block, list(reversed(sub_qubits)))
            return "_replace_window", bug
        muts.append(("window-reversed-qubits", "window replacement uses the wrong qubit order", window_bug))
    if hasattr(opt, "_peephole_successors"):
        def peep_bug(counter):
            def bug(circuit):
                succ = []
                for i in range(len(circuit.data) - 1):
                    a, b = circuit.data[i], circuit.data[i + 1]
                    if a.operation.name in {"x", "y", "z", "h", "cx", "cz"} \
                            and a.operation.name == b.operation.name:
                        if a.qubits != b.qubits:
                            counter["applied"] += 1
                        new = circuit.copy_empty_like()
                        for j, ins in enumerate(circuit.data):
                            if j not in (i, i + 1):
                                new.append(ins.operation, ins.qubits, ins.clbits)
                        succ.append(new)
                return succ
            return "_peephole_successors", bug
        muts.append(("peephole-ignores-qubits",
                     "cancels adjacent equal gates even when they act on different qubits", peep_bug))
    return muts


def run_mutation_tests(opt, suite, cfg, key_fn):
    subset = [it for it in suite if it["circuit"].num_qubits <= cfg.mutation_max_qubits
              and len(it["circuit"].data) <= cfg.mutation_max_gates]
    step = max(1, len(subset) // max(1, cfg.mutation_circuits))
    subset = subset[::step][: cfg.mutation_circuits]
    small = argparse.Namespace(**vars(cfg))
    small.variants, small.walks, small.depth, small.succ_cap = 10, 1, 3, 8

    plans = []
    for name, desc, fn, expect in OUTPUT_MUTANTS:
        def factory(counter, fn=fn):
            return "get_matrix_decompositions", _post_wrapper(opt.get_matrix_decompositions, fn, counter)
        plans.append((name, desc, factory, expect))
    plans += [(name, desc, factory, "fail") for name, desc, factory in internal_mutants(opt)]

    results = []
    for name, desc, factory, expect in plans:
        counter = Counter()
        rng = np.random.default_rng(cfg.seed + 99)
        mm = Metrics()
        used = 0
        target, repl = factory(counter)
        with patched(opt, target, repl):
            for item in subset:
                used += 1
                check_circuit(item, opt, small, rng, mm, key_fn)
                if (expect == "fail" and mm.fails > 0) or (expect == "phase" and mm.phase > 0):
                    break
        detected = mm.fails > 0 if expect == "fail" else (mm.phase > 0 and mm.fails == 0)
        status = "NOT EXERCISED" if counter["applied"] == 0 else ("DETECTED" if detected else "SURVIVED")
        results.append(dict(name=name, description=desc, applied=counter["applied"],
                            circuits_used=used, status=status))
        print(f"  {name:<28}{status:<15}bug applied {counter['applied']:>5}x, "
              f"needed {used} circuit(s)  - {desc}")
    return results


def _noise_model():
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx", "cz", "ecr"])
    return nm


def e2e_check(opt, items, cfg, key_fn):
    try:
        import qiskit_aer
        from qiskit.transpiler import CouplingMap
    except ImportError:
        return dict(status="skipped", reason="qiskit_aer not installed", rows=[])
    nm = _noise_model()
    rows = []
    for it in items:
        c = it["circuit"]
        if c.num_qubits < 2 or c.num_qubits > cfg.e2e_max_qubits or len(c.data) > cfg.e2e_max_gates:
            continue
        try:
            t = time.perf_counter()
            res = opt.optimize_circuit(c, nm, cfg.f1q, cfg.f2q, max_evaluations=cfg.e2e_evaluations,
                                       beam_width=cfg.beam_width,
                                       coupling_map=CouplingMap.from_line(c.num_qubits), seed=cfg.seed)
            chk = Checker(c.num_qubits, cfg.max_unitary_qubits, cfg.states, cfg.seed)
            err, _ = chk.dist(chk.sig(c), chk.sig(res["best_circuit"]))
            rows.append(dict(name=it["name"], err=err, ok=bool(err <= cfg.tol),
                             changed=key_fn(res["best_circuit"]) != key_fn(c),
                             seconds=time.perf_counter() - t))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            rows.append(dict(name=it["name"], err=None, ok=False, changed=False,
                             note=f"{type(exc).__name__}: {str(exc)[:150]}{_crash_where(exc)}"))
    return dict(status="run", rows=rows)


def optimizer_digest(opt, cfg, key_fn):
    h = hashlib.sha256()
    for c in (random_circuit(5, 8, max_operands=3, seed=2), hea(6)):
        for cand, _ in opt.collect_circuit_variants_beam(c, 20, cfg.f1q, cfg.f2q, cfg.beam_width):
            h.update(key_fn(cand).encode())
    return h.hexdigest()


def pipeline_digest(opt, cfg, key_fn):
    from qiskit.transpiler import CouplingMap
    res = opt.optimize_circuit(staggered(4, 12, 11), _noise_model(), cfg.f1q, cfg.f2q, max_evaluations=12,
                               beam_width=4, coupling_map=CouplingMap.from_line(4), seed=7)
    return _digest(key_fn(res["best_circuit"]) + f"{res['best_true_fidelity']:.10f}"
                   + f"{res['original_true_fidelity']:.10f}")


def reproducible(cfg, child_flag):
    digests = []
    for hash_seed in ("1", "2"):
        cmd = [sys.executable, str(Path(__file__).resolve()), child_flag, "--src", str(cfg.src),
               "--optimizer", cfg.optimizer, "--beam-width", str(cfg.beam_width),
               "--f1q", str(cfg.f1q), "--f2q", str(cfg.f2q)]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(),
                             env=dict(os.environ, PYTHONHASHSEED=hash_seed))
        lines = out.stdout.strip().splitlines()
        digests.append(lines[-1] if out.returncode == 0 and lines else "ERROR: " + out.stderr.strip()[-300:])
    return digests[0] == digests[1] and not digests[0].startswith("ERROR"), digests


def pipeline_status(cfg):
    try:
        import qiskit_aer
    except ImportError:
        return "skipped", "qiskit_aer not installed"
    ok, d = reproducible(cfg, "--child-pipeline")
    if d[0].startswith("ERROR"):
        return "error", d[0]
    return ("pass" if ok else "fail"), f"{d[0][:12]} vs {d[1][:12]}"


def bench_status(r, cfg):
    if r["fails"]:
        return "FAIL"
    if r["transformed"] == 0:
        return "NO MOVE APPLIES"
    if r["transformed"] >= cfg.min_transformations:
        return "ok"
    return "ok (search exhausted)" if r["exhausted"] else "TOO FEW"


def move_totals_from(per_move):
    tot = Counter()
    for label, b in per_move.items():
        if label in ("beam", "cumulative", "unknown"):
            continue
        tot[label.split("+")[0]] += b["tests"]
        for j in REQUIRED_JUMPS:
            if j in label:
                tot[j] += b["tests"]
    return tot


def move_totals(m):
    return move_totals_from(m.per_move)


def print_report(cfg, m, bench_rows):
    if bench_rows:
        print("\nBENCHMARK RESULTS  (transformed = distinct transformed circuits verified against the original)")
        hdr = (f"  {'circuit':<52}{'n':>3}{'gates':>6}{'check':>8}{'transformed':>12}{'tests':>7}"
               f"{'FAIL':>6}{'max err':>10}  status")
        print(hdr + "\n  " + "-" * (len(hdr) - 2))
        for r in bench_rows:
            print(f"  {r['name'][:52]:<52}{r['n']:>3}{r['gates']:>6}{r['mode']:>8}{r['transformed']:>12}"
                  f"{r['tests']:>7}{r['fails']:>6}{r['max_err']:>10.1e}  {bench_status(r, cfg)}")

    print("\nMOVE COVERAGE (step-level: every sampled successor vs its parent)")
    hdr = f"  {'move kind':<24}{'comparisons':>12}{'FAIL':>6}{'phase':>7}{'max err':>10}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for label in sorted(m.per_move):
        b = m.per_move[label]
        print(f"  {label:<24}{b['tests']:>12}{b['fails']:>6}{b['phase']:>7}{b['max_err']:>10.1e}")

    print("\nCIRCUIT VARIETY")
    print("  qubits : " + "  ".join(f"{n}q:{m.per_size[n]['circuits']}" for n in sorted(m.per_size)))
    print(f"  source : benchmark {len(bench_rows)}  generated "
          + "  ".join(f"{f}:{m.per_family[f]['circuits']}" for f in FAMILIES))
    gates = [c["gates"] for c in m.per_circuit] or [0]
    print(f"  gates per circuit: min {min(gates)}, median {int(np.median(gates))}, max {max(gates)}")
    print(f"  distinct transformed circuits verified: {len(m.distinct)}   "
          f"(total comparisons {m.tests}, moves that returned an unchanged circuit: {m.noops})")
    print(f"  time: optimizer {m.gen_s:.0f}s, checker {m.chk_s:.0f}s")


def build_checklist(cfg, m, bench_items, bench_rows, skipped, probe_ok, mutation, det, pipe, e2e):
    checks = []

    def add(name, ok, detail=""):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    n_units = len(m.per_circuit)
    too_few = [r["name"] for r in bench_rows if bench_status(r, cfg) == "TOO FEW"]
    untestable = [r["name"] for r in bench_rows if bench_status(r, cfg) == "NO MOVE APPLIES"]
    by_design = ("no gates", "mid-circuit", "contains classical", "duplicate", "cannot build")
    problem_skips = [f"{n}: {why}" for n, why in skipped
                     if not why.startswith(by_design) and " > --bench-max" not in why]
    testable_share = (len(bench_rows) - len(untestable)) / len(bench_rows) if bench_rows else 0.0
    tot = move_totals(m)

    add("synthesis works (2q and 3q)", probe_ok)
    add("benchmark circuits found", len(bench_items) > 0, f"{len(bench_items)} usable")
    add("every benchmark circuit that could be built was loaded and checked",
        not problem_skips and bool(bench_items),
        (f"{len(problem_skips)} problems, e.g. {problem_skips[0][:120]}" if problem_skips else
         f"{len(skipped)} excluded by design (empty, too large, duplicate, ...) - listed above"))
    add(f">= {100 * cfg.min_testable:.0f}% of benchmark circuits admit at least one transformation",
        testable_share >= cfg.min_testable,
        f"{len(bench_rows) - len(untestable)}/{len(bench_rows)}; no move applies to {len(untestable)}")
    add(f"every testable benchmark circuit got >= {cfg.min_transformations} distinct transformed circuits "
        f"(or its search ended earlier)", bool(bench_rows) and not too_few,
        ("too few: " + ", ".join(too_few[:5])) if too_few else "")
    add("no failures", m.fails == 0 and m.failed_circuits == 0,
        f"{m.fails} failing comparisons in {m.failed_circuits} circuits")
    add(f">= {cfg.min_circuits} independent circuits in total", n_units >= cfg.min_circuits, f"{n_units}")
    if cfg.circuits > 0:
        sizes = range(cfg.min_qubits, cfg.max_qubits + 1)
        add(f"every size {cfg.min_qubits}..{cfg.max_qubits} qubits has >= {cfg.min_per_size} circuits",
            all(m.per_size[n]["circuits"] >= cfg.min_per_size for n in sizes),
            "  ".join(f"{n}q:{m.per_size[n]['circuits']}" for n in sizes))
        add(f"every generated family has >= {cfg.min_per_family} circuits",
            all(m.per_family[f]["circuits"] >= cfg.min_per_family for f in FAMILIES),
            "  ".join(f"{f}:{m.per_family[f]['circuits']}" for f in FAMILIES))
    for kind in REQUIRED_MOVES + REQUIRED_JUMPS:
        add(f"move '{kind}' exercised >= {cfg.min_per_move} times", tot[kind] >= cfg.min_per_move, f"{tot[kind]}")
    if not cfg.no_mutation:
        detected = sum(r["status"] == "DETECTED" for r in mutation)
        survived = [r["name"] for r in mutation if r["status"] == "SURVIVED"]
        not_ex = [r["name"] for r in mutation if r["status"] == "NOT EXERCISED"]
        add("every injected bug detected", detected + len(survived) > 0 and not survived,
            f"{detected}/{detected + len(survived)} detected"
            + (f", SURVIVED: {', '.join(survived)}" if survived else "")
            + (f", not exercised: {', '.join(not_ex)}" if not_ex else ""))
    if not cfg.no_determinism:
        add("search reproducible across processes", det[0], f"{det[1][0][:12]} vs {det[1][1][:12]}")
        add("pipeline reproducible (if Aer available)", pipe[0] in ("pass", "skipped"), f"{pipe[0]}: {pipe[1]}")
    if not cfg.no_e2e:
        if e2e["status"] == "skipped":
            add("end to end (skipped)", True, e2e["reason"])
        else:
            ok_n = sum(r["ok"] for r in e2e["rows"])
            add("best_circuit equivalent to the input in every end-to-end run",
                bool(e2e["rows"]) and ok_n == len(e2e["rows"]), f"{ok_n}/{len(e2e['rows'])}")
    return checks


def build_claim(cfg, m, bench_rows, skipped, gen_items, ub, mutation, det_ok, e2e):
    parts = []
    untestable = [r for r in bench_rows if r["transformed"] == 0]
    testable = [r for r in bench_rows if r["transformed"] > 0]
    n_units = len(m.per_circuit)
    tot = move_totals(m)
    if testable:
        ns, gs = [r["n"] for r in testable], [r["gates"] for r in testable]
        all_ok = not any(r["fails"] for r in testable)
        parts.append(
            f"Of {len(bench_rows)} benchmark circuits, {len(testable)} admitted at least one "
            f"transformation and were tested"
            + (f" ({len(untestable)} had no applicable move and are not evidence)" if untestable else "")
            + (f"; {len(skipped)} more were excluded (empty, too large or duplicate)" if skipped else "")
            + f". The tested circuits have {min(ns)}-{max(ns)} qubits and {min(gs)}-{max(gs)} gates; "
              f"{sum(r['transformed'] for r in testable)} distinct transformed circuits were generated "
              f"(at least {min(r['transformed'] for r in testable)} per circuit, or all that the search could "
              f"reach) and each was compared with its original: full unitary up to {cfg.max_unitary_qubits} "
              f"qubits, random states above. "
            + ("Every one matched up to global phase " if all_ok else "Not every one matched: see failures. ")
            + f"(largest deviation {max(r['max_err'] for r in testable):.1e}, tolerance {cfg.tol:.0e}).")
    if gen_items:
        gsz = sorted({r["n"] for r in m.per_circuit if r["family"] != "benchmark"})
        parts.append(
            f"In addition {len(gen_items)} random circuits ({', '.join(FAMILIES)}; {gsz[0]}-{gsz[-1]} qubits) "
            f"were tested. Over all {n_units} independent circuits, {m.failed_circuits} failed, so the "
            f"per-circuit failure probability is below {100 * ub:.1f}% at 95% confidence (Clopper-Pearson).")
    detected = sum(r["status"] == "DETECTED" for r in mutation)
    scored = detected + sum(r["status"] == "SURVIVED" for r in mutation)
    parts.append(
        f"{len(m.distinct)} distinct transformed circuits and {m.tests} comparisons in total. Each move kind "
        f"({', '.join(REQUIRED_MOVES)}, and blocks with gates reordered in either direction) was exercised at "
        f"least {min(tot[k] for k in REQUIRED_MOVES + REQUIRED_JUMPS)} times."
        + (f" The checker detected {detected} of {scored} deliberately injected bugs." if scored else "")
        + (" Results are reproducible with a fixed seed across processes." if det_ok else "")
        + (f" optimize_circuit's best_circuit was equivalent to its input in "
           f"{sum(r['ok'] for r in e2e['rows'])} of {len(e2e['rows'])} end-to-end runs."
           if e2e["status"] == "run" and e2e["rows"] else ""))
    return " ".join(parts)


def save_results(run_dir, cfg, m, bench_rows, bench_where, skipped, mutation, e2e, checks, all_ok,
                 det, pipe, claim, ub, runtime):
    fields = ["name", "family", "n", "gates", "mode", "transformed", "beam_variants", "exhausted",
              "tests", "fails", "max_err"]
    for fname, rows in (("per_circuit.csv", m.per_circuit), ("benchmarks.csv", bench_rows)):
        with open(run_dir / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    with open(run_dir / "per_move.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["move", "comparisons", "fails", "phase_only", "max_err"])
        for label in sorted(m.per_move):
            b = m.per_move[label]
            w.writerow([label, b["tests"], b["fails"], b["phase"], b["max_err"]])
    (run_dir / "claim.txt").write_text(claim + "\n", encoding="utf-8")
    report = dict(
        config=vars(cfg),
        environment=dict(qiskit=qiskit.__version__, numpy=np.__version__,
                         python=platform.python_version(), platform=platform.platform()),
        benchmark=dict(where=bench_where, skipped=skipped, rows=bench_rows),
        totals=dict(circuits=len(m.per_circuit), comparisons=m.tests, failures=m.fails,
                    failed_circuits=m.failed_circuits, phase_only=m.phase, noops=m.noops,
                    distinct_transformed=len(m.distinct), max_err=m.max_err, failure_upper_95=ub,
                    optimizer_s=m.gen_s, checker_s=m.chk_s, runtime_s=runtime),
        move_totals=dict(move_totals(m)), per_move=m.per_move, per_size=m.per_size,
        per_family=m.per_family, mutation=mutation, end_to_end=e2e, checklist=checks,
        supports_claim=all_ok, determinism=dict(search=det[0], pipeline=list(pipe)), failures=m.failures)
    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def parse_seeds(text):
    text = str(text).strip()
    if "," in text:
        return sorted({int(x) for x in text.split(",") if x.strip()})
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return list(range(int(text)))


def project_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "src").is_dir() and (p / "results").is_dir():
            return p
    return here.parent


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = project_root()
    add = ap.add_argument

    add("--src", default=str(root / "src"), help="folder with the optimizer and circuits.py")
    add("--optimizer", default="optimization_search", help="optimizer module name inside --src")
    add("--circuits-module", default="circuits", help="benchmark module name inside --src")
    add("--results-dir", default=str(root / "results" / "equivalence_checker"))
    add("--exp", default=EXP_NAME, help="subfolder of --results-dir that receives this experiment")
    add("--mode", default=MODE, choices=sorted(PRESETS))

    add("--seeds", default=SEEDS,
        help="N for seeds 0..N-1, or a range like 0-49, or a list like 0,1,5; finished seeds are skipped")
    add("--jobs", type=int, default=JOBS, help="seeds to run in parallel (separate processes)")
    add("--rerun", action="store_true", help="delete and recompute the requested seeds")
    add("--aggregate-only", action="store_true", help="only rebuild the global results from finished seeds")

    add("--circuits", type=int, default=150, help="number of generated circuits per seed (0 = none)")
    add("--min-qubits", type=int, default=3)
    add("--max-qubits", type=int, default=8)

    add("--only", default=None, help="comma list: keep only benchmarks whose name contains one of these")
    add("--mqt-sizes", default="3,4,5,6,8,10")
    add("--mqt-forms", default="alg,native")
    add("--bench-variants", type=int, default=80)
    add("--bench-walks", type=int, default=4)
    add("--bench-depth", type=int, default=6)
    add("--bench-max-qubits", type=int, default=16)
    add("--bench-max-gates", type=int, default=800)

    add("--variants", type=int, default=30)
    add("--beam-width", type=int, default=10)
    add("--walks", type=int, default=3)
    add("--depth", type=int, default=6)
    add("--succ-cap", type=int, default=10)
    add("--tol", type=float, default=1e-7)
    add("--strict-phase", action="store_true")
    add("--max-unitary-qubits", type=int, default=6)
    add("--states", type=int, default=3)
    add("--f1q", type=float, default=0.999)
    add("--f2q", type=float, default=0.99)
    add("--max-dumps", type=int, default=20)

    add("--no-e2e", action="store_true")
    add("--e2e-max-qubits", type=int, default=6)
    add("--e2e-max-gates", type=int, default=150)
    add("--e2e-evaluations", type=int, default=15)
    add("--e2e-generated", type=int, default=5)

    add("--no-mutation", action="store_true")
    add("--mutation-circuits", type=int, default=16)
    add("--mutation-max-qubits", type=int, default=6)
    add("--mutation-max-gates", type=int, default=80)
    add("--no-determinism", action="store_true")

    add("--min-circuits", type=int, default=150)
    add("--min-per-size", type=int, default=10)
    add("--min-per-family", type=int, default=25)
    add("--min-per-move", type=int, default=50)
    add("--min-testable", type=float, default=0.8)
    add("--min-transformations", type=int, default=30)
    add("--child-pipeline", action="store_true", help=argparse.SUPPRESS)
    add("--print-digest", action="store_true", help=argparse.SUPPRESS)
    chosen, _ = ap.parse_known_args(argv)
    ap.set_defaults(**PRESETS[chosen.mode])
    return ap.parse_args(argv)


def seed_mismatch(sdir, cfg):
    rp = sdir / "report.json"
    if not rp.exists():
        return "no report.json"
    try:
        old = json.loads(rp.read_text(encoding="utf-8"))["config"]
    except Exception:
        return "unreadable report.json"
    diff = [f"{k}: saved {old.get(k)!r}, now {v!r}" for k, v in vars(cfg).items()
            if k not in SETTING_EXCLUDE and old.get(k) != v]
    return ("settings differ -> " + "; ".join(diff)) if diff else None


def run_seed(cfg, seed, root, quiet):
    scfg = argparse.Namespace(**vars(cfg))
    scfg.seed = seed
    scfg.no_determinism = True
    sys.path.insert(0, str(Path(scfg.src).resolve()))
    opt = importlib.import_module(scfg.optimizer)
    bench_mod = importlib.import_module(scfg.circuits_module)
    key_fn = getattr(opt, "_circuit_key", None) or qasm3.dumps
    scfg.optimizer_file = opt.__file__

    sdir = root / f"seed{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    ckpt = sdir / "checkpoint.json"
    resuming = ckpt.exists()
    if resuming:
        done, m, rng_state = load_checkpoint(ckpt)
    else:
        done, m, rng_state = set(), Metrics(), None

    log = open(sdir / "log.txt", "a" if resuming else "w", encoding="utf-8", buffering=1)
    old = sys.stdout
    sys.stdout = log if quiet else Tee(old, log)
    try:
        print(f"\n===== seed {seed} =====")
        if resuming:
            print(f"RESUMING from checkpoint: {len(done)} circuits already done, {ckpt}")
        return run(scfg, opt, bench_mod, key_fn, sdir, ckpt, done, m, rng_state)
    finally:
        sys.stdout = old
        log.close()


def run_all(cfg, seeds, root):
    pending, blocked = [], []
    if cfg.mode == "full" and not cfg.rerun:
        absent = [s for s in seeds if s in REUSE_ONLY and not (root / f"seed{s}" / "report.json").exists()]
        if absent:
            print(f"seed(s) {absent} must be reused, never recomputed, but no report.json was found in:")
            for s in absent:
                print(f"  {root / f'seed{s}'}")
            print("copy your existing results for these seeds into those folders, then run again")
            return False
    for s in seeds:
        sdir = root / f"seed{s}"
        if cfg.rerun and sdir.exists():
            shutil.rmtree(sdir)
        why = seed_mismatch(sdir, cfg)
        if why is None:
            continue
        if (sdir / "report.json").exists():
            blocked.append((s, why))
        else:
            pending.append(s)
    if blocked:
        print("these seeds already have results that do not match the current settings; nothing was run "
              "and nothing was overwritten:")
        for s, why in blocked:
            print(f"  seed {s}: {why}")
        print("fix the settings to match, or pass --rerun to recompute these seeds")
        return False
    print(f"seeds requested: {len(seeds)}, already complete (reused): {len(seeds) - len(pending)}, "
          f"to run: {len(pending)}  ->  {root}")

    det_path = root / "determinism.json"
    if not cfg.no_determinism and (cfg.rerun or not det_path.exists()):
        print("\nREPRODUCIBILITY (seed-independent, checked once)")
        det = reproducible(cfg, "--print-digest")
        print(f"  search output identical across processes (PYTHONHASHSEED 1 vs 2): "
              f"{'yes' if det[0] else 'NO'}  {det[1][0][:12]} vs {det[1][1][:12]}")
        pipe = pipeline_status(cfg)
        print(f"  optimize_circuit pipeline (transpile + Aer, seed=7): {pipe[0]}  {pipe[1]}")
        det_path.write_text(json.dumps(dict(search=det[0], digests=det[1], pipeline=list(pipe)), indent=2),
                            encoding="utf-8")

    t0 = time.perf_counter()
    if cfg.jobs <= 1:
        for i, s in enumerate(pending, 1):
            code = run_seed(cfg, s, root, False)
            print(f"\nseed {s} finished (exit {code})  [{i}/{len(pending)}, {time.perf_counter() - t0:.0f}s]")
    elif pending:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=cfg.jobs) as ex:
            futs = {ex.submit(run_seed, cfg, s, root, True): s for s in pending}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    code = f.result()
                except Exception as exc:
                    code = f"crashed: {type(exc).__name__}: {str(exc)[:150]}"
                print(f"seed {futs[f]} finished (exit {code})  [{i}/{len(pending)}, "
                      f"{time.perf_counter() - t0:.0f}s]")
    return True


def load_seed_reports(root, seeds):
    out = {}
    for s in seeds:
        sdir = root / f"seed{s}"
        rp = sdir / "report.json"
        if not rp.exists():
            continue
        rep = json.loads(rp.read_text(encoding="utf-8"))
        dp = sdir / "distinct.txt"
        rep["_distinct"] = set(dp.read_text(encoding="utf-8").split()) if dp.exists() else None
        out[s] = rep
    return out


def _span(seeds):
    seeds = sorted(seeds)
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(str(s) for s in seeds)


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def aggregate(cfg, seeds, root):
    reps = load_seed_reports(root, seeds)
    gdir = root / "global"
    gdir.mkdir(parents=True, exist_ok=True)
    log = open(gdir / "summary.txt", "w", encoding="utf-8")
    old, sys.stdout = sys.stdout, Tee(sys.stdout, log)
    try:
        return _aggregate(cfg, seeds, root, reps, gdir)
    finally:
        sys.stdout = old
        log.close()


def _aggregate(cfg, seeds, root, reps, gdir):
    missing = [s for s in seeds if s not in reps]
    if not reps:
        print(f"no finished seeds found in {root}")
        return 2
    det_path = root / "determinism.json"
    det = json.loads(det_path.read_text(encoding="utf-8")) if det_path.exists() else None

    seed_rows, failures = [], []
    per_move = defaultdict(lambda: dict(tests=0, fails=0, phase=0, max_err=0.0))
    per_family = defaultdict(lambda: dict(circuits=0, tests=0, fails=0))
    per_size = defaultdict(lambda: dict(circuits=0, tests=0, fails=0))
    bench, mutation, mut_desc = {}, defaultdict(Counter), {}
    distinct, distinct_sum, distinct_complete = set(), 0, True
    e2e_runs = e2e_ok = e2e_changed = e2e_n = 0
    gen_total = gen_failed = comparisons = fails = phase_only = noops = 0
    max_err = runtime = optimizer_s = checker_s = 0.0

    for s in sorted(reps):
        rep = reps[s]
        t = rep["totals"]
        rows = rep["benchmark"]["rows"]
        gen_total += t["circuits"] - len(rows)
        gen_failed += t["failed_circuits"] - sum(1 for r in rows if r["fails"])
        comparisons += t["comparisons"]
        fails += t["failures"]
        phase_only += t["phase_only"]
        noops += t["noops"]
        max_err = max(max_err, t["max_err"])
        runtime += t["runtime_s"]
        optimizer_s += t["optimizer_s"]
        checker_s += t["checker_s"]
        distinct_sum += t["distinct_transformed"]
        if rep["_distinct"] is None:
            distinct_complete = False
        else:
            distinct |= rep["_distinct"]

        for label, b in rep["per_move"].items():
            g = per_move[label]
            g["tests"] += b["tests"]
            g["fails"] += b["fails"]
            g["phase"] += b["phase"]
            g["max_err"] = max(g["max_err"], b["max_err"])
        for label, b in rep["per_family"].items():
            g = per_family[label]
            for k in ("circuits", "tests", "fails"):
                g[k] += b[k]
        for label, b in rep["per_size"].items():
            g = per_size[int(label)]
            for k in ("circuits", "tests", "fails"):
                g[k] += b[k]

        for r in rows:
            g = bench.setdefault(r["name"], dict(
                name=r["name"], n=r["n"], gates=r["gates"], mode=r["mode"], seeds=0,
                min_transformed=r["transformed"], max_transformed=r["transformed"], tests=0, fails=0,
                max_err=0.0, failed_seeds=[]))
            g["seeds"] += 1
            g["min_transformed"] = min(g["min_transformed"], r["transformed"])
            g["max_transformed"] = max(g["max_transformed"], r["transformed"])
            g["tests"] += r["tests"]
            g["fails"] += r["fails"]
            g["max_err"] = max(g["max_err"], r["max_err"])
            if r["fails"]:
                g["failed_seeds"].append(s)

        for mu in rep["mutation"]:
            mutation[mu["name"]][mu["status"]] += 1
            mut_desc[mu["name"]] = mu["description"]

        e2e = rep["end_to_end"]
        if e2e["status"] == "run":
            e2e_runs += 1
            e2e_n += len(e2e["rows"])
            e2e_ok += sum(r["ok"] for r in e2e["rows"])
            e2e_changed += sum(r["changed"] for r in e2e["rows"])

        for f in rep["failures"]:
            failures.append(dict(seed=s, **f))

        failed_checks = [c["name"] for c in rep["checklist"] if not c["ok"]]
        seed_rows.append(dict(
            seed=s, circuits=t["circuits"], comparisons=t["comparisons"], failures=t["failures"],
            failed_circuits=t["failed_circuits"], phase_only=t["phase_only"],
            distinct_transformed=t["distinct_transformed"], max_err=t["max_err"],
            runtime_s=round(t["runtime_s"], 1), supports_claim=rep["supports_claim"],
            failed_checks="; ".join(failed_checks)))

    bench_failed = sum(1 for g in bench.values() if g["fails"])
    n_indep = gen_total + len(bench)
    k_indep = gen_failed + bench_failed
    ub = cp_upper(k_indep, n_indep)
    tot = move_totals_from(per_move)
    seeds_txt = _span(list(reps))
    n_distinct = len(distinct) if distinct_complete else distinct_sum
    distinct_note = "union over seeds" if distinct_complete else "sum over seeds, overlaps possible"
    testable = [g for g in bench.values() if g["max_transformed"] > 0]

    print(f"GLOBAL RESULTS  {len(reps)} seeds ({seeds_txt})  {root}")
    if missing:
        print(f"  MISSING seeds (no report.json): {_span(missing)}")

    print("\nPER SEED")
    hdr = (f"  {'seed':>5}{'circuits':>9}{'comparisons':>12}{'FAIL':>6}{'phase':>7}{'distinct':>10}"
           f"{'max err':>10}{'time s':>8}  status")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for r in seed_rows:
        status = "ok" if r["supports_claim"] else "CHECK: " + r["failed_checks"][:80]
        print(f"  {r['seed']:>5}{r['circuits']:>9}{r['comparisons']:>12}{r['failures']:>6}{r['phase_only']:>7}"
              f"{r['distinct_transformed']:>10}{r['max_err']:>10.1e}{r['runtime_s']:>8.0f}  {status}")

    print("\nBENCHMARKS ACROSS SEEDS  (same circuits every seed; seeds change parameters, check states, walks)")
    hdr = (f"  {'circuit':<52}{'n':>3}{'gates':>6}{'seeds':>6}{'min transf.':>12}{'tests':>8}{'FAIL':>6}"
           f"{'max err':>10}  status")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for g in sorted(bench.values(), key=lambda x: x["name"]):
        status = ("FAIL in seeds " + _span(g["failed_seeds"])) if g["fails"] else (
            "NO MOVE APPLIES" if g["max_transformed"] == 0 else "ok")
        print(f"  {g['name'][:52]:<52}{g['n']:>3}{g['gates']:>6}{g['seeds']:>6}{g['min_transformed']:>12}"
              f"{g['tests']:>8}{g['fails']:>6}{g['max_err']:>10.1e}  {status}")

    print("\nMOVE COVERAGE (all seeds)")
    hdr = f"  {'move kind':<24}{'comparisons':>12}{'FAIL':>6}{'phase':>7}{'max err':>10}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for label in sorted(per_move):
        b = per_move[label]
        print(f"  {label:<24}{b['tests']:>12}{b['fails']:>6}{b['phase']:>7}{b['max_err']:>10.1e}")

    print("\nCIRCUIT VARIETY (all seeds)")
    print("  qubits : " + "  ".join(f"{n}q:{per_size[n]['circuits']}" for n in sorted(per_size)))
    print("  generated: " + "  ".join(f"{f}:{per_family[f]['circuits']}" for f in FAMILIES)
          + f"   benchmark runs: {per_family['benchmark']['circuits']}")
    print(f"  distinct transformed circuits verified: {n_distinct} ({distinct_note}); "
          f"comparisons {comparisons}; unchanged-successor moves {noops}")
    print(f"  time: {runtime:.0f}s of seed runs (optimizer {optimizer_s:.0f}s, checker {checker_s:.0f}s)")

    print("\nFAILURE RATE  independent circuits = every generated circuit + each benchmark circuit once")
    print(f"  {k_indep} of {n_indep} failed -> per-circuit failure probability < {100 * ub:.3f}% "
          f"(95% confidence, Clopper-Pearson)")

    if mutation:
        print("\nMUTATION TESTING ACROSS SEEDS")
        for name in mutation:
            c = mutation[name]
            print(f"  {name:<28}detected {c['DETECTED']:>3}  survived {c['SURVIVED']:>3}  "
                  f"not exercised {c['NOT EXERCISED']:>3}  - {mut_desc[name]}")

    if e2e_runs:
        print(f"\nEND TO END  {e2e_ok}/{e2e_n} runs equivalent over {e2e_runs} seeds, "
              f"{e2e_changed} of them actually changed by the optimizer")

    checks = []

    def add(name, ok, detail=""):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    bad_seeds = [r["seed"] for r in seed_rows if not r["supports_claim"]]
    add(f"all {len(seeds)} requested seeds finished", not missing, f"missing: {_span(missing)}" if missing else "")
    add("every seed passes its own checklist", not bad_seeds, f"failing seeds: {_span(bad_seeds)}" if bad_seeds else "")
    add("no failures in any seed", fails == 0 and k_indep == 0,
        f"{fails} failing comparisons, {k_indep} of {n_indep} independent circuits failed")
    if mutation:
        survived = [n for n, c in mutation.items() if c["SURVIVED"]]
        add("every injected bug detected in every seed that exercised it", not survived,
            f"SURVIVED: {', '.join(survived)}" if survived else
            "  ".join(f"{n}:{c['DETECTED']}" for n, c in mutation.items()))
    if det is not None:
        add("search reproducible across processes", det["search"], "")
        add("pipeline reproducible (if Aer available)", det["pipeline"][0] in ("pass", "skipped"),
            f"{det['pipeline'][0]}")
    elif not cfg.no_determinism:
        add("reproducibility checked", False, "determinism.json missing")
    if e2e_runs:
        add("best_circuit equivalent to the input in every end-to-end run", e2e_ok == e2e_n, f"{e2e_ok}/{e2e_n}")

    print("\nGLOBAL CHECKLIST")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}" + (f"   ({c['detail']})" if c["detail"] else ""))
    all_ok = all(c["ok"] for c in checks)

    parts = [f"The equivalence suite was run with {len(reps)} independent seeds ({seeds_txt}); every seed "
             f"generates its own random circuits, check states, search walks and benchmark parameter values."]
    if testable:
        ns, gs = [g["n"] for g in testable], [g["gates"] for g in testable]
        parts.append(f"{len(testable)} of {len(bench)} benchmark circuits ({min(ns)}-{max(ns)} qubits, "
                     f"{min(gs)}-{max(gs)} gates) admitted at least one transformation.")
    if gen_total:
        parts.append(f"{gen_total} random circuits ({', '.join(FAMILIES)}) were generated in total, "
                     f"{gen_total // len(reps)} per seed.")
    parts.append(f"{n_distinct} distinct transformed circuits ({distinct_note}) and {comparisons} comparisons "
                 f"with their originals or parents were checked (full unitary up to {cfg.max_unitary_qubits} qubits, "
                 f"random states above); {fails} failures, largest deviation {max_err:.1e}, "
                 f"tolerance {cfg.tol:.0e}.")
    parts.append(f"Counting every generated circuit and each benchmark circuit once, {k_indep} of {n_indep} "
                 f"independent circuits failed, so the per-circuit failure probability is below "
                 f"{100 * ub:.3f}% at 95% confidence (Clopper-Pearson).")
    if tot:
        parts.append(f"Each move kind ({', '.join(REQUIRED_MOVES)}, and reordering in either direction) was "
                     f"exercised at least {min(tot[k] for k in REQUIRED_MOVES + REQUIRED_JUMPS)} times in total.")
    if mutation:
        d = sum(c["DETECTED"] for c in mutation.values())
        sc = d + sum(c["SURVIVED"] for c in mutation.values())
        parts.append(f"The checker detected {d} of {sc} deliberately injected bugs across all seeds.")
    if det is not None and det["search"]:
        parts.append("Search output is identical across processes.")
    if e2e_runs:
        parts.append(f"optimize_circuit's best_circuit was equivalent to its input in {e2e_ok} of {e2e_n} "
                     f"end-to-end runs.")
    claim = " ".join(parts)
    print("\n" + ("SUPPORTS THE CLAIM." if all_ok else "DOES NOT YET SUPPORT THE CLAIM - see the FAIL lines above."))
    print("Claim text:\n  " + claim)

    _write_csv(gdir / "seeds.csv", seed_rows,
               ["seed", "circuits", "comparisons", "failures", "failed_circuits", "phase_only",
                "distinct_transformed", "max_err", "runtime_s", "supports_claim", "failed_checks"])
    _write_csv(gdir / "benchmarks.csv",
               [dict(g, failed_seeds=" ".join(str(x) for x in g["failed_seeds"])) for g in bench.values()],
               ["name", "n", "gates", "mode", "seeds", "min_transformed", "max_transformed", "tests",
                "fails", "max_err", "failed_seeds"])
    _write_csv(gdir / "per_move.csv",
               [dict(move=k, comparisons=v["tests"], fails=v["fails"], phase_only=v["phase"], max_err=v["max_err"])
                for k, v in sorted(per_move.items())], ["move", "comparisons", "fails", "phase_only", "max_err"])
    _write_csv(gdir / "per_family.csv", [dict(family=k, **v) for k, v in per_family.items()],
               ["family", "circuits", "tests", "fails"])
    _write_csv(gdir / "per_size.csv", [dict(qubits=k, **v) for k, v in sorted(per_size.items())],
               ["qubits", "circuits", "tests", "fails"])
    _write_csv(gdir / "mutation.csv",
               [dict(name=n, description=mut_desc[n], detected=c["DETECTED"], survived=c["SURVIVED"],
                     not_exercised=c["NOT EXERCISED"]) for n, c in mutation.items()],
               ["name", "description", "detected", "survived", "not_exercised"])
    (gdir / "claim.txt").write_text(claim + "\n", encoding="utf-8")
    report = dict(
        config={k: v for k, v in vars(cfg).items() if k not in ("child_pipeline", "print_digest")},
        seeds=sorted(reps), missing=missing,
        totals=dict(seeds=len(reps), generated_circuits=gen_total, unique_benchmark_circuits=len(bench),
                    independent_circuits=n_indep, independent_failures=k_indep, comparisons=comparisons,
                    failures=fails, phase_only=phase_only, noops=noops, distinct_transformed=n_distinct,
                    distinct_is_union=distinct_complete, max_err=max_err, failure_upper_95=ub,
                    optimizer_s=optimizer_s, checker_s=checker_s, runtime_s=runtime),
        move_totals=dict(tot), per_move=per_move, per_size=per_size, per_family=per_family,
        per_seed=seed_rows, benchmarks=list(bench.values()),
        mutation={n: dict(c) for n, c in mutation.items()},
        end_to_end=dict(seeds_run=e2e_runs, runs=e2e_n, equivalent=e2e_ok, changed=e2e_changed),
        determinism=det, checklist=checks, supports_claim=all_ok, claim=claim, failures=failures[:500])
    with open(gdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nglobal results saved to {gdir}")
    return 0 if all_ok else 1


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.print_digest or cfg.child_pipeline:
        sys.path.insert(0, str(Path(cfg.src).resolve()))
        opt = importlib.import_module(cfg.optimizer)
        key_fn = getattr(opt, "_circuit_key", None) or qasm3.dumps
        print(optimizer_digest(opt, cfg, key_fn) if cfg.print_digest else pipeline_digest(opt, cfg, key_fn))
        return 0

    seeds = parse_seeds(cfg.seeds)
    root = Path(cfg.results_dir) / cfg.exp
    root.mkdir(parents=True, exist_ok=True)
    if not cfg.aggregate_only and not run_all(cfg, seeds, root):
        return 2
    return aggregate(cfg, seeds, root)


def run(cfg, opt, bench_mod, key_fn, run_dir, ckpt, done, m, rng_state):
    t_start = time.perf_counter()
    print(f"qiskit {qiskit.__version__}, seed {cfg.seed}")
    print(f"optimizer: {cfg.optimizer_file}")
    print(f"results -> {run_dir}")
    if not selftest(cfg.tol):
        print("Checker self-test failed - not trusting any results.")
        return 2
    probe_ok = synthesis_probe(opt)

    bench_items, skipped = load_benchmarks(cfg, bench_mod)
    bench_where = f"{cfg.circuits_module} (MQT sizes {cfg.mqt_sizes}; forms {cfg.mqt_forms})" \
                  + (f"; only names containing: {cfg.only}" if cfg.only else "")
    print(f"\nBENCHMARK CIRCUITS  from {bench_where}")
    print(f"  usable: {len(bench_items)}   skipped: {len(skipped)}")
    for name, why in skipped:
        print(f"    skipped {name}: {why}")
    gen_items = build_suite(cfg.circuits, cfg.min_qubits, cfg.max_qubits, cfg.seed) if cfg.circuits > 0 else []
    print(f"GENERATED CIRCUITS  {len(gen_items)}: "
          + ", ".join(f"{f} {sum(1 for s in gen_items if s['family'] == f)}" for f in FAMILIES))

    bcfg = argparse.Namespace(**vars(cfg))
    bcfg.variants, bcfg.walks, bcfg.depth = cfg.bench_variants, cfg.bench_walks, cfg.bench_depth
    work = [(it, bcfg) for it in bench_items] + [(it, cfg) for it in gen_items]
    remaining = [(it, c) for it, c in work if it["name"] not in done]
    if done:
        print(f"  {len(work) - len(remaining)} of {len(work)} circuits already done, {len(remaining)} left")

    rng = np.random.default_rng(cfg.seed)
    if rng_state is not None:
        rng.bit_generator.state = rng_state
    fail_dir = str(run_dir / "failures")
    print("\nRUNNING  (every transformed circuit is compared with its original)")
    for k, (item, c) in enumerate(remaining, len(work) - len(remaining) + 1):
        try:
            check_circuit(item, opt, c, rng, m, key_fn, fail_dir)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            where = _crash_where(exc)
            m.fails += 1
            m.failed_circuits += 1
            m.failures.append(dict(circuit=item["name"], tag="crash", label="-", ref="-", err=None,
                                   note=f"optimizer raised {type(exc).__name__}: {str(exc)[:150]}{where}"))
            print(f"    CRASH {item['name']}: {type(exc).__name__}: {str(exc)[:150]}{where}")
            dump_crash(fail_dir, item, exc)
        else:
            if item["family"] == "benchmark":
                r = m.per_circuit[-1]
                print(f"  [{k:>4}/{len(work)}] {item['name'][:52]:<52} n={r['n']:<2} gates={r['gates']:<4} "
                      f"transformed={r['transformed']:<4} FAIL={r['fails']}")
            elif k % 10 == 0 or k == len(work):
                print(f"  [{k:>4}/{len(work)}] {m.tests} comparisons, {m.fails} failures, "
                      f"{time.perf_counter() - t_start:.0f}s elapsed")
        done.add(item["name"])
        save_checkpoint(ckpt, done, m, rng.bit_generator.state)
    for f in m.failures[:10]:
        print(f"    FAIL {f['circuit']} {f['tag']} {f['label']} vs {f['ref']}: err={f['err']} {f.get('note', '')}")

    bench_rows = [r for r in m.per_circuit if r["family"] == "benchmark"]
    print_report(cfg, m, bench_rows)
    n_units = len(m.per_circuit)
    ub = cp_upper(m.failed_circuits, n_units)
    print(f"\nFAILURE RATE  {m.failed_circuits} of {n_units} independent circuits had a failure "
          f"-> per-circuit failure probability < {100 * ub:.2f}% (95% confidence, Clopper-Pearson)")

    mutation = []
    if not cfg.no_mutation:
        print("\nMUTATION TESTING (inject a bug, the suite must FAIL)")
        mutation = run_mutation_tests(opt, gen_items + bench_items, cfg, key_fn)

    det, pipe = (None, []), ("skipped", "not run")
    if not cfg.no_determinism:
        print("\nREPRODUCIBILITY")
        det = reproducible(cfg, "--print-digest")
        print(f"  search output identical across processes (PYTHONHASHSEED 1 vs 2): "
              f"{'yes' if det[0] else 'NO'}  {det[1][0][:12]} vs {det[1][1][:12]}")
        pipe = pipeline_status(cfg)
        print(f"  optimize_circuit pipeline (transpile + Aer, seed=7): {pipe[0]}  {pipe[1]}")

    e2e = dict(status="skipped", reason="--no-e2e", rows=[])
    if not cfg.no_e2e:
        print("\nEND TO END  optimize_circuit(...).best_circuit vs the original circuit")
        small_gen = [it for it in gen_items if it["circuit"].num_qubits <= cfg.e2e_max_qubits]
        e2e = e2e_check(opt, bench_items + small_gen[: cfg.e2e_generated], cfg, key_fn)
        if e2e["status"] == "skipped":
            print(f"  skipped: {e2e['reason']}")
        else:
            for r in e2e["rows"]:
                print(f"  {r['name'][:52]:<52} {'ok  ' if r['ok'] else 'FAIL'} "
                      + (f"err={r['err']:.1e}  best_circuit {'differs from' if r['changed'] else 'equals'} input"
                         if r["err"] is not None else r.get("note", "")))
            print(f"  {sum(r['ok'] for r in e2e['rows'])}/{len(e2e['rows'])} equivalent, "
                  f"{sum(r['changed'] for r in e2e['rows'])} of them actually changed by the optimizer")

    checks = build_checklist(cfg, m, bench_items, bench_rows, skipped, probe_ok, mutation, det, pipe, e2e)
    print("\nCLAIM CHECKLIST")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}" + (f"   ({c['detail']})" if c["detail"] else ""))
    all_ok = all(c["ok"] for c in checks)
    claim = build_claim(cfg, m, bench_rows, skipped, gen_items, ub, mutation, det[0], e2e)
    print("\n" + ("SUPPORTS THE CLAIM." if all_ok else "DOES NOT YET SUPPORT THE CLAIM - see the FAIL lines above."))
    print("Claim text:\n  " + claim)

    runtime = time.perf_counter() - t_start
    (run_dir / "distinct.txt").write_text("\n".join(sorted(m.distinct)) + "\n", encoding="utf-8")
    save_results(run_dir, cfg, m, bench_rows, bench_where, skipped, mutation, e2e, checks, all_ok,
                 det, pipe, claim, ub, runtime)
    if ckpt.exists():
        ckpt.unlink()
    print(f"\nsaved to {run_dir}  (total {runtime:.0f}s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())