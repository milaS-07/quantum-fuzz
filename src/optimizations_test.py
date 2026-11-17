import argparse
import csv
import json
import math
import os
import platform
import random
import re
import signal
import sys
import time
import traceback
import warnings
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import qiskit
from qiskit import qasm3, transpile

import optimization_search as opt
from circuits import BENCHMARK_REGISTRY, get_benchmarks, get_mqt_circuit
from src.noise.noise_metrics import calculate_fidelity, calculate_js_divergence, calculate_tvd
from src.noise.noise_parameters import get_sherbrooke_noise_model
from src.predictors.esp_fidelity import gate_errors_from_params, predict_fidelity_esp

warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT_NAME = "exp05"
UNROLL_BASIS = ["u", "cx"]
SEED_NAMES = ("params", "transpile", "beam", "simulator", "shots")
SETTING_KEYS = ("max_evaluations", "patience", "top_k", "beam_width", "shots", "baseline_levels",
                "max_qubits", "coupling")

_STATE = {}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_int_list(text):
    text = str(text).strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return sorted({int(x) for x in text.split(",") if x.strip()})


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sherbrooke_coupling_map():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    except ImportError:
        from qiskit.providers.fake_provider import FakeSherbrookeV2 as FakeSherbrooke
    backend = FakeSherbrooke()
    return getattr(backend, "coupling_map", None) or backend.target.build_coupling_map()


def _adjacency(coupling_map):
    adjacency = {q: set() for q in range(coupling_map.size())}
    for a, b in coupling_map.get_edges():
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _edge_count(adjacency, qubits):
    qset = set(qubits)
    return sum(len(adjacency[q] & qset) for q in qset) // 2


def _connected(adjacency, qubits):
    qset = set(qubits)
    start = next(iter(qset))
    seen = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for q in frontier:
            for nb in adjacency[q] & qset:
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return seen == qset


def _greedy_subset(adjacency, n_qubits, start):
    qset = {start}
    while len(qset) < n_qubits:
        boundary = set().union(*(adjacency[q] - qset for q in qset))
        if not boundary:
            return None
        best = max(boundary, key=lambda q: (len(adjacency[q] & qset), len(adjacency[q]), -q))
        qset.add(best)
    return qset


def _improve_subset(adjacency, qubits, max_iters=50):
    qset = set(qubits)
    edges = _edge_count(adjacency, qset)
    for _ in range(max_iters):
        boundary = set().union(*(adjacency[q] - qset for q in qset))
        improved = False
        for out_q in sorted(qset):
            for in_q in sorted(boundary):
                candidate = (qset - {out_q}) | {in_q}
                if not _connected(adjacency, candidate):
                    continue
                c_edges = _edge_count(adjacency, candidate)
                if c_edges > edges:
                    qset, edges, improved = candidate, c_edges, True
                    break
            if improved:
                break
        if not improved:
            break
    return qset


def densest_connected_subset(coupling_map, n_qubits, top_k_seeds=8):
    adjacency = _adjacency(coupling_map)
    found = []
    for start in sorted(adjacency):
        patch = _greedy_subset(adjacency, n_qubits, start)
        if patch is not None:
            found.append((_edge_count(adjacency, patch), sorted(patch), patch))
    if not found:
        raise ValueError(f"no connected component with {n_qubits} qubits")
    found.sort(key=lambda row: (-row[0], row[1]))
    best_edges, best = -1, None
    for _, _, patch in found[:top_k_seeds]:
        refined = _improve_subset(adjacency, patch)
        edges = _edge_count(adjacency, refined)
        if edges > best_edges:
            best_edges, best = edges, refined
    return sorted(best)


def worker_state():
    if not _STATE:
        _STATE["noise_model"] = get_sherbrooke_noise_model()
        _STATE["gate_errors"] = gate_errors_from_params()
        _STATE["maps"] = {}
    return _STATE


def get_coupling(n_qubits, mode):
    if mode == "none":
        return None, None
    state = worker_state()
    if "full_map" not in state:
        state["full_map"] = get_sherbrooke_coupling_map()
    cache = state["maps"]
    if n_qubits not in cache:
        qubits = densest_connected_subset(state["full_map"], n_qubits)
        reduced = state["full_map"].reduce(qubits)
        reduced.make_symmetric()
        cache[n_qubits] = (reduced, qubits)
    return cache[n_qubits]


def derive_seeds(seed, benchmark, n_qubits):
    sequence = np.random.SeedSequence([seed, zlib.crc32(benchmark.encode()), n_qubits])
    return dict(zip(SEED_NAMES, (int(v) for v in sequence.generate_state(len(SEED_NAMES)))))


def features(circuit):
    ops = circuit.count_ops()
    return dict(
        depth=circuit.depth(),
        size=circuit.size(),
        num_qubits=circuit.num_qubits,
        gate_counts={k: int(v) for k, v in ops.items()},
        two_qubit_gate_count=sum(1 for i in circuit.data if i.operation.num_qubits == 2)
    )


def safe_qasm(circuit):
    try:
        return qasm3.dumps(circuit)
    except Exception:
        return None


def _normalize(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    return p / p.sum()


def _counts(p, shots, rng):
    n_bits = int(round(math.log2(len(p))))
    drawn = rng.multinomial(shots, p)
    return {format(i, f"0{n_bits}b"): int(c) for i, c in enumerate(drawn) if c}


def distribution_metrics(p_ideal, p_noisy, shots, shots_seed, tag):
    p = _normalize(p_ideal)
    q = _normalize(p_noisy)
    mid = 0.5 * (p + q)
    mp, mq = p > 0, q > 0
    jsd = (0.5 * float(np.sum(p[mp] * np.log2(p[mp] / mid[mp])))
           + 0.5 * float(np.sum(q[mq] * np.log2(q[mq] / mid[mq]))))
    rng = np.random.default_rng([shots_seed, tag])
    ci = _counts(p, shots, rng)
    cn = _counts(q, shots, rng)
    return dict(
        hellinger_fidelity=float(np.sum(np.sqrt(p * q)) ** 2),
        tvd=0.5 * float(np.sum(np.abs(p - q))),
        js_divergence=jsd,
        shot_fidelity=calculate_fidelity(ci, cn, shots),
        shot_tvd=calculate_tvd(ci, cn, shots),
        shot_js_divergence=calculate_js_divergence(ci, cn, shots)
    )


def _ranks(values):
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def result_path(root, seed, category, benchmark, n_qubits):
    return Path(root) / f"seed{seed}" / slugify(category) / slugify(benchmark) / f"n{n_qubits}.json"


def error_path(path):
    return path.with_name(path.stem + ".error.json")


def build_original(benchmark, n_qubits, seeds, max_qubits):
    random.seed(seeds["params"])
    np.random.seed(seeds["params"])
    raw = get_mqt_circuit(benchmark, num_qubits=n_qubits)
    raw = raw.remove_final_measurements(inplace=False)
    if any(i.operation.name in ("measure", "reset") for i in raw.data):
        raise ValueError("mid-circuit measurement or reset")
    parameters_bound = False
    if len(raw.parameters):
        rng = np.random.default_rng(seeds["params"])
        values = {p: float(rng.uniform(0, 2 * np.pi)) for p in sorted(raw.parameters, key=lambda p: p.name)}
        raw = raw.assign_parameters(values)
        parameters_bound = True
    original = transpile(raw, basis_gates=UNROLL_BASIS, optimization_level=0,
                         seed_transpiler=seeds["transpile"])
    if original.num_qubits > max_qubits:
        raise ValueError(f"{original.num_qubits} qubits > max_qubits {max_qubits}")
    if original.size() == 0:
        raise ValueError("circuit has no gates")
    return original, parameters_bound


def execute(benchmark, category, n_qubits, seed, cfg):
    state = worker_state()
    noise_model, gate_errors = state["noise_model"], state["gate_errors"]
    seeds = derive_seeds(seed, benchmark, n_qubits)
    t_total = time.perf_counter()

    original, parameters_bound = build_original(benchmark, n_qubits, seeds, cfg["max_qubits"])
    coupling, physical_qubits = get_coupling(original.num_qubits, cfg["coupling"])
    max_evaluations = cfg["max_evaluations"]

    result = opt.optimize_circuit(
        original,
        noise_model,
        gate_errors,
        max_evaluations=max_evaluations,
        beam_width=cfg["beam_width"],
        top_k=cfg["top_k"],
        patience=cfg["patience"],
        coupling_map=coupling,
        seed=seeds["transpile"],
        beam_seed=seeds["beam"],
        sim_seed=seeds["simulator"]
    )

    candidates = result["candidates"]
    verified_rows = []
    for v in result["verified"]:
        cand = candidates[v["index"]]
        row = dict(
            index=v["index"],
            depth=cand["depth"],
            parent=cand["parent"],
            esp=cand["esp"],
            esp_rank=result["esp_rank"][v["index"]],
            fidelity=v["fidelity"],
            features=features(cand["circuit"]),
            routed_features=features(v["routed"])
        )
        row.update(distribution_metrics(v["p_ideal"], v["p_noisy"], cfg["shots"], seeds["shots"], v["index"]))
        verified_rows.append(row)

    baselines = {}
    for level in cfg["baseline_levels"]:
        t = time.perf_counter()
        routed = opt.hardware_transpile(original, coupling, seed=seeds["transpile"], optimization_level=level)
        transpile_s = time.perf_counter() - t
        sim = opt.verify_true_fidelity([routed], noise_model, seeds["simulator"])[0]
        row = dict(
            level=level,
            esp=predict_fidelity_esp(routed, gate_errors),
            fidelity=sim["fidelity"],
            routed_features=features(routed),
            transpile_seconds=transpile_s
        )
        row.update(distribution_metrics(sim["p_ideal"], sim["p_noisy"], cfg["shots"], seeds["shots"],
                                        10_000 + level))
        baselines[str(level)] = row

    esp_values = [r["esp"] for r in verified_rows]
    fid_values = [r["fidelity"] for r in verified_rows]
    original_row = verified_rows[0]
    best_row = next(r for r in verified_rows if r["index"] == result["best_index"])

    summary = dict(
        original_fidelity=result["original_true_fidelity"],
        optimized_fidelity=result["best_true_fidelity"],
        esp_selected_fidelity=result["esp_selected_true_fidelity"],
        improvement=result["improvement"],
        esp_selected_improvement=result["esp_selected_true_fidelity"] - result["original_true_fidelity"],
        original_esp=candidates[0]["esp"],
        best_esp=result["best_esp_score"],
        max_esp=max(c["esp"] for c in candidates),
        esp_spearman=spearman(esp_values, fid_values),
        esp_selected_is_sim_best=result["esp_selected_index"] == result["best_index"],
        optimized_is_original=result["best_index"] == 0,
        original_two_qubit_routed=original_row["routed_features"]["two_qubit_gate_count"],
        optimized_two_qubit_routed=best_row["routed_features"]["two_qubit_gate_count"],
        original_depth_routed=original_row["routed_features"]["depth"],
        optimized_depth_routed=best_row["routed_features"]["depth"],
        original_hellinger=original_row["hellinger_fidelity"],
        optimized_hellinger=best_row["hellinger_fidelity"],
        original_tvd=original_row["tvd"],
        optimized_tvd=best_row["tvd"],
        original_js=original_row["js_divergence"],
        optimized_js=best_row["js_divergence"],
        nodes_evaluated=result["search_stats"]["nodes_evaluated"],
        generations=result["search_stats"]["generations"],
        stop_reason=result["search_stats"]["stop_reason"],
        best_depth_in_graph=candidates[result["best_index"]]["depth"],
        baseline_fidelity={k: v["fidelity"] for k, v in baselines.items()}
    )

    return dict(
        experiment=EXPERIMENT_NAME,
        benchmark=benchmark,
        category=category,
        requested_qubits=n_qubits,
        num_qubits=original.num_qubits,
        seed=seed,
        task_seeds=seeds,
        parameters_bound=parameters_bound,
        physical_qubits=physical_qubits,
        timestamp=datetime.now().isoformat(),
        settings={k: cfg[k] for k in SETTING_KEYS},
        max_evaluations=max_evaluations,
        summary=summary,
        original=dict(qasm=safe_qasm(original), features=features(original)),
        optimized=dict(
            qasm=safe_qasm(result["best_circuit"]),
            routed_qasm=safe_qasm(result["best_routed"]),
            index=result["best_index"],
            features=features(result["best_circuit"]),
            routed_features=features(result["best_routed"])
        ),
        verified=verified_rows,
        baselines=baselines,
        candidate_graph=[dict(index=i, esp=c["esp"], depth=c["depth"], parent=c["parent"])
                         for i, c in enumerate(candidates)],
        search_stats=dict(result["search_stats"], max_evaluations=max_evaluations),
        timing=dict(result["timing"], total_seconds=time.perf_counter() - t_total)
    )


def run_task(task, cfg):
    benchmark, category, n_qubits, seed = task
    path = result_path(cfg["results_dir"], seed, category, benchmark, n_qubits)
    epath = error_path(path)
    t0 = time.perf_counter()
    try:
        record = execute(benchmark, category, n_qubits, seed, cfg)
        save_json(path, record)
        if epath.exists():
            epath.unlink()
        s = record["summary"]
        return "done", (f"seed {seed} {benchmark} n={record['num_qubits']}: orig={s['original_fidelity']:.4f} "
                        f"opt={s['optimized_fidelity']:.4f} esp-pick={s['esp_selected_fidelity']:.4f} "
                        f"evals={s['nodes_evaluated']} {time.perf_counter() - t0:.0f}s")
    except Exception as exc:
        save_json(epath, dict(
            benchmark=benchmark, category=category, requested_qubits=n_qubits, seed=seed,
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
            traceback=traceback.format_exc(), timestamp=datetime.now().isoformat()))
        return "failed", f"seed {seed} {benchmark} n={n_qubits}: {type(exc).__name__}: {str(exc)[:150]}"


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def environment():
    return dict(qiskit=qiskit.__version__, numpy=np.__version__, python=platform.python_version(),
                platform=platform.platform())


def check_settings(seed_dir, settings, force):
    seed_dir.mkdir(parents=True, exist_ok=True)
    meta_path = seed_dir / "meta.json"
    if meta_path.exists():
        old = load_json(meta_path).get("settings", {})
        diff = [f"{k}: saved {old.get(k)!r}, now {v!r}" for k, v in settings.items() if old.get(k) != v]
        if diff and not force:
            return diff
    save_json(meta_path, dict(experiment=EXPERIMENT_NAME, settings=settings, environment=environment(),
                              updated=datetime.now().isoformat()))
    return []


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def aggregate(root):
    root = Path(root)
    runs = [load_json(p) for p in sorted(root.glob("seed*/*/*/n*.json")) if not p.name.endswith(".error.json")]
    errors = [load_json(p) for p in sorted(root.glob("seed*/*/*/n*.error.json"))]
    out = root / "summary"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    if not runs:
        print(f"no finished runs in {root}")
        return 2

    levels = sorted({k for r in runs for k in r["summary"]["baseline_fidelity"]}, key=int)
    flat = []
    for r in runs:
        s = r["summary"]
        row = dict(seed=r["seed"], benchmark=r["benchmark"], category=r["category"], num_qubits=r["num_qubits"])
        row.update({k: v for k, v in s.items() if k != "baseline_fidelity"})
        for lv in levels:
            row[f"qiskit_L{lv}_fidelity"] = s["baseline_fidelity"].get(lv)
        row["total_seconds"] = r["timing"]["total_seconds"]
        row["search_seconds"] = r["timing"]["search_seconds"]
        row["simulation_seconds"] = r["timing"]["simulation_seconds"]
        flat.append(row)

    with open(out / "per_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    fid_cols = ["original_fidelity", "optimized_fidelity", "esp_selected_fidelity"] + [
        f"qiskit_L{lv}_fidelity" for lv in levels]
    extra_cols = ["improvement", "original_hellinger", "optimized_hellinger", "original_tvd",
                  "optimized_tvd", "original_js", "optimized_js", "esp_spearman", "nodes_evaluated"]

    groups = {}
    for row in flat:
        groups.setdefault((row["category"], row["benchmark"], row["num_qubits"]), []).append(row)

    grouped_rows = []
    for (category, benchmark, n), rows in sorted(groups.items()):
        g = dict(category=category, benchmark=benchmark, num_qubits=n, seeds=len(rows))
        for col in fid_cols + extra_cols:
            m, sd = mean_std([r[col] for r in rows])
            g[f"{col}_mean"], g[f"{col}_std"] = m, sd
        grouped_rows.append(g)
    with open(out / "by_benchmark_qubits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grouped_rows[0].keys()))
        w.writeheader()
        w.writerows(grouped_rows)

    by_n = {}
    for row in flat:
        by_n.setdefault(row["num_qubits"], []).append(row)
    by_n_rows = []
    for n, rows in sorted(by_n.items()):
        g = dict(num_qubits=n, runs=len(rows))
        for col in fid_cols + extra_cols:
            g[f"{col}_mean"], g[f"{col}_std"] = mean_std([r[col] for r in rows])
        g["win_rate"] = float(np.mean([r["improvement"] > 1e-12 for r in rows]))
        by_n_rows.append(g)
    with open(out / "by_qubits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(by_n_rows[0].keys()))
        w.writeheader()
        w.writerows(by_n_rows)

    imp = np.array([r["improvement"] for r in flat])
    esp_imp = np.array([r["esp_selected_improvement"] for r in flat])
    spear = [r["esp_spearman"] for r in flat if r["esp_spearman"] is not None]
    glob = dict(
        generated=datetime.now().isoformat(),
        runs=len(flat),
        failed_runs=len(errors),
        stop_reasons=dict(Counter(r["stop_reason"] for r in flat)),
        seeds=sorted({r["seed"] for r in flat}),
        improved=int((imp > 1e-12).sum()),
        unchanged=int((np.abs(imp) <= 1e-12).sum()),
        mean_improvement=float(imp.mean()),
        median_improvement=float(np.median(imp)),
        mean_esp_selected_improvement=float(esp_imp.mean()),
        esp_selected_worse_than_original=int((esp_imp < -1e-12).sum()),
        esp_selected_is_sim_best_share=float(np.mean([r["esp_selected_is_sim_best"] for r in flat])),
        mean_esp_spearman=float(np.mean(spear)) if spear else None,
        median_esp_spearman=float(np.median(spear)) if spear else None,
        mean_nodes_evaluated=float(np.mean([r["nodes_evaluated"] for r in flat])),
        total_seconds=float(sum(r["total_seconds"] for r in flat)),
        environment=environment(),
        errors=[dict(seed=e["seed"], benchmark=e["benchmark"], n=e["requested_qubits"], error=e["error"])
                for e in errors]
    )
    for lv in levels:
        col = f"qiskit_L{lv}_fidelity"
        diffs = np.array([r["optimized_fidelity"] - r[col] for r in flat if r[col] is not None])
        glob[f"vs_qiskit_L{lv}"] = dict(
            optimizer_better=int((diffs > 1e-12).sum()),
            equal=int((np.abs(diffs) <= 1e-12).sum()),
            optimizer_worse=int((diffs < -1e-12).sum()),
            mean_difference=float(diffs.mean()))
    save_json(out / "summary.json", glob)

    labels = {"original_fidelity": "Original", "optimized_fidelity": "Optimized",
              "esp_selected_fidelity": "ESP top-1"}
    labels.update({f"qiskit_L{lv}_fidelity": f"Qiskit L{lv}" for lv in levels})

    def draw(ax, xs, rows_by_x, title):
        for col in fid_cols:
            means, stds = [], []
            for x in xs:
                m, sd = mean_std([r[col] for r in rows_by_x[x]])
                means.append(np.nan if m is None else m)
                stds.append(0.0 if sd is None else sd)
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=labels[col])
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Fidelity")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw(ax, sorted(by_n), by_n, f"Average fidelity, {len(flat)} runs")
    fig.tight_layout()
    fig.savefig(out / "plots" / "summary_average.png", dpi=150)
    plt.close(fig)

    per_bench = {}
    for row in flat:
        per_bench.setdefault((row["category"], row["benchmark"]), {}).setdefault(row["num_qubits"], []).append(row)
    for (category, benchmark), by_x in per_bench.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        draw(ax, sorted(by_x), by_x, benchmark)
        fig.tight_layout()
        d = out / "plots" / slugify(category)
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{slugify(benchmark)}.png", dpi=150)
        plt.close(fig)

    def save_fig(fig, name):
        fig.tight_layout()
        fig.savefig(out / "plots" / name, dpi=150)
        plt.close(fig)

    names = sorted({r["benchmark"] for r in flat})
    imp_by = {nm: [r["improvement"] for r in flat if r["benchmark"] == nm] for nm in names}
    order = sorted(names, key=lambda nm: np.mean(imp_by[nm]))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
    ax.barh(order, [np.mean(imp_by[nm]) for nm in order], xerr=[np.std(imp_by[nm]) for nm in order],
            capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean fidelity improvement (optimized - original)")
    ax.set_title("Improvement per benchmark (all sizes and seeds)")
    save_fig(fig, "improvement_per_benchmark.png")

    ns_sorted = sorted(by_n)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([str(n) for n in ns_sorted],
           [np.mean([r["improvement"] for r in by_n[n]]) for n in ns_sorted],
           yerr=[np.std([r["improvement"] for r in by_n[n]]) for n in ns_sorted], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Mean fidelity improvement")
    ax.set_title("Improvement by circuit size")
    save_fig(fig, "improvement_by_qubits.png")

    xs, ys, cs = [], [], []
    for r in runs:
        base = r["verified"][0]
        for v in r["verified"][1:]:
            xs.append(v["esp"] - base["esp"])
            ys.append(v["fidelity"] - base["fidelity"])
            cs.append(r["num_qubits"])
    if xs:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=14, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="qubits")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("ESP gain over original")
        ax.set_ylabel("True fidelity gain over original")
        ax.set_title("Does ESP predict the true gain? (verified candidates)")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "esp_vs_true_gain.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(ns_sorted))
    for reason in ("plateau", "budget", "exhausted"):
        counts = np.array([sum(r["stop_reason"] == reason for r in by_n[n]) for n in ns_sorted])
        ax.bar([str(n) for n in ns_sorted], counts, bottom=bottom, label=reason)
        bottom += counts
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Runs")
    ax.set_title("Why the search stopped")
    ax.legend()
    save_fig(fig, "stop_reasons.png")

    width = 0.38
    pos = np.arange(len(ns_sorted))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(pos - width / 2, [np.mean([r["original_two_qubit_routed"] for r in by_n[n]]) for n in ns_sorted],
           width, label="Original")
    ax.bar(pos + width / 2, [np.mean([r["optimized_two_qubit_routed"] for r in by_n[n]]) for n in ns_sorted],
           width, label="Optimized")
    ax.set_xticks(pos)
    ax.set_xticklabels([str(n) for n in ns_sorted])
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Mean two-qubit gates after transpile")
    ax.set_title("Two-qubit gate count")
    ax.legend()
    save_fig(fig, "two_qubit_gates.png")

    print(f"aggregated {len(flat)} runs ({len(errors)} failed) -> {out}")
    print(f"  improved {glob['improved']}, unchanged {glob['unchanged']}, "
          f"mean improvement {glob['mean_improvement']:.5f}, "
          f"mean ESP/fidelity Spearman {glob['mean_esp_spearman']}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("--seeds", default="0-4", help="range like 0-4 or list like 0,2,5")
    add("--qubits", default="2,4,6,8,10")
    add("--jobs", type=int, default=4)
    add("--only", default=None, help="comma list: keep benchmarks whose name contains one of these")
    add("--results-dir", default=str(ROOT / "results" / "optimizer" / EXPERIMENT_NAME))
    add("--max-evaluations", type=int, default=1000, help="hard cap on candidates per run")
    add("--patience", type=int, default=5, help="stop after this many generations without ESP gain (0 = off)")
    add("--coupling", default="none", choices=["none", "sherbrooke"])
    add("--top-k", type=int, default=5)
    add("--beam-width", type=int, default=10)
    add("--shots", type=int, default=8192)
    add("--baseline-levels", default="1,3")
    add("--max-qubits", type=int, default=10)
    add("--retry-failed", action="store_true")
    add("--force-settings", action="store_true")
    add("--aggregate-only", action="store_true")
    return ap.parse_args(argv)


def log_line(root, text):
    print(text, flush=True)
    with open(Path(root) / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}  {text}\n")


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        return aggregate(root)

    cfg = vars(args)
    cfg["baseline_levels"] = parse_int_list(args.baseline_levels)
    cfg["patience"] = None if args.patience <= 0 else args.patience
    seeds = parse_int_list(args.seeds)
    qubits = parse_int_list(args.qubits)
    settings = {k: cfg[k] for k in SETTING_KEYS}

    for seed in seeds:
        diff = check_settings(root / f"seed{seed}", settings, args.force_settings)
        if diff:
            print(f"seed {seed} already has results made with other settings; nothing was run:")
            for d in diff:
                print(f"  {d}")
            print("use the same settings, a new --results-dir, or --force-settings")
            return 2

    names = [n for n in get_benchmarks() if BENCHMARK_REGISTRY[n].get("scalable", True) and n != "Shor's"]
    if args.only:
        terms = [t.strip().lower() for t in args.only.split(",") if t.strip()]
        names = [n for n in names if any(t in n.lower() for t in terms)]

    tasks, skipped_done, skipped_failed = [], 0, 0
    for seed in seeds:
        for name in names:
            category = BENCHMARK_REGISTRY[name]["category"]
            for n in qubits:
                path = result_path(root, seed, category, name, n)
                if path.exists():
                    skipped_done += 1
                elif error_path(path).exists() and not args.retry_failed:
                    skipped_failed += 1
                else:
                    tasks.append((name, category, n, seed))

    log_line(root, f"tasks: {len(tasks)} to run, {skipped_done} already done, {skipped_failed} previously "
                   f"failed (use --retry-failed)  seeds={seeds} qubits={qubits} jobs={args.jobs}")

    t0 = time.perf_counter()
    failed = 0
    try:
        if args.jobs <= 1:
            for i, task in enumerate(tasks, 1):
                status, text = run_task(task, cfg)
                failed += status == "failed"
                log_line(root, f"[{i}/{len(tasks)}] {status.upper()} {text}")
        elif tasks:
            ex = ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker)
            try:
                futures = {ex.submit(run_task, t, cfg): t for t in tasks}
                for i, f in enumerate(as_completed(futures), 1):
                    try:
                        status, text = f.result()
                    except Exception as exc:
                        status, text = "failed", f"{futures[f]}: worker crashed: {type(exc).__name__}: {exc}"
                    failed += status == "failed"
                    log_line(root, f"[{i}/{len(tasks)}] {status.upper()} {text}  ({time.perf_counter() - t0:.0f}s)")
            except KeyboardInterrupt:
                ex.shutdown(wait=False, cancel_futures=True)
                for p in list(getattr(ex, "_processes", {}).values()):
                    p.terminate()
                raise
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        print("\ninterrupted - finished tasks are saved, run the same command again to continue")
        return 130

    log_line(root, f"finished: {len(tasks) - failed} done, {failed} failed, {time.perf_counter() - t0:.0f}s")
    return aggregate(root)


if __name__ == "__main__":
    sys.exit(main())