import sys, re, time, json, warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from qiskit import transpile, qasm3
from circuits import get_mqt_circuit, get_benchmarks, BENCHMARK_REGISTRY
from noise.noise_parametrs import get_sherbrooke_noise_model
from predictors.epa_fidelity import get_avg_gate_fidelities
from optimization_search import optimize_circuit, DEFAULT_HARDWARE_BASIS

warnings.filterwarnings("ignore", category=UserWarning)

QUBIT_RANGE = [2, 4, 6, 8]
MAX_EVALUATIONS_BASE = 50
MAX_EVALUATIONS_PER_QUBIT = 100
UNROLL_BASIS = ["u", "cx"]
EXPERIMENT_NAME = "exp04_bfs_survey"

OUT_DIR = Path.cwd() / "results" / "optimizer" / "exp04"
PLOTS_DIR = OUT_DIR / "plots"


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_sherbrooke_coupling_map():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    except ImportError:
        from qiskit.providers.fake_provider import FakeSherbrookeV2 as FakeSherbrooke

    backend = FakeSherbrooke()
    return getattr(backend, "coupling_map", None) or backend.target.build_coupling_map()


def _undirected_adjacency(coupling_map):
    adjacency = {q: set() for q in range(coupling_map.size())}

    for src, dst in coupling_map.get_edges():
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    return adjacency


def _induced_edge_count(adjacency, qubit_set):
    qset = set(qubit_set)
    return sum(len(adjacency[q] & qset) for q in qset) // 2


def _is_connected(adjacency, qubit_set):
    qset = set(qubit_set)

    if not qset:
        return True

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


def _greedy_dense_subset(adjacency, n_qubits, start):
    qset = {start}

    while len(qset) < n_qubits:
        boundary = set().union(
            *(adjacency[q] - qset for q in qset)
        )

        if not boundary:
            return None

        best = max(
            boundary,
            key=lambda q: (
                len(adjacency[q] & qset),
                len(adjacency[q])
            )
        )

        qset.add(best)

    return qset


def _local_search_improve(adjacency, qubit_set, max_iters=50):
    qset = set(qubit_set)
    current_edges = _induced_edge_count(adjacency, qset)

    for _ in range(max_iters):
        boundary = set().union(
            *(adjacency[q] - qset for q in qset)
        )

        improved = False

        for out_q in list(qset):
            for in_q in boundary:
                candidate = (qset - {out_q}) | {in_q}

                if not _is_connected(adjacency, candidate):
                    continue

                candidate_edges = _induced_edge_count(
                    adjacency,
                    candidate
                )

                if candidate_edges > current_edges:
                    qset = candidate
                    current_edges = candidate_edges
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return qset


def _densest_connected_subset(coupling_map, n_qubits, top_k_seeds=8):
    adjacency = _undirected_adjacency(coupling_map)
    candidates = []

    for start in adjacency:
        patch = _greedy_dense_subset(
            adjacency,
            n_qubits,
            start
        )

        if patch is not None:
            candidates.append(
                (_induced_edge_count(adjacency, patch), patch)
            )

    if not candidates:
        raise ValueError(
            f"Coupling map has no connected component with at least {n_qubits} qubits"
        )

    candidates.sort(key=lambda row: row[0], reverse=True)

    best_edges = -1
    best_patch = None

    for _, patch in candidates[:top_k_seeds]:
        refined = _local_search_improve(adjacency, patch)
        refined_edges = _induced_edge_count(adjacency, refined)

        if refined_edges > best_edges:
            best_edges = refined_edges
            best_patch = refined

    return sorted(best_patch)


def get_reduced_coupling_map(full_coupling_map, n_qubits):
    qubits = _densest_connected_subset(
        full_coupling_map,
        n_qubits
    )
    return full_coupling_map.reduce(qubits)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)

    tmp_path.replace(path)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def circuit_features(qc):
    ops = qc.count_ops()

    return {
        "depth": qc.depth(),
        "num_qubits": qc.num_qubits,
        "gate_counts": {k: int(v) for k, v in ops.items()},
        "two_qubit_gate_count": (
            ops.get("cx", 0)
            + ops.get("ecr", 0)
            + ops.get("cz", 0)
        )
    }


def max_evaluations_for(n_qubits):
    return (
        MAX_EVALUATIONS_BASE
        + MAX_EVALUATIONS_PER_QUBIT * n_qubits
    )


def run_single_combo(
    benchmark_name,
    category,
    n_qubits,
    noise_model,
    coupling_map,
    f_1q,
    f_2q,
    hardware_basis
):
    t0 = time.time()

    raw_circuit = get_mqt_circuit(
        benchmark_name,
        num_qubits=n_qubits
    )

    original_circuit = transpile(
        raw_circuit,
        basis_gates=UNROLL_BASIS,
        optimization_level=0
    )

    max_evaluations = max_evaluations_for(n_qubits)

    t_search = time.time()

    result = optimize_circuit(
        original_circuit,
        noise_model,
        f_1q,
        f_2q,
        max_evaluations=max_evaluations,
        coupling_map=coupling_map,
        hardware_basis=hardware_basis
    )

    search_seconds = time.time() - t_search
    best_circuit = result["best_circuit"]

    return {
        "benchmark": benchmark_name,
        "category": category,
        "num_qubits": n_qubits,
        "timestamp": datetime.now().isoformat(),
        "original": {
            "qasm": qasm3.dumps(original_circuit),
            "features": circuit_features(original_circuit),
            "true_fidelity": result["original_true_fidelity"]
        },
        "optimized": {
            "qasm": qasm3.dumps(best_circuit),
            "features": circuit_features(best_circuit),
            "true_fidelity": result["best_true_fidelity"]
        },
        "improvement": {
            "true_fidelity_delta": result["improvement"]
        },
        "search_stats": {
            **result["search_stats"],
            "max_evaluations": max_evaluations
        },
        "timing": {
            "search_seconds": search_seconds,
            "total_seconds": time.time() - t0
        }
    }


def build_summary():
    per_benchmark = {}
    per_qubit_totals = {
        n: {
            "orig_sum": 0.0,
            "opt_sum": 0.0,
            "count": 0
        }
        for n in QUBIT_RANGE
    }

    for result_path in sorted(OUT_DIR.glob("*/*/n*.json")):
        record = load_json(result_path)
        name = record["benchmark"]
        n = record["num_qubits"]

        entry = per_benchmark.setdefault(
            name,
            {
                "category": record["category"],
                "points": []
            }
        )

        entry["points"].append({
            "num_qubits": n,
            "original_fidelity": record["original"]["true_fidelity"],
            "optimized_fidelity": record["optimized"]["true_fidelity"]
        })

        if n in per_qubit_totals:
            per_qubit_totals[n]["orig_sum"] += (
                record["original"]["true_fidelity"]
            )
            per_qubit_totals[n]["opt_sum"] += (
                record["optimized"]["true_fidelity"]
            )
            per_qubit_totals[n]["count"] += 1

    for entry in per_benchmark.values():
        entry["points"].sort(
            key=lambda p: p["num_qubits"]
        )

    averages = []

    for n in QUBIT_RANGE:
        totals = per_qubit_totals[n]

        if totals["count"] > 0:
            averages.append({
                "num_qubits": n,
                "avg_original_fidelity": (
                    totals["orig_sum"] / totals["count"]
                ),
                "avg_optimized_fidelity": (
                    totals["opt_sum"] / totals["count"]
                ),
                "num_circuits": totals["count"]
            })

    return {
        "generated": datetime.now().isoformat(),
        "per_benchmark": per_benchmark,
        "averages_by_qubit_count": averages
    }


def generate_plots(summary):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, entry in summary["per_benchmark"].items():
        points = entry["points"]

        if not points:
            continue

        xs = [p["num_qubits"] for p in points]
        orig = [p["original_fidelity"] for p in points]
        opt = [p["optimized_fidelity"] for p in points]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(xs, orig, marker="o", label="Original")
        ax.plot(xs, opt, marker="o", label="Optimized")
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Fidelity")
        ax.set_title(name)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        category_dir = (
            PLOTS_DIR /
            slugify(entry["category"])
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            category_dir / f"{slugify(name)}.png",
            dpi=150
        )

        plt.close(fig)

    averages = summary["averages_by_qubit_count"]

    if averages:
        xs = [a["num_qubits"] for a in averages]
        orig = [a["avg_original_fidelity"] for a in averages]
        opt = [a["avg_optimized_fidelity"] for a in averages]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            xs,
            orig,
            marker="o",
            label="Original (avg)"
        )

        ax.plot(
            xs,
            opt,
            marker="o",
            label="Optimized (avg)"
        )

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Average fidelity")
        ax.set_title(
            f"Average fidelity across {len(summary['per_benchmark'])} benchmarks"
        )
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / "summary_average.png",
            dpi=150
        )

        plt.close(fig)


def run_experiment():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "=" * 70
        + "\nSTARTING EXP04: BFS RESYNTHESIS SURVEY\n"
        + "=" * 70,
        flush=True
    )

    print(
        f"[*] Results dir: {OUT_DIR}",
        flush=True
    )

    print(
        "[*] Loading noise model (IBM Sherbrooke)...",
        flush=True
    )

    noise_model = get_sherbrooke_noise_model()
    full_coupling_map = get_sherbrooke_coupling_map()

    f_1q, f_2q = get_avg_gate_fidelities(
        noise_model
    )

    hardware_basis = DEFAULT_HARDWARE_BASIS

    coupling_maps_by_n = {
        n: get_reduced_coupling_map(
            full_coupling_map,
            n
        )
        for n in QUBIT_RANGE
    }

    meta = {
        "experiment": EXPERIMENT_NAME,
        "qubit_range": QUBIT_RANGE,
        "max_evaluations_base": MAX_EVALUATIONS_BASE,
        "max_evaluations_per_qubit": MAX_EVALUATIONS_PER_QUBIT,
        "coupling_map_size_by_n": {
            n: cm.size()
            for n, cm in coupling_maps_by_n.items()
        },
        "hardware_basis": list(hardware_basis),
        "f_1q": f_1q,
        "f_2q": f_2q,
        "updated": datetime.now().isoformat()
    }

    save_json(
        OUT_DIR / "meta.json",
        meta
    )

    print(
        f"[*] f_1q={f_1q:.6f}, f_2q={f_2q:.6f}",
        flush=True
    )

    benchmark_names = [
        name
        for name in get_benchmarks()
        if (
            BENCHMARK_REGISTRY.get(name, {}).get(
                "scalable",
                True
            )
            and name != "Shor's"
        )
    ]

    print(
        f"[*] {len(benchmark_names)} scalable benchmark(s), "
        f"qubit range {QUBIT_RANGE}",
        flush=True
    )

    errors_path = OUT_DIR / "errors.json"

    errors = (
        load_json(errors_path)["errors"]
        if errors_path.exists()
        else []
    )

    for b_idx, benchmark_name in enumerate(
        benchmark_names,
        1
    ):
        category = BENCHMARK_REGISTRY[
            benchmark_name
        ]["category"]

        benchmark_dir = (
            OUT_DIR
            / slugify(category)
            / slugify(benchmark_name)
        )

        benchmark_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\n[{b_idx}/{len(benchmark_names)}] "
            f"{benchmark_name} "
            f"(category: {category})",
            flush=True
        )

        for n_qubits in QUBIT_RANGE:
            result_path = (
                benchmark_dir
                / f"n{n_qubits}.json"
            )

            if result_path.exists():
                print(
                    f"    n={n_qubits}: already done, skipping",
                    flush=True
                )
                continue

            max_evaluations = (
                max_evaluations_for(n_qubits)
            )

            print(
                f"    n={n_qubits}: running "
                f"(max_evaluations={max_evaluations})...",
                flush=True
            )

            try:
                record = run_single_combo(
                    benchmark_name,
                    category,
                    n_qubits,
                    noise_model,
                    coupling_maps_by_n[n_qubits],
                    f_1q,
                    f_2q,
                    hardware_basis
                )

                save_json(
                    result_path,
                    record
                )

                orig_fid = (
                    record["original"]["true_fidelity"]
                )

                opt_fid = (
                    record["optimized"]["true_fidelity"]
                )

                delta = (
                    record["improvement"]
                    ["true_fidelity_delta"]
                )

                evals = (
                    record["search_stats"]
                    ["nodes_evaluated"]
                )

                secs = (
                    record["timing"]
                    ["total_seconds"]
                )

                print(
                    f"    n={n_qubits}: done "
                    f"(orig_fid={orig_fid:.6f}, "
                    f"opt_fid={opt_fid:.6f}, "
                    f"improvement={delta:.6f}, "
                    f"evaluated={evals}, "
                    f"{secs:.1f}s)",
                    flush=True
                )

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print(
                    f"    n={n_qubits}: FAILED ({e})",
                    flush=True
                )

                errors.append({
                    "benchmark": benchmark_name,
                    "num_qubits": n_qubits,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

                save_json(
                    errors_path,
                    {"errors": errors}
                )

    print(
        "\n[*] Sweep complete. "
        "Building summary and plots...",
        flush=True
    )

    summary = build_summary()

    save_json(
        OUT_DIR / "summary.json",
        summary
    )

    generate_plots(summary)

    print(
        f"[*] Done. Results under: {OUT_DIR}\n"
        + "=" * 70,
        flush=True
    )


if __name__ == "__main__":
    run_experiment()