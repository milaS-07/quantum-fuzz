import json
import os
import statistics
from collections import defaultdict

BASE_DIR = r"D:\results_quantum"

CATEGORY_DISPLAY = {
    "Clifford": "Clifford",
    "Non-Parametric Non-Clifford": "Non-Clifford Non-Parametric",
    "PQC": "PQC",
    "Control": "Control",
}
CATEGORY_ORDER = ["Clifford", "Non-Parametric Non-Clifford", "PQC", "Control"]

DEPTH_BUCKETS = [
    (r"$D \le 10$", 0, 10),
    (r"$11 - 30$", 11, 30),
    (r"$31 - 60$", 31, 60),
    (r"$61 - 100$", 61, 100),
    (r"$D > 100$", 101, float("inf")),
]

TARGET_QUBIT_ROWS = [2, 4, 6, 8, 10, 12, 14, 16, 18]


def load_all_results(base_dir):
    """Walk every metrics.json on disk, return {bench_name: bench_data}."""
    results = {}
    for root, _, files in os.walk(base_dir):
        if "metrics.json" not in files:
            continue
        with open(os.path.join(root, "metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        results[data["metadata"]["name"]] = data
    return results


def fmt(values):
    """Mean ± population-std, 4 decimals. 'N/A' if no data."""
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def valid_entries(bench_data):
    """Per-qubit entries with real simulated data — excludes the
    'signal drowned in noise' placeholder rows (depth == -1, no counts)."""
    return [r for r in bench_data["results_per_qubit"].values() if r.get("depth", -1) != -1]


def table_by_category(all_results):
    buckets = defaultdict(lambda: {"F": [], "TVD": [], "JSD": []})
    for data in all_results.values():
        cat = data["metadata"]["category"]
        for r in valid_entries(data):
            buckets[cat]["F"].append(r["fidelity"])
            buckets[cat]["TVD"].append(r["tvd"])
            buckets[cat]["JSD"].append(r["js_divergence"])

    print("#### Rezultati svih kola\n")
    print("| Kategorija | Fidelity ($F$) `[Mean ± Std]` | TVD `[Mean ± Std]` | JSD `[Mean ± Std]` |")
    print("| :--- | :---: | :---: | :---: |")
    for cat in CATEGORY_ORDER:
        v = buckets.get(cat, {"F": [], "TVD": [], "JSD": []})
        if not v["F"]:
            continue
        print(f"| **{CATEGORY_DISPLAY[cat]}** | {fmt(v['F'])} | {fmt(v['TVD'])} | {fmt(v['JSD'])} |")
    print("\n---\n")


def table_by_qubit_count(all_results):
    buckets = {n: {"F": [], "TVD": [], "JSD": []} for n in TARGET_QUBIT_ROWS}
    for data in all_results.values():
        for r in valid_entries(data):
            n = r["num_qubits"]
            if n in buckets:
                buckets[n]["F"].append(r["fidelity"])
                buckets[n]["TVD"].append(r["tvd"])
                buckets[n]["JSD"].append(r["js_divergence"])

    print("#### Rezultati po broju kjubita\n")
    print("| Broj kubita ($N$) | Fidelity ($F$) `[Mean ± Std]` | TVD `[Mean ± Std]` | JSD `[Mean ± Std]` |")
    print("| :---: | :---: | :---: | :---: |")
    for n in TARGET_QUBIT_ROWS:
        v = buckets[n]
        print(f"| **{n}** | {fmt(v['F'])} | {fmt(v['TVD'])} | {fmt(v['JSD'])} |")
    print("\n---\n")


def table_by_depth(all_results):
    buckets = {label: {"F": [], "TVD": [], "JSD": []} for label, _, _ in DEPTH_BUCKETS}
    for data in all_results.values():
        for r in valid_entries(data):
            d = r["depth"]
            for label, lo, hi in DEPTH_BUCKETS:
                if lo <= d <= hi:
                    buckets[label]["F"].append(r["fidelity"])
                    buckets[label]["TVD"].append(r["tvd"])
                    buckets[label]["JSD"].append(r["js_divergence"])
                    break

    print("### Rezultati po dubini kola\n")
    print("| Opseg dubine ($D$) | Fidelity ($F$) `[Mean ± Std]` | TVD `[Mean ± Std]` | JSD `[Mean ± Std]` |")
    print("| :---: | :---: | :---: | :---: |")
    for label, _, _ in DEPTH_BUCKETS:
        v = buckets[label]
        print(f"| **{label}** | {fmt(v['F'])} | {fmt(v['TVD'])} | {fmt(v['JSD'])} |")
    print("\n---\n")


def table_scaling_deltas(all_results):
    rows_by_category = defaultdict(list)
    all_deltas = {"F": [], "TVD": [], "JSD": []}

    for bench_name, data in all_results.items():
        if not data["metadata"]["scalable"]:
            continue
        entries = sorted(valid_entries(data), key=lambda r: r["num_qubits"])
        if len(entries) < 2:
            continue
        first, last = entries[0], entries[-1]
        dF = last["fidelity"] - first["fidelity"]
        dTVD = last["tvd"] - first["tvd"]
        dJSD = last["js_divergence"] - first["js_divergence"]
        rows_by_category[data["metadata"]["category"]].append((bench_name, dF, dTVD, dJSD))
        all_deltas["F"].append(dF)
        all_deltas["TVD"].append(dTVD)
        all_deltas["JSD"].append(dJSD)

    print("### Promena metrika pri skaliranju skalabilnih kola\n")
    print(r"| Kolo | Promena Fidelity-ja ($\Delta F$) | Promena TVD-a ($\Delta \text{TVD}$) | Promena JSD-a ($\Delta \text{JSD}$) |")
    print("| :--- | :---: | :---: | :---: |")
    for cat in CATEGORY_ORDER:
        rows = rows_by_category.get(cat)
        if not rows:
            continue
        print(f"| **{CATEGORY_DISPLAY[cat]}** | | | |")
        for name, dF, dTVD, dJSD in sorted(rows):
            print(f"| {name} | {dF:+.4f} | {dTVD:+.4f} | {dJSD:+.4f} |")

    if all_deltas["F"]:
        print(f"| **Prosek svih kola** | {statistics.mean(all_deltas['F']):+.4f} | "
              f"{statistics.mean(all_deltas['TVD']):+.4f} | {statistics.mean(all_deltas['JSD']):+.4f} |")


if __name__ == "__main__":
    all_results = load_all_results(BASE_DIR)
    table_by_category(all_results)
    table_by_qubit_count(all_results)
    table_by_depth(all_results)
    table_scaling_deltas(all_results)