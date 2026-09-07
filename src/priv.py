import json
import scipy.stats as stats
from pathlib import Path


def run_aggregate_check():
    project_root = Path.cwd()
    results_dir = project_root / "results"

    emp_path = results_dir / "depth_noise_correlation.json"
    theo_path = results_dir / "theoretical_fidelities.json"
    out_path = results_dir / "aggregate_correlation_check.json"

    if not emp_path.exists() or not theo_path.exists():
        print(f"Error: Could not find both JSON files in {results_dir}")
        return

    with open(emp_path, "r") as f:
        emp_data = json.load(f)
    with open(theo_path, "r") as f:
        theo_data = json.load(f)

    theo_lookup = {}
    for c in theo_data.get("circuits", []):
        key = f"{c['base_name']}__{c['num_qubits']}"
        theo_lookup[key] = {}
        for v in c.get("variants_tested", []):
            theo_lookup[key][v["variant_id"]] = v

    # 1. Build the same flat, per-variant lists as before (for reference/comparison)
    flat_tot_depth, flat_2q_depth, flat_emp_fid, flat_th_raw = [], [], [], []

    # 2. Also group everything by base circuit, so we can average WITHIN each
    #    group before correlating - this collapses the ~456 (non-independent,
    #    clustered) variant rows down to ~24 (independent) circuit-level rows.
    groups = {}  # key -> {"tot_depth": [...], "2q_depth": [...], "emp_fid": [...], "th_raw": [...]}

    for c_emp in emp_data.get("circuits", []):
        base = c_emp["base_name"]
        nq = c_emp["num_qubits"]
        key = f"{base}__{nq}"

        if key not in theo_lookup:
            continue

        if key not in groups:
            groups[key] = {"tot_depth": [], "2q_depth": [], "emp_fid": [], "th_raw": []}

        for v_emp in c_emp.get("variants_tested", []):
            vid = v_emp["variant_id"]
            if vid not in theo_lookup[key]:
                continue

            v_theo = theo_lookup[key][vid]

            tot_depth = v_emp["features"]["total_depth"]
            depth_2q = v_emp["features"]["depth_2q"]
            emp_fid = v_emp["metrics"]["fidelity"]
            th_raw = v_theo["theoretical_metrics"]["fidelity_raw"]

            flat_tot_depth.append(tot_depth)
            flat_2q_depth.append(depth_2q)
            flat_emp_fid.append(emp_fid)
            flat_th_raw.append(th_raw)

            groups[key]["tot_depth"].append(tot_depth)
            groups[key]["2q_depth"].append(depth_2q)
            groups[key]["emp_fid"].append(emp_fid)
            groups[key]["th_raw"].append(th_raw)

    if not flat_emp_fid:
        print("Error: No matching data points found between the two files.")
        return

    def mean(xs):
        return sum(xs) / len(xs)

    # Collapse each group to its mean - one row per base circuit
    agg_tot_depth = [mean(g["tot_depth"]) for g in groups.values()]
    agg_2q_depth = [mean(g["2q_depth"]) for g in groups.values()]
    agg_emp_fid = [mean(g["emp_fid"]) for g in groups.values()]
    agg_th_raw = [mean(g["th_raw"]) for g in groups.values()]

    print("=" * 70)
    print("AGGREGATE (CIRCUIT-LEVEL) CORRELATION CHECK")
    print("=" * 70)
    print(f"Flat variant-level rows : {len(flat_emp_fid)}  (pseudoreplicated - NOT independent)")
    print(f"Aggregated circuit rows : {len(agg_emp_fid)}   (one row per base circuit - independent)")

    def report(name, x_flat, x_agg):
        r_flat, p_flat = stats.pearsonr(x_flat, flat_emp_fid)
        rho_flat, sp_flat = stats.spearmanr(x_flat, flat_emp_fid)
        r_agg, p_agg = stats.pearsonr(x_agg, agg_emp_fid)
        rho_agg, sp_agg = stats.spearmanr(x_agg, agg_emp_fid)
        print(f"\n{name}")
        print(f"  Flat (n={len(x_flat):3d}, pseudoreplicated) : "
              f"r={r_flat:7.4f} (p={p_flat:.3e})   rho={rho_flat:7.4f} (p={sp_flat:.3e})")
        print(f"  Aggregated (n={len(x_agg):3d}, honest)       : "
              f"r={r_agg:7.4f} (p={p_agg:.3e})   rho={rho_agg:7.4f} (p={sp_agg:.3e})")
        return {
            "flat": {"n": len(x_flat), "pearson_r": r_flat, "pearson_p": p_flat,
                     "spearman_rho": rho_flat, "spearman_p": sp_flat},
            "aggregated": {"n": len(x_agg), "pearson_r": r_agg, "pearson_p": p_agg,
                           "spearman_rho": rho_agg, "spearman_p": sp_agg},
        }

    results = {
        "total_depth_vs_fidelity": report("Total Depth", flat_tot_depth, agg_tot_depth),
        "2q_depth_vs_fidelity": report("2-Qubit Depth", flat_2q_depth, agg_2q_depth),
        "th_raw_vs_empirical": report("Paper's Raw Formula", flat_th_raw, agg_th_raw),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved comparison to: {out_path}")
    print("=" * 70)
    print("If the aggregated p-values are much larger than the flat ones (they")
    print("almost certainly will be), that confirms the flat p-values were")
    print("inflated by treating clustered/non-independent variants as if they")
    print("were independent samples. The r/rho values are still meaningful either")
    print("way - it's specifically the p-values (statistical significance) that")
    print("were unreliable at the flat/variant level.")


if __name__ == "__main__":
    run_aggregate_check()