import json
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

def run_comparison():
    project_root = Path.cwd()
    results_dir = project_root / "results"
    
    emp_path = results_dir / "depth_noise_correlation.json"
    theo_path = results_dir / "theoretical_fidelities.json"
    out_json_path = results_dir / "merged_fidelity_comparison.json"
    out_spearman_json_path = results_dir / "merged_fidelity_comparison_with_spearman.json"
    out_plot_path = results_dir / "merged_fidelity_plots.png"
    
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

    all_tot_depth = []
    all_2q_depth = []
    
    all_emp_fid = []
    all_th_raw = []
    all_th_eff = []

    merged_circuits = []

    print("=" * 70)
    print("MERGING EMPIRICAL AND THEORETICAL DATA")
    print("=" * 70)

    for c_emp in emp_data.get("circuits", []):
        base = c_emp["base_name"]
        nq = c_emp["num_qubits"]
        key = f"{base}__{nq}"
        
        if key not in theo_lookup:
            print(f"[!] Warning: {key} not found in theoretical data. Skipping.")
            continue
            
        merged_variants = []
        for v_emp in c_emp.get("variants_tested", []):
            vid = v_emp["variant_id"]
            
            if vid not in theo_lookup[key]:
                continue
                
            v_theo = theo_lookup[key][vid]
            
            emp_fid = v_emp["metrics"]["fidelity"]
            tot_depth = v_emp["features"]["total_depth"]
            depth_2q = v_emp["features"]["depth_2q"]
            
            th_raw = v_theo["theoretical_metrics"]["fidelity_raw"]
            th_eff = v_theo["theoretical_metrics"]["fidelity_eff"]
            
            all_tot_depth.append(tot_depth)
            all_2q_depth.append(depth_2q)
            all_emp_fid.append(emp_fid)
            all_th_raw.append(th_raw)
            all_th_eff.append(th_eff)
            
            merged_variants.append({
                "variant_id": vid,
                "features": v_theo["features"],
                "empirical_fidelity": emp_fid,
                "theoretical_fidelity_raw": th_raw,
                "theoretical_fidelity_eff": th_eff
            })
            
        merged_circuits.append({
            "base_name": base,
            "num_qubits": nq,
            "variants_tested": merged_variants
        })

    if not all_emp_fid:
        print("Error: No matching data points found between the two files.")
        return

    pearson_tot_depth, p_tot = stats.pearsonr(all_tot_depth, all_emp_fid)
    pearson_2q_depth, p_2q = stats.pearsonr(all_2q_depth, all_emp_fid)
    pearson_th_raw, p_raw = stats.pearsonr(all_th_raw, all_emp_fid)
    pearson_th_eff, p_eff = stats.pearsonr(all_th_eff, all_emp_fid)

    spearman_tot_depth, sp_p_tot = stats.spearmanr(all_tot_depth, all_emp_fid)
    spearman_2q_depth, sp_p_2q = stats.spearmanr(all_2q_depth, all_emp_fid)
    spearman_th_raw, sp_p_raw = stats.spearmanr(all_th_raw, all_emp_fid)
    spearman_th_eff, sp_p_eff = stats.spearmanr(all_th_eff, all_emp_fid)

    print(f"Total paired circuit variants analyzed: {len(all_emp_fid)}")
    print("\nPEARSON CORRELATIONS (linear fit, vs Empirical Fidelity):")
    print(f"  1. Total Depth          : r = {pearson_tot_depth:7.4f} (p-value: {p_tot:.4e})")
    print(f"  2. 2-Qubit Depth        : r = {pearson_2q_depth:7.4f} (p-value: {p_2q:.4e})")
    print(f"  3. Paper's Raw Formula  : r = {pearson_th_raw:7.4f} (p-value: {p_raw:.4e})")
    print(f"  4. Normalized (Eff)     : r = {pearson_th_eff:7.4f} (p-value: {p_eff:.4e})")

    print("\nSPEARMAN CORRELATIONS (rank/monotonic fit, vs Empirical Fidelity):")
    print(f"  1. Total Depth          : rho = {spearman_tot_depth:7.4f} (p-value: {sp_p_tot:.4e})")
    print(f"  2. 2-Qubit Depth        : rho = {spearman_2q_depth:7.4f} (p-value: {sp_p_2q:.4e})")
    print(f"  3. Paper's Raw Formula  : rho = {spearman_th_raw:7.4f} (p-value: {sp_p_raw:.4e})")
    print(f"  4. Normalized (Eff)     : rho = {spearman_th_eff:7.4f} (p-value: {sp_p_eff:.4e})")

    merged_data = {
        "dataset_size": len(all_emp_fid),
        "statistical_analysis": {
            "pearson_total_depth_vs_fidelity": {"r": float(pearson_tot_depth), "p_value": float(p_tot)},
            "pearson_2q_depth_vs_fidelity": {"r": float(pearson_2q_depth), "p_value": float(p_2q)},
            "pearson_th_raw_vs_empirical": {"r": float(pearson_th_raw), "p_value": float(p_raw)},
            "pearson_th_eff_vs_empirical": {"r": float(pearson_th_eff), "p_value": float(p_eff)},
        },
        "circuits": merged_circuits
    }

    with open(out_json_path, "w") as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"\nSaved merged data and stats to: {out_json_path}")

    merged_data_with_spearman = {
        "dataset_size": len(all_emp_fid),
        "statistical_analysis": {
            "total_depth_vs_fidelity": {
                "pearson_r": float(pearson_tot_depth), "pearson_p_value": float(p_tot),
                "spearman_rho": float(spearman_tot_depth), "spearman_p_value": float(sp_p_tot)
            },
            "2q_depth_vs_fidelity": {
                "pearson_r": float(pearson_2q_depth), "pearson_p_value": float(p_2q),
                "spearman_rho": float(spearman_2q_depth), "spearman_p_value": float(sp_p_2q)
            },
            "th_raw_vs_empirical": {
                "pearson_r": float(pearson_th_raw), "pearson_p_value": float(p_raw),
                "spearman_rho": float(spearman_th_raw), "spearman_p_value": float(sp_p_raw)
            },
            "th_eff_vs_empirical": {
                "pearson_r": float(pearson_th_eff), "pearson_p_value": float(p_eff),
                "spearman_rho": float(spearman_th_eff), "spearman_p_value": float(sp_p_eff)
            },
        },
        "circuits": merged_circuits
    }

    with open(out_spearman_json_path, "w") as f:
        json.dump(merged_data_with_spearman, f, indent=4)

    print(f"Saved merged data with Pearson + Spearman stats to: {out_spearman_json_path}")

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(all_tot_depth, all_emp_fid, alpha=0.7, color='teal')
    plt.title(f"Total Depth vs Empirical Fidelity\n(r = {pearson_tot_depth:.3f}, rho = {spearman_tot_depth:.3f})")
    plt.xlabel("Total Circuit Depth")
    plt.ylabel("Empirical Fidelity")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(all_2q_depth, all_emp_fid, alpha=0.7, color='blue')
    plt.title(f"2-Qubit Depth vs Empirical Fidelity\n(r = {pearson_2q_depth:.3f}, rho = {spearman_2q_depth:.3f})")
    plt.xlabel("2-Qubit Circuit Depth")
    plt.ylabel("Empirical Fidelity")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.scatter(all_th_raw, all_emp_fid, alpha=0.7, color='purple')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Diagonal line
    plt.title(f"Paper's Exact Formula (Raw) vs Empirical\n(r = {pearson_th_raw:.3f}, rho = {spearman_th_raw:.3f})")
    plt.xlabel("Theoretical Fidelity (Raw depth)")
    plt.ylabel("Empirical Fidelity")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.scatter(all_th_eff, all_emp_fid, alpha=0.7, color='green')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f"Normalized Formula (Effective) vs Empirical\n(r = {pearson_th_eff:.3f}, rho = {spearman_th_eff:.3f})")
    plt.xlabel("Theoretical Fidelity (Effective depth)")
    plt.ylabel("Empirical Fidelity")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_plot_path)
    plt.close()
    
    print(f"Saved correlation plots to: {out_plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_comparison()