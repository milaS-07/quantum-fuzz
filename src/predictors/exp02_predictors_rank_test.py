from pathlib import Path
import json
import numpy as np
import scipy.stats as stats
from datetime import datetime 

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = ROOT / "results" / "noise_predictors" / "exp02"
INPUT_JSON = EXP_DIR / "fidelity_comparison.json"
OUTPUT_JSON = EXP_DIR / "statistical_analysis.json"

def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return stats.pearsonr(x, y)[0]

def safe_spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return stats.spearmanr(x, y)[0]

def safe_kendall(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return stats.kendalltau(x, y)[0]

def calculate_metrics_for_group(true_fids, pred_vals, is_cost_metric=False):
    """
    true_fids: The ground truth quantum fidelities (Higher is better)
    pred_vals: The predictor values (Fidelities -> Higher better. Depths -> Lower better)
    is_cost_metric: True if the predictor is Depth (lower is better).
    """
    # 1. Correlations (If it's a cost metric like Depth, it should inversely correlate with Fidelity)
    # We take the absolute value or adjust sign so +1.0 always means "perfect prediction of reality"
    sign = -1.0 if is_cost_metric else 1.0
    
    pearson = safe_pearson(true_fids, pred_vals) * sign
    spearman = safe_spearman(true_fids, pred_vals) * sign
    kendall = safe_kendall(true_fids, pred_vals) * sign

    # 2. Ranking
    # True Ranks: Highest fidelity gets Rank 1
    true_ranks = stats.rankdata([-x for x in true_fids], method='min')
    
    # Predictor Ranks: 
    # If fidelity, highest gets Rank 1 (negative sort). If depth, lowest gets Rank 1 (positive sort).
    if is_cost_metric:
        pred_ranks = stats.rankdata(pred_vals, method='min')
    else:
        pred_ranks = stats.rankdata([-x for x in pred_vals], method='min')

    # 3. Rank Errors
    rank_errors = np.abs(true_ranks - pred_ranks)
    avg_rank_error = np.mean(rank_errors)

    # 4. Top-1 Accuracy / Mistake
    # Find the variant(s) the predictor ranked as #1
    pred_best_indices = np.where(pred_ranks == 1)[0]
    # What was the true rank of the variant(s) the predictor chose?
    # (If tied, take the average true rank of the tied elements)
    top1_true_rank = np.mean(true_ranks[pred_best_indices])

    # 5. MAE & RMSE (Only valid for fidelity predictors, not depth)
    if not is_cost_metric:
        mae = np.mean(np.abs(np.array(true_fids) - np.array(pred_vals)))
        rmse = np.sqrt(np.mean((np.array(true_fids) - np.array(pred_vals))**2))
    else:
        mae, rmse = None, None

    return {
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "avg_rank_error": avg_rank_error,
        "top1_true_rank": top1_true_rank,
        "mae": mae,
        "rmse": rmse
    }

def run_analysis():
    if not INPUT_JSON.exists():
        print(f"[!] Input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    circuits = data.get("circuits", [])
    if not circuits:
        print("[!] No circuits found in the JSON.")
        return

    # To aggregate results globally
    predictors = ["EPA", "Theory_Eff", "Total_Depth", "2Q_Depth", "1Q_Depth"]
    global_results = {p: {
        "pearson": [], "spearman": [], "kendall": [], 
        "avg_rank_error": [], "top1_true_rank": [],
        "mae": [], "rmse": []
    } for p in predictors}

    per_group_results = []

    print("=" * 115)
    print(f"{'Circuit Benchmark':<35} | {'Predictor':<12} | {'Pearson':>7} | {'Spearman':>8} | {'Kendall':>7} | {'Avg R-Err':>9} | {'Top-1 Rank':>10}")
    print("-" * 115)

    for circ in circuits:
        base_name = circ["base_name"]
        num_qubits = circ["num_qubits"]
        variants = circ["variants_tested"]
        
        # We need at least 2 variants to compute correlations
        if len(variants) < 2:
            continue

        group_name = f"{base_name} ({num_qubits}q)"
        
        true_fids = []
        epa_fids = []
        theff_fids = []
        total_depths = []
        depth_2qs = []
        depth_1qs = []

        for v in variants:
            # Handle key fallback if `fidelity_real_quantum` isn't present
            true_fid = v["metrics"].get("fidelity_real_quantum", v["metrics"].get("fidelity_real", 0.0))
            true_fids.append(true_fid)
            
            epa_fids.append(v["metrics"]["fidelity_epa"])
            theff_fids.append(v["metrics"]["fidelity_theoretical_eff"])
            
            td = v["features"]["total_depth"]
            d2 = v["features"]["depth_2q"]
            total_depths.append(td)
            depth_2qs.append(d2)
            depth_1qs.append(td - d2) # 1Q depth proxy

        # Calculate metrics for each predictor
        results_epa = calculate_metrics_for_group(true_fids, epa_fids, is_cost_metric=False)
        results_theff = calculate_metrics_for_group(true_fids, theff_fids, is_cost_metric=False)
        results_td = calculate_metrics_for_group(true_fids, total_depths, is_cost_metric=True)
        results_d2q = calculate_metrics_for_group(true_fids, depth_2qs, is_cost_metric=True)
        results_d1q = calculate_metrics_for_group(true_fids, depth_1qs, is_cost_metric=True)

        group_dict = {
            "group_name": group_name,
            "num_variants": len(variants),
            "predictors": {
                "EPA": results_epa,
                "Theory_Eff": results_theff,
                "Total_Depth": results_td,
                "2Q_Depth": results_d2q,
                "1Q_Depth": results_d1q
            }
        }
        per_group_results.append(group_dict)

        # Print to console
        for p_name, res in group_dict["predictors"].items():
            print(f"{group_name[:35]:<35} | {p_name:<12} | {res['pearson']:>7.3f} | {res['spearman']:>8.3f} | {res['kendall']:>7.3f} | {res['avg_rank_error']:>9.2f} | {res['top1_true_rank']:>10.2f}")
            
            # Aggregate for global averages
            global_results[p_name]["pearson"].append(res["pearson"])
            global_results[p_name]["spearman"].append(res["spearman"])
            global_results[p_name]["kendall"].append(res["kendall"])
            global_results[p_name]["avg_rank_error"].append(res["avg_rank_error"])
            global_results[p_name]["top1_true_rank"].append(res["top1_true_rank"])
            if res["mae"] is not None:
                global_results[p_name]["mae"].append(res["mae"])
                global_results[p_name]["rmse"].append(res["rmse"])
        print("-" * 115)

    # Calculate Global Averages
    print("=" * 115)
    print("GLOBAL AVERAGES ACROSS ALL CIRCUITS")
    print("=" * 115)
    print(f"{'Predictor':<15} | {'Pearson':>7} | {'Spearman':>8} | {'Kendall':>7} | {'Avg R-Err':>9} | {'Top-1 Rank':>10} | {'MAE':>6} | {'RMSE':>6}")
    print("-" * 115)
    
    final_averages = {}
    for p_name, metrics in global_results.items():
        avg_pearson = np.mean(metrics["pearson"])
        avg_spearman = np.mean(metrics["spearman"])
        avg_kendall = np.mean(metrics["kendall"])
        avg_rank_err = np.mean(metrics["avg_rank_error"])
        avg_top1 = np.mean(metrics["top1_true_rank"])
        
        avg_mae = np.mean(metrics["mae"]) if metrics["mae"] else 0.0
        avg_rmse = np.mean(metrics["rmse"]) if metrics["rmse"] else 0.0

        final_averages[p_name] = {
            "pearson": avg_pearson, "spearman": avg_spearman, "kendall": avg_kendall,
            "avg_rank_error": avg_rank_err, "top1_true_rank": avg_top1,
            "mae": avg_mae, "rmse": avg_rmse
        }

        mae_str = f"{avg_mae:.4f}" if avg_mae > 0 else "N/A"
        rmse_str = f"{avg_rmse:.4f}" if avg_rmse > 0 else "N/A"

        print(f"{p_name:<15} | {avg_pearson:>7.3f} | {avg_spearman:>8.3f} | {avg_kendall:>7.3f} | {avg_rank_err:>9.2f} | {avg_top1:>10.2f} | {mae_str:>6} | {rmse_str:>6}")

    print("=" * 115)

    # Save to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "global_averages": final_averages,
        "per_group_results": per_group_results
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n[+] Detailed statistical results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_experiment = run_analysis
    run_analysis()