from datetime import datetime
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = ROOT / "results" / "noise_predictors" / "exp03"

PREDICTORS = {
    "ESP": False,
    "RQC": False,
    "Total_Depth": True,
    "2Q_Depth": True,
}
GROUP_METRICS = [
    "pearson",
    "spearman",
    "kendall",
    "avg_rank_error",
    "top1_true_rank",
    "top1_regret",
    "mae",
    "rmse",
]
SE_Z = 2.0
MIN_FID_DELTA = 1e-4
BOOTSTRAP_ROUNDS = 2000


def safe_corr(func, x, y):
  if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
    return None
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    value = float(func(x, y)[0])
  return value if np.isfinite(value) else None


def signed(value, sign):
  return None if value is None else value * sign


def pairwise_counts(true_fids, true_ses, pred_vals, is_cost):
  sign = -1.0 if is_cost else 1.0
  strict = 0
  ties = 0
  total = 0
  n = len(true_fids)
  for i in range(n):
    for j in range(i + 1, n):
      dt = true_fids[i] - true_fids[j]
      tol = max(
          SE_Z * float(np.sqrt(true_ses[i] ** 2 + true_ses[j] ** 2)),
          MIN_FID_DELTA,
      )
      if abs(dt) <= tol:
        continue
      dp = (pred_vals[i] - pred_vals[j]) * sign
      total += 1
      if dp == 0:
        ties += 1
      elif dt * dp > 0:
        strict += 1
  return strict, ties, total


def group_metrics(true_fids, true_ses, pred_vals, is_cost):
  sign = -1.0 if is_cost else 1.0
  true_arr = np.asarray(true_fids, dtype=float)
  pred_arr = np.asarray(pred_vals, dtype=float)
  n = len(true_arr)

  true_ranks = stats.rankdata(-true_arr, method="min")
  pred_ranks = stats.rankdata(
      pred_arr if is_cost else -pred_arr, method="min"
  )
  rank_errors = np.abs(true_ranks - pred_ranks)
  best = np.where(pred_ranks == 1)[0]

  if is_cost:
    mae, rmse = None, None
  else:
    mae = float(np.mean(np.abs(true_arr - pred_arr)))
    rmse = float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))

  strict, ties, total = pairwise_counts(
      true_fids, true_ses, pred_vals, is_cost
  )

  return {
      "pearson": signed(safe_corr(stats.pearsonr, true_arr, pred_arr), sign),
      "spearman": signed(safe_corr(stats.spearmanr, true_arr, pred_arr), sign),
      "kendall": signed(safe_corr(stats.kendalltau, true_arr, pred_arr), sign),
      "avg_rank_error": float(np.mean(rank_errors)) / max(n - 1, 1),
      "top1_true_rank": float(np.mean(true_ranks[best])),
      "top1_regret": float(true_arr.max() - np.mean(true_arr[best])),
      "mae": mae,
      "rmse": rmse,
      "pair_strict": strict,
      "pair_ties": ties,
      "pair_total": total,
  }


def mean_or_none(values):
  values = [v for v in values if v is not None]
  return float(np.mean(values)) if values else None


def bootstrap_ci(arr, num_col, den_col):
  if len(arr) < 2:
    return None, None
  rng = np.random.default_rng(0)
  idx = rng.integers(0, len(arr), size=(BOOTSTRAP_ROUNDS, len(arr)))
  num = arr[idx, num_col].sum(axis=1)
  den = arr[idx, den_col].sum(axis=1)
  valid = den > 0
  if not valid.any():
    return None, None
  accs = num[valid] / den[valid]
  return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def variant_values(v):
  m = v["metrics"]
  true_fid = m.get("fidelity_real", m.get("fidelity_real_quantum", 0.0))
  esp = m.get("fidelity_esp", m.get("fidelity_epa", 0.0))
  rqc = m.get("fidelity_rqc", m.get("fidelity_theoretical_depth", 0.0))
  se = v.get("simulation", {}).get("fidelity_real_se", 0.0) or 0.0
  return true_fid, se, {
      "ESP": esp,
      "RQC": rqc,
      "Total_Depth": v["features"]["total_depth"],
      "2Q_Depth": v["features"]["depth_2q"],
  }


def circuit_tier(circ):
  if circ.get("method"):
    return circ["method"]
  for v in circ["variants_tested"]:
    method = v.get("simulation", {}).get("method")
    if method:
      return method
  return "unknown"


def summarize_tier(group_rows, num_dropped):
  summary = {}
  for p_name in PREDICTORS:
    rows = [g["predictors"][p_name] for g in group_rows]
    entry = {
        "num_groups": len(rows),
        "num_uninformative_groups": num_dropped,
    }
    for key in GROUP_METRICS:
      entry[key] = mean_or_none([r[key] for r in rows])
      entry[f"{key}_groups"] = sum(1 for r in rows if r[key] is not None)

    clusters = {}
    for g, r in zip(group_rows, rows):
      acc = clusters.setdefault(g["base_name"], [0, 0, 0])
      acc[0] += r["pair_strict"]
      acc[1] += r["pair_ties"]
      acc[2] += r["pair_total"]

    strict = sum(c[0] for c in clusters.values())
    ties = sum(c[1] for c in clusters.values())
    total = sum(c[2] for c in clusters.values())
    nonties = total - ties

    arr = np.array(
        [[s + 0.5 * t, n, s, n - t] for s, t, n in clusters.values()],
        dtype=float,
    )
    lo, hi = bootstrap_ci(arr, 0, 1)
    slo, shi = bootstrap_ci(arr, 2, 3)

    entry["num_clusters"] = len(clusters)
    entry["pairwise_pairs"] = int(total)
    entry["pairwise_accuracy"] = (strict + 0.5 * ties) / total if total else None
    entry["pairwise_ci_low"] = lo
    entry["pairwise_ci_high"] = hi
    entry["tie_rate"] = ties / total if total else None
    entry["strict_pairs"] = int(nonties)
    entry["strict_accuracy"] = strict / nonties if nonties else None
    entry["strict_ci_low"] = slo
    entry["strict_ci_high"] = shi
    summary[p_name] = entry
  return summary


def analyze(input_json: Path):
  with open(input_json, "r") as f:
    data = json.load(f)

  groups_by_tier = {}
  dropped_by_tier = {}
  per_group_results = []

  for circ in data.get("circuits", []):
    variants = circ["variants_tested"]
    if len(variants) < 2:
      continue

    tier = circuit_tier(circ)
    true_fids, true_ses = [], []
    pred_series = {p: [] for p in PREDICTORS}
    for v in variants:
      true_fid, se, preds = variant_values(v)
      true_fids.append(true_fid)
      true_ses.append(se)
      for p_name in PREDICTORS:
        pred_series[p_name].append(preds[p_name])

    group = {
        "group_name": f"{circ['base_name']} ({circ['num_qubits']}q)",
        "base_name": circ["base_name"],
        "num_qubits": circ["num_qubits"],
        "tier": tier,
        "num_variants": len(variants),
        "true_fid_range": float(max(true_fids) - min(true_fids)),
        "predictors": {
            p_name: group_metrics(
                true_fids, true_ses, pred_series[p_name], is_cost
            )
            for p_name, is_cost in PREDICTORS.items()
        },
    }
    group["informative"] = group["predictors"]["ESP"]["pair_total"] > 0
    per_group_results.append(group)

    if group["informative"]:
      groups_by_tier.setdefault(tier, []).append(group)
    else:
      dropped_by_tier[tier] = dropped_by_tier.get(tier, 0) + 1

  if not groups_by_tier:
    return None

  all_groups = [g for rows in groups_by_tier.values() for g in rows]
  summary = {
      "all": summarize_tier(all_groups, sum(dropped_by_tier.values()))
  }
  for tier, rows in groups_by_tier.items():
    summary[tier] = summarize_tier(rows, dropped_by_tier.get(tier, 0))

  return {
      "timestamp": datetime.now().isoformat(),
      "se_z_threshold": SE_Z,
      "min_fidelity_delta": MIN_FID_DELTA,
      "summary": summary,
      "per_group_results": per_group_results,
  }


def resolve_input():
  if len(sys.argv) > 1:
    p = Path(sys.argv[1]).resolve()
    return p / "fidelity_comparison.json" if p.is_dir() else p
  candidates = [EXP_DIR / "fidelity_comparison.json"]
  for base in [Path.cwd(), Path(__file__).resolve().parent]:
    for parent in [base, *base.parents]:
      candidates.append(
          parent
          / "results"
          / "noise_predictors"
          / "exp03"
          / "fidelity_comparison.json"
      )
  for c in candidates:
    if c.exists():
      return c
  return candidates[0]


def fmt(value, width=7, digits=3):
  return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def ci_str(lo, hi):
  return "n/a" if lo is None else f"[{lo:.3f}, {hi:.3f}]"


def print_summary(summary):
  for tier, preds in summary.items():
    first = next(iter(preds.values()))
    print(
        f"\n[{tier}] informative groups={first['num_groups']}"
        f" dropped={first['num_uninformative_groups']}"
        f" benchmarks={first['num_clusters']}"
        f" pairs={first['pairwise_pairs']}"
    )
    print(
        f"{'Predictor':<12} | {'Pearson':>8} | {'Spearman':>8} |"
        f" {'Kendall':>8} | {'RankErr':>7} | {'Top1':>6} |"
        f" {'Regret':>8} | {'MAE':>7} | {'RMSE':>7}"
    )
    for p_name, e in preds.items():
      print(
          f"{p_name:<12} | {fmt(e['pearson'], 8)} |"
          f" {fmt(e['spearman'], 8)} | {fmt(e['kendall'], 8)} |"
          f" {fmt(e['avg_rank_error'])} | {fmt(e['top1_true_rank'], 6, 2)} |"
          f" {fmt(e['top1_regret'], 8, 5)} | {fmt(e['mae'])} |"
          f" {fmt(e['rmse'])}"
      )
    print(
        f"{'Predictor':<12} | {'PairAcc':>7} | {'95% CI':>15} |"
        f" {'TieRate':>7} | {'StrictAcc':>9} | {'95% CI':>15} |"
        f" {'StrictN':>7}"
    )
    for p_name, e in preds.items():
      print(
          f"{p_name:<12} | {fmt(e['pairwise_accuracy'])} |"
          f" {ci_str(e['pairwise_ci_low'], e['pairwise_ci_high']):>15} |"
          f" {fmt(e['tie_rate'])} | {fmt(e['strict_accuracy'], 9)} |"
          f" {ci_str(e['strict_ci_low'], e['strict_ci_high']):>15} |"
          f" {e['strict_pairs']:>7}"
      )


def main():
  input_json = resolve_input()
  print(f"using: {input_json} (exists: {input_json.exists()})")
  if not input_json.exists():
    print("file not found; pass it explicitly:")
    print("  python analyze_predictors.py path/to/exp03")
    return

  result = analyze(input_json)
  if result is None:
    print("no groups with at least one informative pair")
    return

  output_json = input_json.parent / "statistical_analysis.json"
  with open(output_json, "w") as f:
    json.dump(result, f, indent=4)

  print_summary(result["summary"])
  print(f"\nsaved: {output_json}")


if __name__ == "__main__":
  main()