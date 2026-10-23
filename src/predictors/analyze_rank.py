import argparse
import json
import math
import sys
from datetime import datetime

import numpy as np
import scipy.stats as stats

from analyze_predictors import (
    MIN_FID_DELTA,
    PREDICTORS,
    SE_Z,
    pairwise_counts,
    resolve_input,
    variant_values,
)

VARIANT_COUNTS = [5, 10, 15, 20, 30]
COVERAGES = [0.95, 0.99, 1.0]
ROUNDS = 200
SEED = 0


def needed_k(true_arr, ses, pred, is_cost, tolerant):
  best = int(np.argmax(true_arr))
  if tolerant:
    tol = np.maximum(
        SE_Z * np.sqrt(ses**2 + ses[best] ** 2), MIN_FID_DELTA
    )
    good = (true_arr[best] - true_arr) <= tol
  else:
    good = np.zeros(len(true_arr), dtype=bool)
    good[best] = True
  ranks = stats.rankdata(pred if is_cost else -pred, method="max")
  return int(ranks[good].min())


def coverage_k(values, coverage):
  ordered = sorted(values)
  index = max(math.ceil(coverage * len(ordered)) - 1, 0)
  return int(ordered[index])


def collect(data, predictor, tolerant):
  rng = np.random.default_rng(SEED)
  is_cost = PREDICTORS[predictor]
  result = {}
  for circ in data.get("circuits", []):
    variants = circ["variants_tested"]
    if len(variants) < 2:
      continue
    fids, ses, preds, esp = [], [], [], []
    for v in variants:
      fid, se, values = variant_values(v)
      fids.append(fid)
      ses.append(se)
      preds.append(values[predictor])
      esp.append(values["ESP"])
    if pairwise_counts(fids, ses, esp, False)[2] == 0:
      continue
    fids = np.asarray(fids, dtype=float)
    ses = np.asarray(ses, dtype=float)
    preds = np.asarray(preds, dtype=float)
    n = len(fids)
    bucket = result.setdefault(
        circ["num_qubits"], {k: [] for k in VARIANT_COUNTS}
    )
    for count in VARIANT_COUNTS:
      if count > n:
        continue
      rounds = 1 if count == n else ROUNDS
      for _ in range(rounds):
        idx = (
            np.arange(n)
            if count == n
            else rng.choice(n, size=count, replace=False)
        )
        bucket[count].append(
            needed_k(fids[idx], ses[idx], preds[idx], is_cost, tolerant)
        )
  return result


def build_summary(per_qubits):
  summary = {}
  pooled = {k: [] for k in VARIANT_COUNTS}
  for qubits in sorted(per_qubits):
    row = {}
    for count, values in per_qubits[qubits].items():
      pooled[count] += values
      if values:
        row[str(count)] = {
            f"k_for_{int(c * 100)}": coverage_k(values, c) for c in COVERAGES
        }
    summary[str(qubits)] = row
  summary["all"] = {
      str(count): {
          f"k_for_{int(c * 100)}": coverage_k(values, c) for c in COVERAGES
      }
      for count, values in pooled.items()
      if values
  }
  return summary


def print_tables(summary, predictor, tolerant):
  mode = "tolerant (within 2*SE of best)" if tolerant else "strict (exact best)"
  print(f"\npredictor: {predictor}, mode: {mode}")
  for c in COVERAGES:
    key = f"k_for_{int(c * 100)}"
    print(f"\nk needed to include the best in {int(c * 100)}% of cases")
    print("| Kubita | " + " | ".join(f"K={k}" for k in VARIANT_COUNTS) + " |")
    print("|:---|" + ":---:|" * len(VARIANT_COUNTS))
    for label, row in summary.items():
      cells = [
          str(row[str(k)][key]) if str(k) in row else "-"
          for k in VARIANT_COUNTS
      ]
      print(f"| {label} | " + " | ".join(cells) + " |")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("path", nargs="?", default=None)
  parser.add_argument("--predictor", choices=list(PREDICTORS), default="ESP")
  parser.add_argument("--tolerant", action="store_true")
  args = parser.parse_args()
  sys.argv = [sys.argv[0]] + ([args.path] if args.path else [])
  return args


def main():
  args = parse_args()
  input_json = resolve_input()
  print(f"using: {input_json} (exists: {input_json.exists()})")
  if not input_json.exists():
    print("file not found; pass it explicitly:")
    print("  python analyze_topk_variants.py path/to/exp03")
    return

  with open(input_json, "r") as f:
    data = json.load(f)

  per_qubits = collect(data, args.predictor, args.tolerant)
  if not per_qubits:
    print("no informative groups")
    return

  summary = build_summary(per_qubits)
  output = {
      "timestamp": datetime.now().isoformat(),
      "predictor": args.predictor,
      "tolerant": args.tolerant,
      "rounds": ROUNDS,
      "summary": summary,
  }
  suffix = "tolerant" if args.tolerant else "strict"
  output_json = input_json.parent / f"topk_by_variants_{suffix}.json"
  with open(output_json, "w") as f:
    json.dump(output, f, indent=4)

  print_tables(summary, args.predictor, args.tolerant)
  print(f"\nsaved: {output_json}")


if __name__ == "__main__":
  main()