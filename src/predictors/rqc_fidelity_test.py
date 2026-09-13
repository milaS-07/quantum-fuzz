import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime

from qiskit import QuantumCircuit, qasm3

from circuits import get_mqt_circuit, MQT_NAME_MAP
from transformations import get_matrix_decompositions
from predictors.rqc_fidelity import average_fidelity, alpha_for_hardware, guess_architecture, Architecture


THEORETICAL_P = 0.001


def get_sherbrooke_coupling_map():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    except ImportError:
        from qiskit.providers.fake_provider import FakeSherbrookeV2 as FakeSherbrooke
    backend = FakeSherbrooke()
    return backend.coupling_map


def extract_circuit_features(qc: QuantumCircuit) -> dict:
    total_depth = qc.depth()
    depth_2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
    ops = qc.count_ops()
    cx_count = ops.get('cx', 0) + ops.get('ecr', 0) + ops.get('cz', 0)

    L = qc.num_qubits
    expected_gates_per_layer = L / 2.0
    effective_2q_depth = (cx_count / expected_gates_per_layer) if expected_gates_per_layer > 0 else 0

    return {
        "total_depth": total_depth,
        "depth_2q": depth_2q,
        "cx_count": cx_count,
        "effective_2q_depth": effective_2q_depth,
    }


def collect_circuit_variants(start_circuit: QuantumCircuit, max_variants: int = 30) -> list:
    visited_hashes = set()
    variants = []
    queue = [start_circuit]

    while queue and len(variants) < max_variants:
        current_qc = queue.pop(0)
        qc_hash = qasm3.dumps(current_qc)

        if qc_hash in visited_hashes:
            continue

        visited_hashes.add(qc_hash)
        variants.append(current_qc)

        successors = get_matrix_decompositions(current_qc)
        for next_qc, _ in successors:
            if len(variants) + len(queue) < max_variants * 2:
                queue.append(next_qc)

    return variants


def save_progress(experiment_data: dict, json_path: Path):
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=4)


def circuit_key(base_name: str, num_qubits: int) -> str:
    return f"{base_name}__{num_qubits}q"


# empirical base_name values are the mqt "short" ids (e.g. "qft"), but
# get_mqt_circuit expects the long descriptive benchmark name -- invert
# the map from circuits.py to go from one to the other.
SHORT_TO_FULL_NAME = {v: k for k, v in MQT_NAME_MAP.items()}


def resolve_full_name(short_name: str) -> str:
    if short_name in SHORT_TO_FULL_NAME:
        return SHORT_TO_FULL_NAME[short_name]
    if short_name in MQT_NAME_MAP:
        return short_name
    raise KeyError(
        f"Can't map empirical base_name '{short_name}' back to a get_mqt_circuit "
        f"benchmark name. Add it to MQT_NAME_MAP or handle it manually."
    )


def main():
    coupling_map = get_sherbrooke_coupling_map()
    architecture: Architecture = guess_architecture(coupling_map)
    alpha = alpha_for_hardware("ibm_sherbrooke")
    print(f"[*] Sherbrooke coupling map -> classified architecture: {architecture}", flush=True)
    print(f"[*] alpha derived from ibm_sherbrooke hardware preset: {alpha:.6f}", flush=True)

    project_root = Path.cwd()
    empirical_path = project_root / "results" / "noise_predictors" / "empirical_depth_noise.json"
    out_dir = project_root / "results" / "noise_predictors" / "exp02"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "theoretical_fidelities.json"

    print("=" * 70, flush=True)
    print("GENERATING THEORETICAL FIDELITIES FOR EMPIRICAL CIRCUIT SET", flush=True)
    print("=" * 70, flush=True)

    with open(empirical_path, "r") as f:
        empirical_data = json.load(f)

    if out_path.exists():
        with open(out_path, "r") as f:
            experiment_data = json.load(f)
        if "circuits" not in experiment_data:
            experiment_data = {"timestamp": datetime.now().isoformat(), "circuits": []}
    else:
        experiment_data = {"timestamp": datetime.now().isoformat(), "circuits": []}

    completed_keys = {
        circuit_key(c["base_name"], c["num_qubits"]) for c in experiment_data["circuits"]
    }

    empirical_circuits = empirical_data["circuits"]
    print(f"[*] Empirical circuits to match: {len(empirical_circuits)}", flush=True)
    if completed_keys:
        print(f"[*] Already done: {len(completed_keys)} circuit(s), will be skipped.", flush=True)

    for idx, emp_circuit in enumerate(empirical_circuits, 1):
        base_name = emp_circuit["base_name"]
        num_qubits = emp_circuit["num_qubits"]
        key = circuit_key(base_name, num_qubits)

        if key in completed_keys:
            print(f"[{idx}/{len(empirical_circuits)}] Skipping (already done): "
                  f"{base_name}, {num_qubits}q", flush=True)
            continue

        emp_variants = emp_circuit["variants_tested"]
        n_expected = len(emp_variants)

        print("\n" + "-" * 50, flush=True)
        print(f"[{idx}/{len(empirical_circuits)}] {base_name}, {num_qubits} qubits "
              f"(expecting {n_expected} variant(s))", flush=True)

        try:
            full_name = resolve_full_name(base_name)
            base_circuit = get_mqt_circuit(full_name, num_qubits=num_qubits)
        except Exception as e:
            print(f"    [!] Could not rebuild base circuit: {e} -- skipping this circuit.", flush=True)
            continue

        variants = collect_circuit_variants(base_circuit, max_variants=max(30, n_expected))

        if len(variants) != n_expected:
            print(f"    [!] MISMATCH: regenerated {len(variants)} variant(s) but the "
                  f"empirical file has {n_expected}. If this keeps happening after "
                  f"switching to the fixed get_matrix_decompositions, treat the "
                  f"empirical entry itself as suspect.", flush=True)

        circuit_data = {
            "base_name": base_name,
            "num_qubits": num_qubits,
            "variants_tested": [],
        }

        n = min(len(variants), n_expected) if n_expected else len(variants)
        for i in range(n):
            qc = variants[i]
            emp_variant = emp_variants[i]
            features = extract_circuit_features(qc)

            emp_feats = emp_variant.get("features", {})
            for field in ("total_depth", "depth_2q"):
                emp_val = emp_feats.get(field)
                if emp_val is not None and emp_val != features[field]:
                    print(f"    [!] Variant {i}: {field} mismatch -- "
                          f"empirical={emp_val}, regenerated={features[field]}. "
                          f"Treat this row with suspicion.", flush=True)

            raw_T = features["depth_2q"]
            eff_T = features["effective_2q_depth"]

            fidelity_raw = average_fidelity(
                L=num_qubits, T=raw_T, p=THEORETICAL_P,
                alpha=alpha, architecture=architecture,
            )
            fidelity_eff = average_fidelity(
                L=num_qubits, T=eff_T, p=THEORETICAL_P,
                alpha=alpha, architecture=architecture,
            )

            circuit_data["variants_tested"].append({
                "variant_id": emp_variant.get("variant_id", i),
                "features": features,
                "theoretical_metrics": {
                    "fidelity_raw": fidelity_raw,
                    "fidelity_eff": fidelity_eff,
                    "architecture": architecture,
                    "alpha": alpha,
                    "p": THEORETICAL_P,
                },
            })

            print(f"      [Variant {i:02d}] 2Q Depth: {raw_T:<3} (Eff: {eff_T:5.2f}) | "
                  f"Fid(raw): {fidelity_raw:.4f} | Fid(eff): {fidelity_eff:.4f}", flush=True)

        experiment_data["circuits"].append(circuit_data)
        completed_keys.add(key)
        save_progress(experiment_data, out_path)
        print(f"    - [saved progress: {len(experiment_data['circuits'])} circuit(s) written "
              f"to {out_path}]", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"DONE. Results saved to: {out_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()