from typing import Dict, List, Any
from src.circuits import BENCHMARK_REGISTRY
import math

def calculate_tvd(counts_ideal: dict, counts_noisy: dict, shots: int) -> float:
    all_keys = set(counts_ideal.keys()).union(set(counts_noisy.keys()))
    tvd = 0.0
    for k in all_keys:
        p_ideal = counts_ideal.get(k, 0) / shots
        p_noisy = counts_noisy.get(k, 0) / shots
        tvd += abs(p_ideal - p_noisy)
    return 0.5 * tvd


def calculate_fidelity(counts_ideal: dict, counts_noisy: dict, shots: int) -> float:
    all_keys = set(counts_ideal.keys()).union(set(counts_noisy.keys()))
    bc = 0.0
    for k in all_keys:
        p_ideal = counts_ideal.get(k, 0) / shots
        p_noisy = counts_noisy.get(k, 0) / shots
        bc += math.sqrt(p_ideal * p_noisy)
    return bc ** 2


def calculate_js_divergence(counts_ideal: dict, counts_noisy: dict, shots: int) -> float:
    all_keys = set(counts_ideal.keys()).union(set(counts_noisy.keys()))
    jsd = 0.0
    
    for k in all_keys:
        p = counts_ideal.get(k, 0) / shots
        q = counts_noisy.get(k, 0) / shots
        m = 0.5 * (p + q)
        
        if p > 0 and m > 0:
            jsd += 0.5 * p * math.log2(p / m)
        if q > 0 and m > 0:
            jsd += 0.5 * q * math.log2(q / m)
            
    return jsd



def get_benchmarks(category: str = None, scalable_only: bool = None) -> List[str]:
    results = []
    for name, meta in BENCHMARK_REGISTRY.items():
        match_cat = (category is None) or (meta["category"] == category)
        match_scale = (scalable_only is None) or (meta["scalable"] == scalable_only)
        
        if match_cat and match_scale:
            results.append(name)
    return results
