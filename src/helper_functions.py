def calculate_ghz_fidelity(counts: dict, shots: int, num_qubits: int) -> float:
    """Rastojanje/fidelitet za GHZ stanje (|00...0> + |11...1>)"""
    ideal_state_0 = "0" * num_qubits
    ideal_state_1 = "1" * num_qubits

    correct_counts = counts.get(ideal_state_0, 0) + counts.get(ideal_state_1, 0)
    return correct_counts / shots


def calculate_tvd(counts_ideal: dict, counts_noisy: dict, shots: int) -> float:
    """Total Variation Distance (TVD) između idealne i šumovite raspodele"""
    all_keys = set(counts_ideal.keys()).union(set(counts_noisy.keys()))
    tvd = 0.0
    for k in all_keys:
        p_ideal = counts_ideal.get(k, 0) / shots
        p_noisy = counts_noisy.get(k, 0) / shots
        tvd += abs(p_ideal - p_noisy)
    return 0.5 * tvd