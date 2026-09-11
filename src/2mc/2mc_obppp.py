import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import List, Tuple, Dict, Sequence, Optional


I_, X_, Y_, Z_ = 0, 1, 2, 3
_LBL = "IXYZ"

_MULT: Dict[Tuple[int, int], Tuple[int, int]] = {}
for _a in range(4):
    _MULT[(I_, _a)] = (_a, 0)
    _MULT[(_a, I_)] = (_a, 0)
_MULT[(X_, X_)] = (I_, 0); _MULT[(Y_, Y_)] = (I_, 0); _MULT[(Z_, Z_)] = (I_, 0)
_MULT[(X_, Y_)] = (Z_, 1); _MULT[(Y_, X_)] = (Z_, 3)
_MULT[(Y_, Z_)] = (X_, 1); _MULT[(Z_, Y_)] = (X_, 3)
_MULT[(Z_, X_)] = (Y_, 1); _MULT[(X_, Z_)] = (Y_, 3)

@dataclass
class PauliWord:
    labels: np.ndarray
    phase: int = 0 

    @staticmethod
    def identity(n: int) -> "PauliWord":
        return PauliWord(np.zeros(n, dtype=np.int8), 0)

    def copy(self) -> "PauliWord":
        return PauliWord(self.labels.copy(), self.phase)

    def commutes_with(self, other: "PauliWord") -> bool:
        diff = (self.labels != other.labels) & (self.labels != I_) & (other.labels != I_)
        return int(diff.sum()) % 2 == 0

    def multiply(self, other: "PauliWord") -> "PauliWord":
        n = len(self.labels)
        new_labels = np.zeros(n, dtype=np.int8)
        extra_phase = 0
        for q in range(n):
            res_label, phase_inc = _MULT[(int(self.labels[q]), int(other.labels[q]))]
            new_labels[q] = res_label
            extra_phase += phase_inc
        return PauliWord(new_labels, (self.phase + other.phase + extra_phase) % 4)

    def local_index(self, qubits: Sequence[int]) -> int:
        idx = 0
        for q in qubits:
            idx = idx * 4 + int(self.labels[q])
        return idx

    def is_diagonal_in_Z(self) -> bool:
        return bool(np.all((self.labels == I_) | (self.labels == Z_)))



def local_ptm_row_matrix(kraus_ops: List[np.ndarray], k: int) -> np.ndarray:
    d = 2**k
    num_paulis = 4**k
    T = np.zeros((num_paulis, num_paulis), dtype=float)
    pauli_labels = list(iproduct(range(4), repeat=k))
    
    def get_matrix(labels):
        m = np.array([[1.0]], dtype=complex)
        p_mats = {
            I_: np.eye(2), X_: np.array([[0,1],[1,0]]), 
            Y_: np.array([[0,-1j],[1j,0]]), Z_: np.array([[1,0],[0,-1]])
        }
        for l in labels:
            m = np.kron(m, p_mats[l])
        return m

    mats = [get_matrix(l) for l in pauli_labels]

    for out_idx in range(num_paulis):
        P_out = mats[out_idx]
        E_adj_Pout = sum(K.conj().T @ P_out @ K for K in kraus_ops)
        for in_idx in range(num_paulis):
            T[out_idx, in_idx] = np.trace(mats[in_idx] @ E_adj_Pout).real / d
            
    return T


@dataclass
class Gate:
    qubits: Tuple[int, ...]
    generator: Optional[Tuple[int, ...]] = None
    param_idx: Optional[int] = None
    clifford_ptm: Optional[np.ndarray] = None

@dataclass
class PQC:
    n_qubits: int
    gates: List[Gate]
    noise_1q: np.ndarray 
    noise_2q: np.ndarray 
    noise_override: Dict[int, np.ndarray] = field(default_factory=dict)


def sample_backprop(T_row: np.ndarray, rng: np.random.Generator):
    weights = np.abs(T_row)
    total_w = weights.sum()
    if total_w < 1e-12: return None
    
    r = rng.random() * total_w
    cum = 0.0
    for idx, w in enumerate(weights):
        cum += w
        if r <= cum:
            return idx, total_w * np.sign(T_row[idx])
    return None

def _propagate_sample(pqc: PQC, theta: np.ndarray, start_word: PauliWord,
                       start_coeff: complex, rng: np.random.Generator,
                       apply_noise: bool) -> complex:
    word = start_word.copy()
    coeff = start_coeff

    for gidx in range(len(pqc.gates) - 1, -1, -1):
        gate = pqc.gates[gidx]

        if apply_noise:
            T = pqc.noise_override.get(gidx, pqc.noise_1q if len(gate.qubits) == 1 else pqc.noise_2q)
            l_idx = word.local_index(gate.qubits)
            res = sample_backprop(T[l_idx], rng)
            if res is None: return 0.0
            new_l_idx, weight = res
            coeff *= weight
            tmp_idx = new_l_idx
            for q in reversed(gate.qubits):
                word.labels[q] = tmp_idx % 4
                tmp_idx //= 4

        if gate.generator is not None:
            th = theta[gate.param_idx]
            gword = PauliWord(np.zeros(pqc.n_qubits, dtype=np.int8), 0)
            for q, l in zip(gate.qubits, gate.generator):
                gword.labels[q] = l
            
            if not word.commutes_with(gword):
                c, s = np.cos(th), np.sin(th)
                if rng.random() < abs(c) / (abs(c) + abs(s)):
                    coeff *= np.sign(c) * (abs(c) + abs(s))
                else:
                    word = gword.multiply(word)
                    coeff *= -1j * np.sign(s) * (abs(c) + abs(s))
        elif gate.clifford_ptm is not None:
            l_idx = word.local_index(gate.qubits)
            row = gate.clifford_ptm[l_idx]
            new_l_idx = np.argmax(np.abs(row))
            coeff *= row[new_l_idx]
            tmp_idx = new_l_idx
            for q in reversed(gate.qubits):
                word.labels[q] = tmp_idx % 4
                tmp_idx //= 4

    if word.is_diagonal_in_Z():
        return coeff * (1j ** word.phase)
    return 0.0


def exact_noiseless_expectation(pqc: PQC, theta: np.ndarray, obs_terms) -> float:
    rng = np.random.default_rng(0)
    weights = np.array([abs(c) for c, _ in obs_terms], dtype=float)
    norm1 = weights.sum()
    if norm1 < 1e-12: return 0.0
    
    total = 0.0
    for c_h, P_h in obs_terms:
        res = _propagate_sample(pqc, theta, P_h, np.sign(c_h) * norm1, rng, apply_noise=False)
        total += res.real
    return total

def mc_noisy_expectation(pqc: PQC, theta: np.ndarray, obs_terms,
                         n_samples: int, rng: np.random.Generator) -> float:
    weights = np.array([abs(c) for c, _ in obs_terms], dtype=float)
    norm1 = weights.sum()
    if norm1 < 1e-12: return 0.0
    
    probs = weights / norm1
    total = 0.0
    for _ in range(n_samples):
        h = rng.choice(len(obs_terms), p=probs)
        c_h, P_h = obs_terms[h]
        res = _propagate_sample(pqc, theta, P_h, np.sign(c_h) * norm1, rng, apply_noise=True)
        total += res.real
    return total / n_samples

def estimate_expressibility(pqc: PQC, n_params: int, n_theta_samples: int = 50, 
                            n_pauli_pairs: int = 50, n_inner_samples: int = 1,
                            apply_noise: bool = True, rng=None) -> float:
    if rng is None: rng = np.random.default_rng()
    n = pqc.n_qubits
    d = 2**n
    max_idx = 4**n
    total_m2 = 0.0
    
    for _ in range(n_pauli_pairs):
        idx = rng.integers(1, max_idx)
        p_word = PauliWord(np.zeros(n, dtype=np.int8))
        tmp_idx = idx
        for q in range(n):
            p_word.labels[q] = tmp_idx % 4
            tmp_idx //= 4
        
        obs = [(1.0, p_word)]
        haar_val = 1.0 / (d + 1)
        
        theta_sum = 0.0
        for _ in range(n_theta_samples):
            theta_a = rng.choice([0.0, np.pi/2, np.pi, 3*np.pi/2], size=n_params)
            theta_b = rng.choice([0.0, np.pi/2, np.pi, 3*np.pi/2], size=n_params)
            
            if apply_noise:
                v1_a = mc_noisy_expectation(pqc, theta_a, obs, n_inner_samples, rng)
                v2_a = mc_noisy_expectation(pqc, theta_a, obs, n_inner_samples, rng)
                v1_b = mc_noisy_expectation(pqc, theta_b, obs, n_inner_samples, rng)
                v2_b = mc_noisy_expectation(pqc, theta_b, obs, n_inner_samples, rng)
            else:
                v1_a = exact_noiseless_expectation(pqc, theta_a, obs)
                v2_a = exact_noiseless_expectation(pqc, theta_a, obs)
                v1_b = exact_noiseless_expectation(pqc, theta_b, obs)
                v2_b = exact_noiseless_expectation(pqc, theta_b, obs)
                
            y_a = v1_a * v2_a
            y_b = v1_b * v2_b
            theta_sum += (y_a - haar_val) * (y_b - haar_val)
            
        total_m2 += theta_sum / n_theta_samples

    return (max_idx - 1) * (total_m2 / n_pauli_pairs)

def estimate_mse(pqc: PQC, obs_terms, n_params: int,
                 n_theta_samples: int, n_inner_samples: int,
                 rng: np.random.Generator) -> float:
    total = 0.0
    for _ in range(n_theta_samples):
        theta = rng.choice([0.0, np.pi/2, np.pi, 3*np.pi/2], size=n_params)
        ideal = exact_noiseless_expectation(pqc, theta, obs_terms)
        y1 = mc_noisy_expectation(pqc, theta, obs_terms, n_inner_samples, rng)
        y2 = mc_noisy_expectation(pqc, theta, obs_terms, n_inner_samples, rng)
        total += (ideal * ideal) - (ideal * (y1 + y2)) + (y1 * y2)
    return total / n_theta_samples

def estimate_trainability(pqc: PQC, obs_terms, n_params: int,
                          n_theta_samples: int, n_inner_samples: int,
                          rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros(n_params)
    for _ in range(n_theta_samples):
        theta = rng.choice([0.0, np.pi/2, np.pi, 3*np.pi/2], size=n_params)
        g1, g2 = np.zeros(n_params), np.zeros(n_params)
        
        for k in range(n_params):
            th_p = theta.copy(); th_p[k] = (theta[k] + np.pi / 2) % (2 * np.pi)
            th_m = theta.copy(); th_m[k] = (theta[k] - np.pi / 2) % (2 * np.pi)
            
            yp1 = mc_noisy_expectation(pqc, th_p, obs_terms, n_inner_samples, rng)
            ym1 = mc_noisy_expectation(pqc, th_m, obs_terms, n_inner_samples, rng)
            g1[k] = 0.5 * (yp1 - ym1)
            
            yp2 = mc_noisy_expectation(pqc, th_p, obs_terms, n_inner_samples, rng)
            ym2 = mc_noisy_expectation(pqc, th_m, obs_terms, n_inner_samples, rng)
            g2[k] = 0.5 * (yp2 - ym2)
            
        acc += g1 * g2
    return acc / n_theta_samples

def _propagate_sample_with_override(pqc: PQC, theta: np.ndarray, start_word: PauliWord,
                                    start_coeff: complex, rng: np.random.Generator,
                                    override_gate_idx: int, override_T: np.ndarray) -> complex:
    word = start_word.copy()
    coeff = start_coeff
    for gidx in range(len(pqc.gates) - 1, -1, -1):
        gate = pqc.gates[gidx]
        if gidx == override_gate_idx:
            T = override_T
            l_idx = word.local_index(gate.qubits)
            weights = np.abs(T[l_idx])
            M = weights.sum()
            if M < 1e-15: return 0.0
            new_l_idx = rng.choice(len(T[l_idx]), p=(weights / M))
            coeff *= M * np.sign(T[l_idx][new_l_idx])
        else:
            T = pqc.noise_override.get(gidx, pqc.noise_1q if len(gate.qubits) == 1 else pqc.noise_2q)
            l_idx = word.local_index(gate.qubits)
            res = sample_backprop(T[l_idx], rng)
            if res is None: return 0.0
            new_l_idx, weight = res
            coeff *= weight
            
        tmp_idx = new_l_idx
        for q in reversed(gate.qubits):
            word.labels[q] = tmp_idx % 4
            tmp_idx //= 4

        if gate.generator is not None:
            th = theta[gate.param_idx]
            gword = PauliWord(np.zeros(pqc.n_qubits, dtype=np.int8), 0)
            for q, l in zip(gate.qubits, gate.generator):
                gword.labels[q] = l
            if not word.commutes_with(gword):
                c, s = np.cos(th), np.sin(th)
                if rng.random() < abs(c) / (abs(c) + abs(s)):
                    coeff *= np.sign(c) * (abs(c) + abs(s))
                else:
                    word = gword.multiply(word)
                    coeff *= -1j * np.sign(s) * (abs(c) + abs(s))
        elif gate.clifford_ptm is not None:
            l_idx = word.local_index(gate.qubits)
            row = gate.clifford_ptm[l_idx]
            new_l_idx = np.argmax(np.abs(row))
            coeff *= row[new_l_idx]
            tmp_idx = new_l_idx
            for q in reversed(gate.qubits):
                word.labels[q] = tmp_idx % 4
                tmp_idx //= 4

    if word.is_diagonal_in_Z():
        return coeff * (1j ** word.phase)
    return 0.0

def estimate_noise_bottleneck(pqc: PQC, obs_terms, n_params: int,
                              kraus_fn, base_param: float, delta: float,
                              n_theta_samples: int, n_inner_samples: int,
                              rng: np.random.Generator) -> Dict[int, float]:
    T_plus = local_ptm_row_matrix(kraus_fn(base_param + delta), 1)
    T_minus = local_ptm_row_matrix(kraus_fn(base_param - delta), 1)
    dT = (T_plus - T_minus) / (2 * delta)

    qubit_sensitivities = {q: 0.0 for q in range(pqc.n_qubits)}
    weights = np.array([abs(c) for c, _ in obs_terms], dtype=float)
    norm1 = weights.sum()
    
    for _ in range(n_theta_samples):
        theta = rng.choice([0.0, np.pi/2, np.pi, 3*np.pi/2], size=n_params)
        ideal = exact_noiseless_expectation(pqc, theta, obs_terms)
        y = mc_noisy_expectation(pqc, theta, obs_terms, n_inner_samples, rng)
        
        for gidx, gate in enumerate(pqc.gates):
            if len(gate.qubits) != 1: continue 
            
            dy = 0.0
            for _ in range(n_inner_samples):
                h = rng.choice(len(obs_terms), p=(weights / norm1))
                c_h, P_h = obs_terms[h]
                res = _propagate_sample_with_override(
                    pqc, theta, P_h, np.sign(c_h) * norm1, rng, gidx, dT
                )
                dy += res.real
            dy /= n_inner_samples
            
            sensitivity = -2.0 * (ideal - y) * dy
            qubit_sensitivities[gate.qubits[0]] += abs(sensitivity)
            
    return {q: val / n_theta_samples for q, val in qubit_sensitivities.items()}




def plot_heatmap(n_qubits, data_dict, title, grid_shape=(2, 2)):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    xs = [q % grid_shape[1] for q in range(n_qubits)]
    ys = [q // grid_shape[1] for q in range(n_qubits)]
    values = [data_dict.get(q, 0.0) for q in range(n_qubits)]
    
    sc = ax.scatter(xs, ys, c=values, cmap='coolwarm', s=500, edgecolors='black')
    
    for i, txt in enumerate(range(n_qubits)):
        ax.annotate(txt, (xs[i], ys[i]), ha='center', va='center', color='white', weight='bold')
        
    plt.colorbar(sc, label='Metric Value')
    ax.set_title(title)
    ax.set_xticks(range(grid_shape[1]))
    ax.set_yticks(range(grid_shape[0]))
    ax.invert_yaxis()
    plt.show()