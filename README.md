## Model šuma

Za simulaciju šuma korišćen je model koji kombinuje **termalnu relaksaciju** ($T_1/T_2$), koja zavisi od vremena i **depolarizacionu grešku**, koja se javlja kod izvršavanja kapija. Model je opisan u radu [A methodology to select and adjust quantum noise models through emulators: benchmarking against real backends](docs/paper1.md)

**Parametri** šuma preuzeti su sa trenutne kalibracije IBM procesora (**ibm_sherbrooke**), na način kako je to urađeno u radu [Enhanced Algorithmic Perfect State Transfer on IBM Quantum Computers](docs/paper2.md).

| | Model1 |
| :--- | :--- |
| **Procesor** | `ibm_sherbrooke` |
| **Model šuma** | `Thermal Relaxation` + `Depolarizing` |
| **$T_1$** | $289.55\ \mu\text{s}$ |
| **$T_2$** | $186.01\ \mu\text{s}$ |
| **Trajanje 1Q kapije ($t_{\text{1Q}}$)** | $42.67\ \text{ns}$ |
| **Trajanje 2Q kapije ($t_{\text{2Q}}$)** | $539.90\ \text{ns}$ |
| **Greška 1Q kapije** | 0.042% |
| **Greška 2Q kapije** | 7.20% |

## Testiranje nad benchmark-om

Korišćen benchmark je **MQT Bench** opisan u radu [MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing](docs/paper4.md).

Kola su podeljena u četiri kategorije:

1. **Parametrizovana kvantna kola (PQC):**
   * *Kola:* Efficient SU2, Real Amplitudes, Two Local, QAOA, QNN
   * *Karakteristike:* Kola sa bar jednom kapijom bez fiksne vrednosti rotacije
2. **Klifordova kola (Clifford):**
   * *Kola:* GHZ State, Graph State
   * *Karakteristike:* Sastoje se isključivo od H, S, CNOT kapija; stvaraju maksimalnu zapletenost
3. **Neparametrizovana ne-Klifordova kola (Non-Param Non-Clifford):**
   * *Kola:* W-State, Deutsch-Jozsa, Grover, Quantum Walk, Shor, QFT, Entangled QFT, QPE (exact/inexact), Amplitude Estimation
   * *Karakteristike:* Klasični kvantni algoritmi sa T-kapijama i kontrolisanim rotacijama
4. **Kontrolna grupa (Control):**
   * *Kola:* Random Circuit
   * *Karakteristike:* Nasumična struktura kapija

---

### Metrike Evaluacije

* **Fidelity ($F$):** Meri preklapanje dve distribucije verovatnoće
  $$F(P, Q) = \left( \sum_{x} \sqrt{P(x) \cdot Q(x)} \right)^2$$
  *Vrednost 1 označava savršeno poklapanje, 0 označava da nema nikakvog preklapanja*

* **Total Variation Distance (TVD):** Meri procenat greške
  $$\text{TVD}(P, Q) = \frac{1}{2} \sum_{x} |P(x) - Q(x)|$$
  *Vrednost 0 označava identične distribucije, dok 1 označava potpuno različite*

* **Jensen-Shannon Divergence (JSD):** Meri sličnost između dve distribucije verovatnoće
  $$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$

---

### Rezultati

#### Rezultati svih kola

| Kategorija | Fidelity ($F$) `[Mean ± Std]` | TVD `[Mean ± Std]` | JSD `[Mean ± Std]` |
| :--- | :---: | :---: | :---: |
| **Clifford** | 0.5249 ± 0.3426 | 0.4704 ± 0.3057 | 0.3685 ± 0.3360 |
| **Non-Clifford Non-Parametric** | 0.3398 ± 0.3650 | 0.6657 ± 0.3426 | 0.5806 ± 0.3817 |
| **PQC** | 0.4341 ± 0.3934 | 0.5585 ± 0.3294 | 0.4737 ± 0.3748 |
| **Control** | 0.1526 ± 0.2334 | 0.8251 ± 0.2072 | 0.7459 ± 0.2621 |

---

#### Rezultati po broju kjubita

| Broj kubita ($N$) | Fidelity ($F$) `[Mean ± Std]` | TVD `[Mean ± Std]` | JSD `[Mean ± Std]` |
| :---: | :---: | :---: | :---: |
| **2** | 0.9492 ± 0.0574 | 0.0787 ± 0.0659 | 0.0284 ± 0.0328 |
| **4** | 0.7467 ± 0.2570 | 0.3104 ± 0.2359 | 0.1768 ± 0.2004 |
| **6** | 0.6172 ± 0.3099 | 0.4475 ± 0.2778 | 0.2908 ± 0.2780 |
| **8** | 0.4616 ± 0.3103 | 0.5693 ± 0.2745 | 0.4276 ± 0.3152 |
| **10** | 0.2672 ± 0.2191 | 0.6821 ± 0.2256 | 0.5818 ± 0.2726 |
| **12** | 0.1421 ± 0.1967 | 0.7900 ± 0.1900 | 0.7322 ± 0.2503 |
| **14** | 0.0869 ± 0.1644 | 0.8731 ± 0.1604 | 0.8310 ± 0.2282 |
| **16** | 0.0835 ± 0.1533 | 0.8933 ± 0.1516 | 0.8471 ± 0.2289 |
| **18** | 0.0648 ± 0.1363 | 0.9251 ± 0.1348 | 0.8862 ± 0.2108 |

---

#### Rezultati po dubini kola

| Opseg dubine ($D$) | Fidelity ($F$) `[Mean ± Std]` | TVD `[Mean ± Std]` | JSD `[Mean ± Std]` |
| :---: | :---: | :---: | :---: |
| **$D \le 10$** | 0.9081 ± 0.0739 | 0.0957 ± 0.0698 | 0.0488 ± 0.0406 |
| **$11 - 30$** | 0.6375 ± 0.3089 | 0.3752 ± 0.2585 | 0.2595 ± 0.2687 |
| **$31 - 60$** | 0.3754 ± 0.3287 | 0.6290 ± 0.2542 | 0.5037 ± 0.3168 |
| **$61 - 100$** | 0.2525 ± 0.2590 | 0.7277 ± 0.2058 | 0.6158 ± 0.2730 |
| **$D > 100$** | 0.0252 ± 0.0589 | 0.9619 ± 0.0632 | 0.9329 ± 0.1101 |

---

#### Promena metrika pri skaliranju skalabilnih kola

| Kolo | Promena Fidelity-ja ($\Delta F$) | Promena TVD-a ($\Delta \text{TVD}$) | Promena JSD-a ($\Delta \text{JSD}$) |
| :--- | :---: | :---: | :---: |
| **Clifford** | | | |
| GHZ State | -0.5747 | +0.5720 | +0.3838 |
| Graph State | -0.9937 | +0.9340 | +0.9925 |
| **Non-Clifford Non-Parametric** | | | |
| Amplitude Estimation (AE) | -0.9988 | +0.9800 | +0.9991 |
| Deutsch-Jozsa | -0.5680 | +0.5680 | +0.3820 |
| Entangled QFT | -0.9090 | +0.8880 | +0.9471 |
| Grover's | -0.9328 | +0.9330 | +0.9655 |
| Quantum Fourier Transformation (QFT) | -0.9977 | +0.9510 | +0.9943 |
| Quantum Phase Estimation (QPE) exact | -0.9440 | +0.9440 | +0.9714 |
| Quantum Phase Estimation (QPE) inexact | -0.9989 | +0.9670 | +0.9992 |
| Quantum Walk | -0.2677 | +0.3130 | +0.4716 |
| W-State | -0.6929 | +0.6890 | +0.5361 |
| **PQC** | | | |
| Efficient SU2 ansatz with Random Parameters | -0.9435 | +0.9140 | +0.9579 |
| Quantum Approximation Optimization Algorithm (QAOA) | -0.9958 | +0.9460 | +0.9920 |
| Quantum Neural Network (QNN) | -0.9778 | +0.8480 | +0.8618 |
| Real Amplitudes ansatz with Random Parameters | -0.9499 | +0.8470 | +0.9413 |
| Two Local ansatz with random parameters | -0.9527 | +0.8750 | +0.9664 |
| **Control** | | | |
| Random Circuit | -0.7649 | +0.7020 | +0.8574 |
| **Prosek svih kola** | -0.8507 | +0.8159 | +0.8364 |