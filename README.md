# Pipeline
1. model šuma/simulacija kvantnog računara 
2. transformacija kola
3. prediktor šuma
4. kretanje kroz graf (pronalazak optimizacije)


## Model šuma

Za simulaciju šuma korišćen je model koji kombinuje **termalnu relaksaciju** ($T_1/T_2$), koja zavisi od vremena i **depolarizacione greške**, koja se javlja kod izvršavanja kapija. Model je opisan u radu [A methodology to select and adjust quantum noise models through emulators: benchmarking against real backends](docs/paper1.md)

**Parametri** šuma preuzeti iz snapshot-a kalibracije IBM procesora (**ibm_sherbrooke**), na način kako je to urađeno u radu [Enhanced Algorithmic Perfect State Transfer on IBM Quantum Computers](docs/paper2.md) - (jedne vrednosti za ceo čip, medijane grešaka kapija). Za razliku od tog rada, u kome je parametar depolarizacije podešavan prema eksperimentu, ovde je on izračunat iz kalibracionih podataka.

| | Model1 |
| :--- | :--- |
| **Procesor** | `ibm_sherbrooke` |
| **Model šuma** | `Thermal Relaxation` + `Depolarizing` |
| **$T_1$** (medijana) | $278.42\ \mu\text{s}$ |
| **$T_2$** (medijana) | $170.01\ \mu\text{s}$ |
| **Trajanje 1Q kapije ($t_{\text{1Q}}$)** | $56.89\ \text{ns}$ |
| **Trajanje 2Q kapije ($t_{\text{2Q}}$)** | $533.33\ \text{ns}$ |
| **Greška 1Q kapije ($r_{\text{1Q}}$)** | 0.0236% |
| **Greška 2Q kapije ($r_{\text{2Q}}$)** | 0.779% |
| **Parametar depolarizacije 1Q ($\lambda_{\text{1Q}}$)** | 0.0181% |
| **Parametar depolarizacije 2Q ($\lambda_{\text{2Q}}$)** | 0.6053% |
| **Datum snapshot-a** | 26.02.2025. |

**Izračunavanje $\lambda$. (verovatnoća da se stanje kjubita tokom kapije promeni potpuno nasumično u mešano stanje)** Za $n$ kjubita, $d = 2^n$:

$F_{\text{cilj}} = 1 - \frac{d+1}{d}\, r$, $\quad F_{\text{th}} = \left(\frac{1 + e^{-t/T_1} + 2e^{-t/T_2}}{4}\right)^{n}$, $\quad \lambda = \frac{F_{\text{th}} - F_{\text{cilj}}}{F_{\text{th}} - 1/d^2}$

Formule su navedene u radu (i izvedene od navedenih) [A simple formula for the average gate fidelity of a quantum dynamical operation](docs/paper8.md).


Kapija `rz` nije uključena u kalibraciju jer je virtuelna i nema ni trajanje ni grešku.

---

### Metrike evaluacije šuma

* **Fidelity ($F$):** Meri preklapanje dve distribucije verovatnoće
  $$F(P, Q) = \left( \sum_{x} \sqrt{P(x) \cdot Q(x)} \right)^2$$
  *Vrednost 1 označava savršeno poklapanje, 0 označava da nema nikakvog preklapanja*

* **Total Variation Distance (TVD):** Meri procenat greške
  $$\text{TVD}(P, Q) = \frac{1}{2} \sum_{x} |P(x) - Q(x)|$$
  *Vrednost 0 označava identične distribucije, dok 1 označava potpuno različite*

* **Jensen-Shannon Divergence (JSD):** Meri sličnost između dve distribucije verovatnoće
  $$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$


## Benchmark MQT Bench

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

### Rezultati

[rezultati testiranja MQT benchmark-a nad opisanim modelom šuma](docs/results_mqtbench.md)

---
## Transformacije kola


### Test

### Rezultati

---
## Prediktori šuma

Na skupu od **8 različitih kola** (za 3, 6 i 9 kubita), generisano je do 30 varijanti po svakom kolu. Na ovom uzorku testirano je **5 različitih prediktora šuma/fidelitija**:

- **Ukupna dubina kola** (Total Depth)
- **Dubina 1q kapija** (1Q Depth)
- **Dubina 2q kapija** (2Q Depth)
- **Teorijska formula** (Eff) iz rada [Fidelity decay and error accumulation in random quantum circuits](docs/paper7.md)
- **EPA (Error Propagation Analysis)** — bazirano na nezavisnoj propagaciji greške kroz depolarizujuće kanale, opisan u radovima o proceni fidelitija velikih kola (npr. [Quantum supremacy using a programmable superconducting processor](https://www.nature.com/articles/s41586-019-1666-5))

Za procenu uspešnosti predviđanja, izračunato je nekoliko mera:

- **Pirsonov koeficijent (r)** — meri koliko dobro se odnos dva niza podataka uklapa u *pravu liniju* (linearna veza). Vrednosti bliže 1 označavaju jaču vezu.
- **Spirmanov koeficijent (ρ)** — meri da li je odnos *monoton*, bez obzira na oblik krive.
- **Kendalov koeficijent (τ)** — slično Spirmanu, meri redosledno poklapanje (broj usaglašenih parova) i otporniji je na greške.
- **Top-1 Rank i Avg Rank Error** — mere koliko dobro prediktor uspeva da identifikuje apsolutno najbolje kolo u grupi varijanti, odnosno koliko u proseku greši u pozicioniranju kola na rang listi.
- **MAE i RMSE** — mere apsolutnu grešku predviđanja (primenjivo samo na EPA i teorijsku formulu jer oni jedini direktno računaju vrednost fidelitija, dok dubina predstavlja samo cenu).

### Rezultati

Sve generisane varijante grupisane su po osnovnom kolu i broju kjubita (ukupno **24 nezavisne grupe**). Prikazane vrednosti u tabeli predstavljaju globalni prosek rezultata dobijenih unutar svake od grupa.

| Prediktor | Pirson r | Spirman ρ | Kendal τ | Avg Rank Err | Top-1 Rank | MAE | RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EPA** | **0.995** | **0.926** | **0.815** | **2.25** | **1.58** | **0.020** | **0.022** |
| Teorijska formula (Eff) | 0.956 | 0.638 | 0.558 | 7.49 | 2.40 | 0.211 | 0.215 |
| Ukupna dubina | 0.672 | 0.812 | 0.670 | 3.57 | 5.56 | - | - |
| Dubina 2-kubitnih gejtova | 0.939 | 0.623 | 0.544 | 7.53 | 2.40 | - | - |
| Dubina 1-kubitnih gejtova | 0.653 | 0.808 | 0.665 | 3.61 | 5.56 | - | - |

**EPA** pokazao se kao   najpouzdaniji prediktor u svim metrikama. Njegova linearna korelacija sa stvarnim kvantnim fidelitijem je gotovo savršena (Pirson r = 0.995), a greška u proceni (MAE) je izuzetno mala (0.020). Takođe, EPA najbolje pronalazi optimalno kolo, sa prosečnom pozicijom istinski najboljeg kola na procenjenom prvom mestu (Top-1 Rank = 1.58). Zbog ovih rezultata, EPA pristup je odabran kao glavni metod evaluacije u daljem algoritmu.

 
 