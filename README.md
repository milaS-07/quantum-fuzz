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

## Transformacije kola

Optimizator iz polaznog kola generiše susedna kola ekvivalentna po unitarnoj matrici (do globalne faze). Koriste se tri transformacije:

- **Resinteza bloka**: blok od najviše 3 kjubita (`CollectMultiQBlocks`) zamenjuje se novim kolom iste unitarne matrice. Za 2 kjubita koristi se **KAK dekompozicija**, a za 3 kjubita **kvantna Šenonova dekompozicija** (`qs_decomposition`).
- **Resinteza prozora**: odsečak između dve susedne 2Q kapije sa zajedničkim kjubitom (najviše 8 pozicija) zamenjuje se resintetizovanim kolom na 2 ili 3 kjubita.
- **Peephole**: uklanja se par susednih istih samoinverznih kapija jer oni ne rade ništa (`x`, `y`, `z`, `h`, `cx`, `cz`).

### Test

Svako transformisano kolo poredi se sa originalom (beam varijante) ili sa roditeljem u nasumičnom lancu od 6 koraka (random walk), a lanac se dodatno poredi sa originalom. Za kola do 6 kjubita poredi se cela unitarna matrica, a za veća dejstvo na 3 slučajna stanja. Rastojanje je $\max|b - e^{i\varphi}a|$, $\varphi = \arg\langle a|b\rangle$, a tolerancija je $10^{-7}$.

Osetljivost testa proverena je mutacionim testiranjem: u optimizator je namerno ubačeno 8 grešaka (izostavljena kapija, dodatna rotacija, zamenjene kontrola i meta, `cx` → `cz`, pogrešno mesto bloka, obrnut redosled kjubita u prozoru, peephole nad različitim kjubitima, promena samo globalne faze).

| | |
| :--- | :--- |
| **Benchmark kola** | 205 (5 slučajnih + MQT Bench, algoritamski i nativni oblik), 3-10 kjubita, 3-643 kapije |
| **Generisana kola** | 5 familija (`random2q`, `random3q`, `staggered`, `redundant`, `structured`), 3-8 kjubita, ~150 po seedu |
| **Seedovi** | 30 |
| **Tolerancija** | $10^{-7}$ |

Za ovaj test nije testirano više od 10 kjubita jer blok i prozor obuhvataju najviše 3 kjubita, a peephole 1 do 2. Ispravnost transformacije zavisi samo od kjubita na kojima ona deluje i od mesta na koje se rezultat vraća u kolo, dok ostali kjubiti ostaju netaknuti. Zato bi veći broj kjubita povećao samo cenu provere ($4^n$ za unitarnu matricu), a ne i raznovrsnost grešaka. Mutaciono testiranje pokazuje da se greške ovog tipa otkrivaju već na 3 do 10 kjubita.

### Rezultati
| Transformacija | Poređenja |
| :--- | ---: |
| beam | 577 745 |
| blok (1Q / 2Q / 3Q) | 12 837 / 56 948 / 286 929 |
| prozor (2Q / 3Q) | 471 071 / 1 158 537 |
| peephole | 16 126 |
| kumulativno | 217 271 |
| **Ukupno** | **2 797 464** |

### Zaključak

Od 4705 nezavisnih kola, 4704 su obrađena do kraja i nijedno nije pokazalo neekvivalentnost: u 2 797 464 poređenja najveće odstupanje je $9.9 \cdot 10^{-9}$, ispod tolerancije od $10^{-7}$. Sa 95% poverenja, verovatnoća neekvivalentne transformacije po kolu je manja od $1 - 0.05^{1/4704} \approx 0.064\%$ za raspodelu kola korišćenu u testu. Svih 8 namerno ubačenih grešaka otkriveno je u svih 30 seedova (240 od 240), pa je test dovoljno osetljiv da uoči greške ovog tipa. Pored toga, `best_circuit` je ekvivalentan ulaznom kolu u svih 4020 pokretanja `optimize_circuit`. Na osnovu toga, nema indicija da su transformacije neispravne, pa se mogu koristiti u pretrazi.

**Jedini neuspeh (seed 10).** Kolo `staggered_081_n7` nije obrađeno do kraja jer je Qiskit u funkciji `qs_decomposition` (sinteza bloka od 3 kjubita) prijavio internu grešku (`TwoQubitWeylDecomposition: failed to diagonalize M2`). Ona je nastala pri generisanju trensformacija polaznog kola, pre bilo kakvog poređenja, pa nijedno transformisano kolo nije proizvedeno niti proglašeno neekvivalentnim. Neuspeh je nastao zbog same biblioteke i nije pogrešna transformacija.

---
## Prediktori šuma

Na skupu od **17 kola** (scalable benchmarks) generisano je do 30 varijanti po svakom kolu, za veličine od 3 do 16 kubita. Stvarni fideliti varijanti dobijen je simulacijom sa modelom šuma: za kola od 3 do 9 kubita korišćena je egzaktna simulacija matrice gustine (bez statističkog šuma), a za kola od 10 do 16 kubita Monte Karlo simulacija trajektorija sa standardnom greškom (SE) do 0.015. Na ovom uzorku testirana su **4 različita prediktora šuma**:

- **Ukupna dubina kola** (Total Depth)
- **Dubina 2q kapija** (2Q Depth)
- **Teorijska formula** (RQC) iz rada [Fidelity decay and error accumulation in random quantum circuits](docs/paper7.md)
- **ESP (Estimated Success Probability)** — bazirano na nezavisnoj propagaciji greške kroz depolarizujuće kanale, opisan u radovima o proceni fidelitija velikih kola [Extracting Success from IBM's 20-Qubit Machines Using Error-Aware Compilation
](docs/paper9)

Za procenu uspešnosti predviđanja, izračunato je nekoliko mera:

- **Pirsonov koeficijent (r)** — meri koliko dobro se odnos dva niza podataka uklapa u *pravu liniju* (linearna veza). Vrednosti bliže 1 označavaju jaču vezu.
- **Spirmanov koeficijent (ρ)** — meri da li je odnos *monoton*, bez obzira na oblik krive.
- **Kendalov koeficijent (τ)** — slično Spirmanu, meri redosledno poklapanje (broj usaglašenih parova) i otporniji je na greške.
- **Top-1 Rank, Avg Rank Error i Regret** — mere koliko dobro prediktor uspeva da identifikuje apsolutno najbolje kolo u grupi varijanti, odnosno koliko u proseku greši u pozicioniranju kola na rang listi. Avg Rank Error je normalizovan (0 – savršen poredak), a Regret je razlika između stvarnog fidelitija najbolje varijante i varijante koju je prediktor izabrao.
- **Pairwise Accuracy, Strict Accuracy i Tie Rate** — mere koliko često prediktor ispravno poređa par varijanti. U obzir se uzimaju samo parovi čiji se stvarni fideliti razlikuju za više od 2·SE (najmanje 10⁻⁴). Pairwise Accuracy računa izjednačenja kao pola tačnog, Strict Accuracy ih izuzima, a Tie Rate je udeo parova koje prediktor ne može da razdvoji. Uz njih je dat i 95% interval poverenja (bootstrap po osnovnim kolima).
- **MAE i RMSE** — mere apsolutnu grešku predviđanja (primenjivo samo na ESP i teorijsku formulu jer oni jedini direktno računaju vrednost fidelitija, dok dubina predstavlja samo cenu).

### Rezultati

Sve generisane varijante grupisane su po osnovnom kolu i broju kjubita (ukupno **174 grupe**, **55 005 poređenih parova**). Grupe u kojima nijedan par varijanti nema dovoljno različit stvarni fideliti odbačene su kao neinformativne (2 grupe). Prikazane vrednosti u tabelama predstavljaju globalni prosek rezultata dobijenih unutar svake od grupa, osim mera na nivou parova (Pairwise i Strict Accuracy, Tie Rate), koje su izračunate nad svim parovima zajedno.

| Prediktor | Pirson r | Spirman ρ | Kendal τ | Avg Rank Err | Top-1 Rank | Regret | MAE | RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ESP** | **0.889** | **0.837** | **0.754** | **0.123** | **2.02** | **0.0029** | **0.022** | **0.024** |
| Teorijska formula (RQC) | 0.842 | 0.811 | 0.720 | 0.156 | 2.70 | 0.0034 | 0.343 | 0.347 |
| Ukupna dubina | 0.842 | 0.784 | 0.681 | 0.135 | 2.39 | 0.0040 | - | - |
| Dubina 2-kjubitnih kapija | 0.848 | 0.803 | 0.713 | 0.152 | 2.61 | 0.0034 | - | - |

| Prediktor | Pairwise Acc | 95% CI | Tie Rate | Strict Acc | 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ESP** | **0.935** | **[0.924, 0.945]** | 0.085 | **0.975** | **[0.968, 0.981]** |
| Teorijska formula (RQC) | 0.901 | [0.884, 0.917] | 0.141 | 0.967 | [0.953, 0.978] |
| Ukupna dubina | 0.905 | [0.885, 0.923] | 0.059 | 0.931 | [0.912, 0.947] |
| Dubina 2-kjubitnih kapija | 0.901 | [0.885, 0.917] | 0.139 | 0.966 | [0.953, 0.978] |


ESP je izabran jer najbolje pronalazi varijantu sa najmanje šuma (najmanja greška rangiranja i regret u oba opsega veličina kola) i jedini uz to daje i procenu apsolutne vrednosti fidelitija (MAE = 0.022). Kod većih kola izabrana varijanta je u proseku 3 – 4 najbolja, ali gubi manje od 1 procentnog poena fidelitija u odnosu na najbolju.





posle će trebati:

#### Koliko najboljih predloga treba proveriti


| Kjubita | Ekvivalentnih najboljoj (od 30) | Top-1 | Top-3 | Top-5 | k za 95% |
|:---|:---:|:---:|:---:|:---:|:---:|
| 3 | 1.3 | 0.65 | 1.00 | 1.00 | 3 |
| 4 | 1.6 | 0.53 | 0.94 | 1.00 | 5 |
| 5 | 1.5 | 0.71 | 1.00 | 1.00 | 3 |
| 6 | 1.5 | 0.59 | 1.00 | 1.00 | 3 |
| 7 | 1.7 | 0.71 | 1.00 | 1.00 | 3 |
| 8 | 1.4 | 0.67 | 0.93 | 1.00 | 4 |
| 9 | 1.5 | 0.67 | 1.00 | 1.00 | 3 |
| 10 | 9.1 | 0.67 | 1.00 | 1.00 | 2 |
| 12 | 9.8 | 0.67 | 1.00 | 1.00 | 2 |
| 14 | 12.4 | 0.64 | 1.00 | 1.00 | 2 |
| 16 | 16.5 | 0.73 | 1.00 | 1.00 | 2 |
| **Ukupno** | **5.0** | **0.66** | **0.99** | **1.00** | **3** |
