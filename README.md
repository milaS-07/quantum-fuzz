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

[rezultati testiranja MQT benchmark-a nad šumom](results_mqtbench.md)

---
## Korelacija različitih parametara i šuma

Kako bi se proces prolaska kroz graf ubrzao, urađeni su testovi posmatranja različitih parametara po tome kolika postoji koleracija između njih i šuma.
 
Na uzorku od ukupno **456 kola**, testirane su 3 metrike:
 
- **Ukupna dubina kola**
- **Dubina 2q kapija**
- **Fedelity formula** iz rada [Fidelity decay and error accumulation in random quantum circuits](docs/paper7.md)

Za svaku metriku izračunate su dve mere korelacije:
 
- **Pirsonov koeficijent (r)** — meri koliko dobro se odnos dva niza podataka
  uklapa u *pravu liniju* (koliko je linearan odnos). Vrednosti idu od -1 do 1; što je bliže 1 (ili -1),
  to je linearna veza jača.
- **Spirmanov koeficijent (ρ)** — meri da li je odnos *monoton* (dosledan po
  rangu), bez obzira na oblik krive. Pogodan je kada se sumnja da veza nije
  linearna, već npr. eksponencijalna.

Uz oba koeficijenta izračunata je i **p-vrednost**, koja pokazuje da li je
dobijena korelacija statistički značajna (da li je verovatno da je nastala
slučajno).
 
## Rezultati
 
Svih 456 varijanti je grupisano po
  osnovnom kolu, i unutar svake grupe je izračunata prosečna vrednost, čime
  se dobija **21 nezavisan red podataka**.

| Prediktor                  | Pirson r | p (Pirson) | Spirman ρ | p (Spirman) |
|-----------------------------|:--------:|:----------:|:---------:|:-----------:|
| Ukupna dubina               | -0.37    | 0.098      | -0.49     | 0.025       |
| Dubina 2-kubitnih gejtova   | -0.39    | 0.083      | -0.54     | 0.011       |
| **Teorijska formula (rad)** | **0.48** | 0.026      | **0.60**  | 0.0044      |
 
Formula iz rada je statistički najznačajnija i dobija se najbolji rezultat u obe metrike pa je zato i korišćena u algoritmu.
 
## Zaključak i izbor formule
 
Teorijska formula iz rada pokazala se kao **ubedljivo najbolji prediktor**
empirijske vernosti — znatno bolji od prostih strukturnih metrika poput
dubine kola. To je i očekivano, s obzirom da formula nije proizvoljna
heuristika, već analitički izvedena iz fizičkog modela akumulacije grešaka
(nasumični 2-kubitni gejtovi + permutacije), dok dubina kola samo posredno
prati taj proces.
 
Na osnovu ovoga, formula je usvojena kao brza zamena za pravu simulaciju
tokom pretrage/optimizacije kola — s tim da se finalni rezultat i dalje
proverava stvarnom simulacijom, jer ni r = 0.64 ne garantuje tačnost za
pojedinačan slučaj, već samo dobar prosečan trend.
 