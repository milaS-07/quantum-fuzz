##Problem Statement

1. * [Pipeline](#pipeline)
    - tok simulacije: ulazno kvantno kolo $\rightarrow$ model šuma $\rightarrow$ merenje izlaza
2. ****
   - 
3. ****
   - 
4. **Optimizator**
   - Testiranje efikasnosti fuzzera nad kolima koja su prethodno optimizovana

## Pipeline

### Model šuma

Za simulaciju šuma korišćen je model koji kombinuje **termalnu relaksaciju** ($T_1/T_2$), koja zavisi od vremena i **depolarizacionu grešku**, koja se javlja kod izvršavanja kapija. Model je opisan u radu [A methodology to select and adjust quantum noise models through emulators: benchmarking against real backends](docs/paper1.md)

**Parametri** šuma preuzeti su sa trenutne kalibracije IBM procesora (**ibm_sherbrooke**), na način kako je to urađeno u radu [Enhanced Algorithmic Perfect State Transfer on IBM Quantum Computers](docs/paper2.pdf).

| Parametar | Vrednost |
| :--- | :--- |
| **Procesor** | `ibm_sherbrooke` |
| **Model šuma** | `Thermal Relaxation` + `Depolarizing` |
| **$T_1$** | $289.55\ \mu\text{s}$ ($2.8955 \times 10^{-4}\ \text{s}$) |
| **$T_2$** | $186.01\ \mu\text{s}$ ($1.8601 \times 10^{-4}\ \text{s}$) |
| **Trajanje 1Q kapije ($t_{\text{1Q}}$)** | $42.67\ \text{ns}$ ($4.2667 \times 10^{-8}\ \text{s}$) |
| **Trajanje 2Q kapije ($t_{\text{2Q}}$)** | $539.90\ \text{ns}$ ($5.3990 \times 10^{-7}\ \text{s}$) |
| **Greška 1Q kapije** | $0.042\%$ ($0.000417$) |
| **Greška 2Q kapije** | $7.20\%$ ($0.072047$) |