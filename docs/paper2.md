# Enhanced Algorithmic Perfect State Transfer on IBM Quantum Computers

Rad se bavi prenosom kvantnog stanja (PST) kroz lanac od 4 kubita na IBM procesorima (`ibm_sherbrooke` i `ibm_brisbane`).

## Konstrukcija modela šuma
Parametre modela šuma su izveli kombinovanjem IBM-ovih kalibracionih merenja i strukture samog kvantnog kola:

* **Vremena koherencije ($T_1, T_2$) i trajanje kapija ($t_{\text{gate}}$):** Očitana su sa IBM kalibracionog fajla i prosređena u `thermal_relaxation_error`, koji po formuli $e^{-t_{\text{gate}}/T_{1,2}}$ računa fizički raspad stanja tokom trajanja svake kapije
* **Greške kapija (`gate_error`):** Medijane grešaka sa čipa pomnožili su sa brojem kapija po Trotter koraku (6 dvokubitnih kapija po koraku) i preveli ih u parametar depolarizacije ($q = 4p/3$)

---
[*link do rada*](https://arxiv.org/abs/2508.18626)