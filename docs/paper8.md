# A simple formula for the average gate fidelity of a quantum dynamical operation

Rad daje vezu između prosečnog fideliteta kapije $F_{\text{avg}}$, koja odgovara kalibracionoj grešci kapije ($r = 1 - F_{\text{avg}}$), i fidelitetu procesa $F_{\text{pro}}$ (u radu: *entanglement fidelity*, $F_e$). Vezu su prvi dali Horodecki et al. (Phys. Rev. A 60, 1888 (1999)), a Nielsen daje jednostavniji dokaz.


1. **Jednačina (3):** $F_{\text{avg}} = \frac{d\,F_{\text{pro}} + 1}{d + 1}$. Koristi se za pretvaranje kalibracione greške kapije u fidelnost procesa, $F_{\text{cilj}} = 1 - \frac{d+1}{d}\,r$, koja je potrebna za računanje parametra depolarizacije $\lambda$.
2. **Jednačina (16):** $F_{\text{pro}} = \frac{1}{d^3}\sum_j \mathrm{tr}\left(U_j^\dagger\,\mathcal{E}(U_j)\right)$ (preko Pauli baze). Iz nje je izvedena fidelnost termalne relaksacije $F_{\text{th}}$.

Izraz za $F_{\text{th}}$ i za $\lambda$ nisu iz rada, već su izvedeni iz ove veze.

---
[*link do rada*](https://arxiv.org/abs/quant-ph/0205035)