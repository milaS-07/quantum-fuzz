# Diagnosing Quantum Circuits: Noise Robustness, Trainability, and Expressibility

Ovaj rad opisuje njihov razvijen alat koji procenjuje osetljivost kola na šum i optimizuje ih. Generiše prostorno-vremensku mapu hotspot-ova šuma.

Zbog precizne detekcije osetljivih mesta su primenili korekciju na manje od 2% kjubita.


## Model šuma
Koriste modele šuma **PCS1** (Pauli Column-wise Sum at most One).

### Uslovi koje šum mora da ispuni kako bi spadao pod PCS1

1. **Uslov ograničenja sume po kolonama:**
   Za Pauli matricu prenosa $S_{\mathcal{E}}$ danog kanala šuma $\mathcal{E}$ — čiji su elementi definisani kao $(S_{\mathcal{E}})_{i,j} = \text{tr}\{\mathcal{E}(\sigma_i)\sigma_j\}$ u normiranoj Pauli bazi — suma apsolutnih vrednosti elemenata u bilo kojoj koloni $j$ ne sme da pređe 1:
   $$\sum_i |(S_{\mathcal{E}})_{i,j}| \le 1 \quad \forall j$$

2. **Validnost stohastičke reprezentacije pri propagaciji unazad:**
   Koeficijenti u Pauli matrici prenosa moraju se moći mapirati na **validnu raspodelu verovatnoća** pri kretanju unazad u Hajzenbergovoj slici. Odnosno, šum ne sme generisati koeficijente čiji bi zbir pri Monte Karlo uzorkovanju staza uzrokovao eksploziju varijanse ili ne-fizičke verovatnoće.

3. **Očuvanje Hermitnosti i trag-očuvajući karakter (TPCP svojstva):**
   Kanal mora predstavljati fizički realističnu kvantnu operaciju (*Completely Positive Trace-Preserving*), čime se osigurava da pod uticajem šuma Pauli operatori prelaze u druge Hermitske operatore bez stvaranja imaginarnih verovatnoća.

4. **Lokalnost delovanja ($\mathcal{O}(1)$-lokalnost):**
   Kanal šuma mora delovati lokalno na fiksiran, konstantan broj kubita (najčešće 1 ili 2), tako da se PCS1 uslov proveri i zadovolji na nivou pojedinačnih lokalnih matricama prenosa malih dimenzija ($4 \times 4$ ili $16 \times 16$).

---
**Fizički kanali koji ispunjavaju ove uslove:** 
Depolarizujući šum, Pauli kanali grešaka, **Amplitude Damping** (gubitak energije), **Phase Damping / Thermal Relaxation** (dekoherencija) i merenja u sredini kola.

---

| Metrika | Definicija | Značenje |
| :--- | :--- | :--- |
| **Robusnost na šum** | $\text{MSE}(\langle O \rangle) = \mathbb{E}_\theta \left( \langle O \rangle_\theta - \langle O_e \rangle_\theta \right)^2$ | Prosečno odstupanje šumnog izlaza od idealnog izlaza kroz sve uglove $\theta$ |
| **Trenirativnost** | $\text{Var}_\theta \left[ \frac{\partial \langle O \rangle_\theta}{\partial \theta_k} \right]$ | Eksponencijalni pad varijanse gradijenta ka 0 ukazuje na pojavu *Barren Plateaus* (nemogućnost optimizacije) |
| **Ekspresivnost** | Odstupanje generisanog stanja od *Haar* raspodele | U *Haar* rasodeli za bilo koje $\text{Var}_\theta, postoji jednaka verovatnoća da će kjubit na kraju izvršavanja završiti u bilo kom stanju |

---

## 1. Robusnost

Mere štete (MSE) definišu kao razliku između idealnog izlaza i izlaza pod šumom.

## 2. Trenirativnost

Varijansa gradijenta

## 3. Ekspresivnost





---
## Beleške

* u realističnim kvantnim uređajima određene komponente kola su mnogo osetljivije na šum od drugih, u radi ta mesta navode kao *noise bottleneck*
* rad se bavi pre svega PQC-om
* kako bi PQC bio koristan mora da zadovoljava navedena tri kriterijuma (rubusnost, trenirativnost, eksresivnost)
* problem je što tačno pronalaženje bottleneck delova raste eksponencijalno sa brojem kjubita (oni to rešavaju koristeći Pauli back propagaciju)
* cilj rada je takođe da smanji potrebu za korekcijom već da može da je primeni samo na te bottleneck delove


---
[*link do rada*](https://arxiv.org/abs/2509.11307)