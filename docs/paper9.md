# Extracting Success from IBM's 20-Qubit Machines Using Error-Aware Compilation

Rad predlaže kompajler svestan grešaka za IBM-ove 20-kjubitne procesore, koji kola mapira na hardver tako da maksimizuje procenjenu verovatnoću uspeha (ESP).

## Šta je preuzeto iz rada

* **Formula:** ESP kola je proizvod verovatnoća uspeha njegovih kapija, $ESP(C) = \prod_{g \in C} (1 - \varepsilon_g)$, gde je $\varepsilon_g$ greška kapije $g$
* **Pretpostavke modela:** svaka kapija ili uspe potpuno ili ne uspe, a verovatnoće neuspeha su međusobno nezavisne

## Razlike u odnosu na rad

* **Izvor grešaka $\varepsilon$:** U radu se $\varepsilon$ uzima direktno iz IBM-ovih kalibracionih vrednosti (dobijenih randomized benchmarking-om) za svaki kjubit i svaki par kjubita posebno. Kod nas postoje samo dve vrednosti, jedna za sve jednokubitne i jedna za sve dvokubitne kapije, izračunate kao $\varepsilon = 1 - F_{avg}$ iz kanala greške istog modela šuma koji generiše "pravi" fideliti (termalna relaksacija + depolarizacija, parametri iz `SHERBROOKE_PARAMS`)
* **Kapije:** `sx`, `x` i `id` dobijaju grešku $\varepsilon_{1q}$, `cx`, `ecr` i `cz` dobijaju $\varepsilon_{2q}$, a `rz` se ne računa ($\varepsilon = 0$) jer je virtuelna kapija. Zato ESP zavisi samo od broja kapija svake vrste, a ne od toga na kojim kjubitima se izvršavaju
* **Greške merenja i pripreme (SPAM):** U radu ulaze u proizvod. Kod nas ne ulaze, jer završna merenja uklanjamo i poredimo fideliti stanja
* **Namena:** U radu se ESP proverava na stvarnom hardveru i koristi unutar pretrage za mapiranje kola. Kod nas se koristi za rangiranje varijanti istog kola, a proverava se prema simulaciji sopstvenog modela šuma

---
[*link do rada*](https://arxiv.org/abs/1903.10963)