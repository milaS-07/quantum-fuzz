# Extracting Success from IBM's 20-Qubit Machines Using Error-Aware Compilation

Rad predlaže kompajler svestan grešaka za IBM-ove 20-kjubitne procesore, koji kola mapira na hardver tako da maksimizuje procenjenu verovatnoću uspeha (ESP).

## Šta je preuzeto iz rada

* **Formula:** ESP kola je proizvod verovatnoća uspeha njegovih kapija, $ESP(C) = \prod_{g \in C} (1 - \varepsilon_g)$, gde je $\varepsilon_g$ greška kapije $g$
* **Pretpostavke modela:** svaka kapija ili uspe potpuno ili ne uspe, a verovatnoće neuspeha su međusobno nezavisne

## Razlike u odnosu na rad

* **Izvor grešaka $\varepsilon$:** U radu se $\varepsilon$ uzima direktno iz IBM-ovih kalibracionih vrednosti (dobijenih randomized benchmarking-om) za svaki kjubit i svaki par kjubita posebno. U ovom radu (mom) se $\varepsilon = 1 - F_{avg}$ računa iz kanala greške modela šuma koji generiše "pravi" fideliti, pa se ESP prilagođava modelu šuma

---
[*link do rada*](https://arxiv.org/abs/1903.10963)