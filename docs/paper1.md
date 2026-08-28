# A methodology to select and adjust quantum noise models through emulators: benchmarking against real backends

Rad analizira metodologije simulacija šuma kvantnih procesora (**Qiskit** i **Qaptiva**) na osnovu njihovih kalibracionih parametara.

### Testirani modeli šuma

U oba okruženja ispitane su tri modela:

1. **Lindblad / Thermal Relaxation model ($T_1/T_2$):** Zanemaruje greške samih operacija i uzima u obzir samo relaksaciju i dekoherenciju kjubita tokom vremena, zbog čega daje visoke vrednosti fideliteta (mera koliko se poklapa model sa stvarnim kvantnim šumom)
2. **Depolarizing channel:** Modelira grešku kapija
3. **Kombinovani model (Depolarizing + Thermal Relaxation):** Obuhvata prve dve stavke i ostvarujeO **najbolje rezultate**, sa minimalnim odstupanjem fideliteta (0.686%) u odnosu na fiziku procesora `ibm_perth`


U njihovim rezultatima 3. model daje najbolje rezultate, pa je zato i izabran u ovom istraživanju. U zaključku isto navode kako jedino simultano modeliranje grešaka kapija (`gate_error`) i toplotne relaksacije ($T_1/T_2$) (3. model) obezbeđuje fizički adekvatnu simulaciju kvantnih kola.