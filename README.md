# werkstatt

Repo für die AI Werkstatt (M4 Forecasting mit uni2ts Modellen).

**Struktur**
- `plots/`: Alle Plots an einem Ort (Unterordner `final/`, `m4/`, `owa/`).
- `scripts/plots/`: Plot-Skripte.
- `scripts/runs/`: Modellläufe für M4.
- `results_final_base/`, `results_final_small/`, `results_MinMax/`: Finale Ergebnisse.
- `results_test/`: Alte Zwischenstände und Tests (frühere Ergebnisse/Plots).
- `data/`: Lokale Daten (M4 CSVs unter `data/m4/datasets`).

**Setup**
- Conda: `conda env create -f environment.yml`
- Aktivieren: `conda activate werkstatt`

**Model Runs (Beispiele)**
Base-Modelle (Default-Checkpoints):
```
python -m scripts.runs.run_all_m4_uni2ts --group Daily --data_dir .\data\m4\datasets --save_dir results_final_base --run_name Daily
```

Small/Tiny-Varianten (Checkpoints überschreiben):
```
python -m scripts.runs.run_all_m4_uni2ts --group Daily --data_dir .\data\m4\datasets --save_dir results_final_small --run_name Daily --chronos_ckpt amazon/chronos-t5-tiny --moirai_repo Salesforce/moirai-1.0-R-small
```

MinMax-Variante:
```
python -m scripts.runs.run_all_m4_uni2ts_minmax --group Daily --data_dir .\data\m4\datasets --save_dir results_MinMax --run_name Daily
```

**References**
- M4 Dataset
- TimesFM
- Chronos
- Moirai
