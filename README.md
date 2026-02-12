# werkstatt

Repo für die AI Werkstatt (M4 Forecasting mit uni2ts Modellen).

**Authors**
- Berkan Coban
- Magdalena Gufler

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

M4 Dataset
Makridakis et al. (M4 Competition)
Usage: benchmark dataset with six frequency groups and fixed horizons

TimesFM
Checkpoints: google/timesfm-1.0-200m-pytorch (Hugging Face)
Reference: Das et al. (TimesFM)
License: see the model card on Hugging Face
Usage: deterministic point forecasts (no sampling) in our evaluation

Chronos
Checkpoints: amazon/chronos-t5-tiny, amazon/chronos-t5-base (Hugging Face)
Reference: Ansari et al. (Chronos)
License: see the respective model cards on Hugging Face
Usage: probabilistic forecasts (sampling); we aggregate samples via the median for point-metric evaluation

Moirai
Checkpoints: Salesforce/moirai-1.0-R-small, Salesforce/moirai-1.0-R-base (Hugging Face)
Reference: Liu et al. (Moirai)
License: see the respective model cards on Hugging Face
Usage: probabilistic forecasts (sampling); we aggregate samples via the median for point-metric evaluation

ChatGPT was used as a supporting tool while drafting and refining parts of the plotting scripts (scripts/plots). 
All code was reviewed and adapted by the authors.