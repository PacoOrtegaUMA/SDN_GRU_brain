# Hierarchical GRU Framework for Network Traffic Forecasting

Code and preprocessed data for the paper:

> F. Ortega-Zamorano, E. J. Palomo and L. Franco. *Hierarchical GRU Framework for Network Traffic Forecasting*. Submitted to the *Journal of Network and Systems Management*.

The framework has two levels that are trained independently: an hourly model and a minute-level model. Their forecasts are combined at the output through a single weight ω:

x̃ = (1 − ω) · x̂ + ω · ŷ

where x̂ is the minute-level forecast and ŷ the hourly forecast for the hour that contains the last observed minute. With ω = 0 the framework reduces to the single-scale model, which is used as a direct baseline. Any hourly forecaster can be plugged in without retraining the minute-level models.

Four GRU-based architectures are evaluated, each with two target formulations (Variant A: logarithmic difference; Variant B: logarithmic value):

| Name  | Architecture                    |
|-------|---------------------------------|
| Arch1 | GRU                             |
| Arch2 | Conv1D + GRU                    |
| Arch3 | Conv1D + BiGRU                  |
| Arch4 | Conv1D + BiGRU + Attention      |

The reference models are ARIMA(0,1,1) and Chronos-Bolt (small), a pretrained time-series foundation model used zero-shot with the same context as the GRU models. The evaluation uses two real networks:

| Network | Flows | Resolutions     |
|---------|-------|-----------------|
| BRAIN   | 81 origin–destination flows | 1 min and 1 h |
| Abilene | 110 origin–destination flows | 5 min and 1 h |

## Repository structure

```
BRAIN/              BRAIN network: data, scripts and notebooks
Abilene/            Abilene network: data, scripts and notebooks
requirements.txt
```

Each folder is self-contained. Scripts and notebooks use relative paths, so they must be run from inside their folder.

## Data

The raw traffic matrices are publicly available from SNDlib (https://sndlib.put.poznan.pl). The preprocessed CSV files used in the paper are included, so the preprocessing step is only needed to regenerate them from the raw XML files.

| File | Content |
|------|---------|
| `BRAIN/brain_FINAL_1h.csv` | BRAIN, hourly series, 81 flows + calendar features (bit/s) |
| `BRAIN/brain_FINAL_1min.csv` | BRAIN, 1-min series, 81 flows + calendar features (bit/s) |
| `BRAIN/brain_FINAL_1min_con_MEDIAH.csv` | As above, plus the true mean of the hour of each minute (`MEDIAH_*` columns, used only for the upper bound); input of the minute-level scripts |
| `BRAIN/brain_MEDIA1h_desde1min.csv` | Hourly means computed from the 1-min series |
| `Abilene/abilene_FINAL_5min.csv` | Abilene, 5-min series, 110 flows + calendar features (Mbit/s) |
| `Abilene/abilene_FINAL_1h.csv` | Abilene, hourly series (mean of the 5-min values) |

In both networks, the evaluation is restricted to instants with traffic above 1,000 bit/s (0.001 Mbit/s for Abilene), and all models are evaluated on exactly the same instants.

## How to reproduce the results

Run the steps in order. Each training or test script runs all the configurations listed at the end of the script (`__main__` block). The GPU scripts select the device with `CUDA_VISIBLE_DEVICES` at the top of the file; adjust it to your machine.

### BRAIN (`BRAIN/`)

| Step | File | What it does | Output |
|------|------|--------------|--------|
| 1 | `PreProceDatos_Brain.ipynb` | Raw SNDlib XML → aggregated flows + calendar features (optional, CSVs included) | `brain_FINAL_*.csv` |
| 2 | `TrainArq_h.py` | Trains the hourly models (Arch1–4, Variants A/B, W = 24 and 168 h) | trained models |
| 3 | `TestArq_h.py` | Hourly forecasts on the test period | `Results_horas/` |
| 4 | `TrainArq_min.py` | Trains the minute-level models (Arch1–4, Variants A/B, W = 60 min, T+5 and T+15) | trained models |
| 5 | `TestArq_min.py` | Minute-level forecasts on the test period (needs step 3) | `Results_min_Base/` |
| 6 | `TestBaseline.py` | ARIMA(0,1,1) for the hourly (`1h`) and minute (`1min`) series | `Results_horas/`, `Results_min_Base/` |
| 7 | `Test_Chronos_h.py`, `Test_Chronos.py` | Chronos-Bolt (small), zero-shot, hourly and minute level (dataset selected with the `DATASET` variable) | `Results_horas/`, `Results_min_Base/` |
| 8 | `Semillas_Arch1.py` | Retrains Arch1 (Variant A) with nine additional random seeds | `Semillas_Arch1_resumen.csv` |
| 9 | `Coste_modelos.py` | Parameters, training time and inference latency of all models | `Coste_modelos.csv` |
| 10 | `Results_h_Brain.ipynb`, `Results_min_Brain.ipynb` | Tables, figures and statistical tests (see below) | tables, figures, `Tests_estadisticos.csv` |

### Abilene (`Abilene/`)

| Step | File | What it does | Output |
|------|------|--------------|--------|
| 1 | `PreProceDatos_Abilene.ipynb` | Raw SNDlib XML → 110 flows (the two Atlanta routers are merged) + calendar features (optional, CSVs included) | `abilene_FINAL_*.csv` |
| 2 | `TrainArq_hora.py`, `TestArq_h.py` | Hourly models and forecasts | `Results_horas_abilene/` |
| 3 | `TrainArq_min.py`, `TestArq_min.py` | 5-min models and forecasts (needs step 2) | `Results_min_Abilene/` |
| 4 | `TestBaseline_abilene.py` | ARIMA(0,1,1): `python TestBaseline_abilene.py 1h` and `python TestBaseline_abilene.py 5min` | `Results_horas_abilene/`, `Results_min_Abilene/` |
| 5 | `TestChronos_min.py` | Chronos-Bolt (small), zero-shot, 5-min level (dataset selected with the `DATASET` variable) | `Results_min_Abilene/` |
| 6 | `Results_min_Abilene.ipynb` | Fig. 5, statistical tests and data checks (see below) | figures, `Tests_estadisticos.csv` |
| 7 | `Results_h_Abilene.ipynb` | Hourly per-flow metrics against ARIMA (not reported in the paper) | tables |

## Where each result of the paper comes from

| Paper | Notebook / script |
|-------|-------------------|
| Spearman correlation between flows (Section 3.1) | `BRAIN/Results_min_Brain.ipynb`, cell `Figura_real_pred_Spearman` with `HACER_SPEARMAN = True` |
| Table 2 (hourly metrics) | `BRAIN/Results_h_Brain.ipynb`, cell `Tablas_por_enlace_h` |
| Table 3 (hourly per-flow comparison) | `BRAIN/Results_h_Brain.ipynb`, cell `Comparar_1v1_h` |
| Table 4 and Appendix A (minute-level metrics) | `BRAIN/Results_min_Brain.ipynb`, cell `Tablas_por_enlace` |
| Tables 5 and 6 (minute-level per-flow comparison) | `BRAIN/Results_min_Brain.ipynb`, cell `Comparar_1v1_Chronos_Base` |
| Table 7 (statistical tests, BRAIN) | `BRAIN/Results_min_Brain.ipynb`, cell `Tests_estadisticos` |
| Fig. 2 (ground truth and forecasts) | `BRAIN/Results_min_Brain.ipynb`, cell `Figura_real_pred_Spearman` |
| Figs. 3 and 4 (effect of ω, BRAIN) | `BRAIN/Results_min_Brain.ipynb`, cell `Figura_combinacion_omega` |
| Random seeds (Sections 5.2 and 5.3) | `BRAIN/Semillas_Arch1.py` → `Semillas_Arch1_resumen.csv` |
| Table 8 (computational cost) | `BRAIN/Coste_modelos.py` → `Coste_modelos.csv` |
| Fig. 5 (effect of ω, Abilene) | `Abilene/Results_min_Abilene.ipynb`, cell `Figura_combinacion_omega_Abilene` |
| Table 9 (statistical tests, Abilene) | `Abilene/Results_min_Abilene.ipynb`, cell `Tests_estadisticos` |
| Abilene data characteristics (Section 5.5) | `Abilene/Results_min_Abilene.ipynb`, cell `Comprobar_Abilene_datos` |

The notebooks are stored with their outputs, so the tables and figures can be inspected on GitHub without running anything.

## Not included

The trained models and the per-flow forecasts are not included because of their size. They are regenerated by the training and test scripts. The previous version of this repository (first submission) is available under the tag `version-original`.

## Requirements

Python 3.11 and the packages listed in `requirements.txt`. The Chronos scripts use PyTorch and the `chronos-forecasting` package. A GPU is recommended for training but not required; all the analysis notebooks run on CPU.

## Authors and funding

Francisco Ortega-Zamorano, Esteban J. Palomo and Leonardo Franco, University of Málaga, Spain.

This work was supported by grant PID2024-155334OB-I00 funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU.
