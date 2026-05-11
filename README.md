# SDN_GRU_brain

Training, evaluation and analysis code for network traffic prediction on the **BRAIN** (Berlin Research Area Information Network) dataset, using GRU-based architectures within a two-level hierarchical forecasting framework.

---

## Repository Structure

```
SDN_GRU_brain/
│
├── brain_FINAL_1h.csv          # BRAIN dataset — hourly granularity (not included, see Dataset)
├── brain_FINAL_1min.csv        # BRAIN dataset — minute granularity (not included, see Dataset)
│
├── TrainArq_h.py               # Train the 4 architectures at hourly level
├── TrainArq_min.py             # Train the 4 architectures at minute level (with anchor ω)
├── TestArq_h.py                # Evaluate hourly models on the test set
├── TestArq_min.py              # Evaluate minute-level models on the test set
├── TestBaseline.py             # Evaluate ARIMA and Naive baselines (hourly and minute)
│
├── MostrarResults_h.ipynb      # Notebook: analysis and visualisation of hourly results
├── MostrarResults_min.ipynb    # Notebook: analysis and visualisation of minute-level results
├── PreProceDatos.ipynb         # Notebook: BRAIN dataset preprocessing
│
├── Models_horas/               # Saved models — hourly level
│   └── models_ArqX_1h_{A|B}_W{24|168}/   # X ∈ {1,2,3,4}, variant A or B, window W
│
├── Models_min/                 # Saved models — minute level
│   └── models_ArqX_{5|15}m_{A|B}_W60_p{00|25|50|75}_H{5|15}/
│       # X ∈ {1,2,3,4}, horizon 5 or 15 min, variant A or B, anchor weight ω ∈ {0,25,50,75}%
│
├── Results_horas/              # CSVs with hourly predictions and metrics
│   └── results_ArqX_1h_{A|B}_W{24|168}/
│
├── Results_min/                # CSVs with minute-level predictions and metrics
│   └── results_ArqX_{5|15}m_{A|B}_W60_p{00|25|50|75}_H{5|15}/
│
└── requirements.txt            # Full reproducible conda environment
```

---

## Implemented Architectures

| ID | Name | Main blocks |
|---|---|---|
| **Arch1** | GRU | `GRU(64)` → `Dropout(0.2)` → `Dense(32, ReLU)` → `Dense(H)` |
| **Arch2** | Conv1D + GRU | `Conv1D(64, k=3)` → `LeakyReLU` → `GRU(64)` → `Dense(32, Swish)` → `Dense(H)` |
| **Arch3** | Conv1D + BiGRU | `Conv1D(64, k=3)` → `LeakyReLU` → `BiGRU(64+64)` → `Dense(32, Swish)` → `Dense(H)` |
| **Arch4** | Conv1D + BiGRU + Attention | `Conv1D(64, k=3)` → `BiGRU(64+64, return_seq)` → `SelfAttention(128)` → `GAP` → `Dense(64)` → `Dense(32)` → `Dense(H)` |

Each architecture is trained under two output variants:
- **Variant A** — log difference: the model predicts `Δlog(1+x)`
- **Variant B** — log value: the model predicts `log(1+x)` directly *(recommended)*

---

## Two-Level Hierarchical Framework

```
Level 1 — Hourly
  └─ Trains Arch2_B with W=24 on the hourly series (375 days)
  └─ Generates hourly predictions → saved in Results_horas/

Level 2 — Minute
  └─ Loads the hourly prediction as macro signal (anchor ω)
  └─ Trains ArqX with window W=60 and weight ω ∈ {0%, 25%, 50%, 75%}
  └─ Horizons: T+5 min and T+15 min
```

The anchor `ω` is concatenated to the input window as an additional feature:
`u_t = [x_{t-W+1:t}, ω · ŷ_hourly]`

When `ω = 0`, the Level 2 model operates without hierarchical information.

---

## Dataset

The **BRAIN** dataset is publicly available from the [SNDlib repository](http://sndlib.zib.de/) (Zuse Institute Berlin).

After downloading, run `PreProceDatos.ipynb` to generate:
- `brain_FINAL_1h.csv` — per-link hourly series with temporal features (hour_sin, hour_cos, day_sin, day_cos, is_weekend)
- `brain_FINAL_1min.csv` — per-link 5-minute series with the same features

The BRAIN network has 9 main nodes (ADH, CVK, HTW, HU, SPK, TU, UP, WIAS, ZIB), resulting in **81 links** after hierarchical aggregation (including intra-node self-loops).

---

## Usage

### 1. Set up the environment

```bash
conda create --name sdn_gru --file requirements.txt
conda activate sdn_gru
```

Or with pip (minimal dependencies):

```bash
pip install tensorflow>=2.10 numpy pandas scikit-learn statsmodels matplotlib joblib
```

### 2. Preprocess the data

Run the notebook `PreProceDatos.ipynb` with the raw BRAIN dataset files to generate the processed CSVs.

### 3. Train hourly models (Level 1)

Edit the desired parameters at the top of `TrainArq_h.py`:

```python
ARQUITECTURA = 2      # 1, 2, 3 or 4
MODO         = "B"    # "A" or "B"
W            = 24     # 24 or 168
```

```bash
python TrainArq_h.py
```

Models are saved to `Models_horas/models_Arq{N}_1h_{mode}_W{W}/`.

### 4. Evaluate hourly models

```bash
python TestArq_h.py
```

Result CSVs are saved to `Results_horas/results_Arq{N}_1h_{mode}_W{W}/`.

### 5. Train minute-level models (Level 2)

Edit the desired parameters at the top of `TrainArq_min.py`:

```python
ARQUITECTURA = 4      # 1, 2, 3 or 4
MODO         = "B"    # "A" or "B"
OMEGA        = 0.25   # 0.0, 0.25, 0.50 or 0.75
HORIZONTE    = 5      # 5 or 15 (minutes)
W            = 60
```

```bash
python TrainArq_min.py
```

### 6. Evaluate baselines (ARIMA + Naive)

```bash
# For hourly granularity — set TIPO = "1h" in TestBaseline.py
python TestBaseline.py

# For minute granularity — set TIPO = "1min" in TestBaseline.py
python TestBaseline.py
```

### 7. Visualise results

Open the notebooks:
- `MostrarResults_h.ipynb` — tables and plots for hourly results
- `MostrarResults_min.ipynb` — MAPE vs ω plots and win-rate against ARIMA

---

## Naming Convention

Directories follow this pattern:

```
{Models|Results}_{horas|min}/
  {models|results}_Arq{1|2|3|4}_{1h|5m|15m}_{A|B}_W{window}_[p{omega}_H{horizon}]/
```

Examples:
- `models_Arq4_5m_B_W60_p25_H5` → Arch4, Variant B, T+5 min, W=60, ω=25%
- `results_Arq2_1h_B_W24` → Arch2, Variant B, hourly, W=24

---

## Environment

- Python 3.11
- TensorFlow / Keras 3.x
- CUDA 12.9 + cuDNN 9.x
- Trained on GPU (NVIDIA)

See `requirements.txt` for the full reproducible conda environment.

---

## Credits

Francisco Ortega-Zamorano · Esteban J. Palomo · Leonardo Franco  
University of Málaga — Dept. of Languages and Computer Science / ITIS Software  
Funded by MICINN · PID2024-155334OB-I00 (co-funded by FEDER)
