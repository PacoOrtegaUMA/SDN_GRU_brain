# SDN_GRU_brain

Código de entrenamiento, evaluación y análisis de predicción de tráfico de red sobre el dataset **BRAIN** (Berlin Research Area Information Network) usando arquitecturas basadas en GRU con un framework jerárquico de dos niveles.

---

## Estructura del repositorio

```
SDN_GRU_brain/
│
├── brain_FINAL_1h.csv          # Dataset BRAIN granularidad horaria (no incluido, ver Dataset)
├── brain_FINAL_1min.csv        # Dataset BRAIN granularidad minuto (no incluido, ver Dataset)
│
├── TrainArq_h.py               # Entrenamiento de las 4 arquitecturas a nivel horario
├── TrainArq_min.py             # Entrenamiento de las 4 arquitecturas a nivel minuto (con anchor ω)
├── TestArq_h.py                # Evaluación de modelos horarios sobre el conjunto de test
├── TestArq_min.py              # Evaluación de modelos de minuto sobre el conjunto de test
├── TestBaseline.py             # Evaluación de baselines ARIMA y Naive (horario y minuto)
│
├── MostrarResults_h.ipynb      # Notebook: análisis y visualización de resultados horarios
├── MostrarResults_min.ipynb    # Notebook: análisis y visualización de resultados de minuto
├── PreProceDatos.ipynb         # Notebook: preprocesamiento del dataset BRAIN
│
├── Models_horas/               # Modelos entrenados a nivel horario
│   └── models_ArqX_1h_{A|B}_W{24|168}/   # X ∈ {1,2,3,4}, variante A o B, ventana W
│
├── Models_min/                 # Modelos entrenados a nivel minuto
│   └── models_ArqX_{5|15}m_{A|B}_W60_p{00|25|50|75}_H{5|15}/
│       # X ∈ {1,2,3,4}, horizonte 5 o 15 min, variante A o B, peso ω ∈ {0,25,50,75}%
│
├── Results_horas/              # CSVs de predicciones y métricas horarias
│   └── results_ArqX_1h_{A|B}_W{24|168}/
│
├── Results_min/                # CSVs de predicciones y métricas de minuto
│   └── results_ArqX_{5|15}m_{A|B}_W60_p{00|25|50|75}_H{5|15}/
│
└── requirements.txt            # Entorno conda completo
```

---

## Arquitecturas implementadas

| ID | Nombre | Bloques principales |
|---|---|---|
| **Arch1** | GRU | `GRU(64)` → `Dropout(0.2)` → `Dense(32, ReLU)` → `Dense(H)` |
| **Arch2** | Conv1D + GRU | `Conv1D(64, k=3)` → `LeakyReLU` → `GRU(64)` → `Dense(32, Swish)` → `Dense(H)` |
| **Arch3** | Conv1D + BiGRU | `Conv1D(64, k=3)` → `LeakyReLU` → `BiGRU(64+64)` → `Dense(32, Swish)` → `Dense(H)` |
| **Arch4** | Conv1D + BiGRU + Attention | `Conv1D(64, k=3)` → `BiGRU(64+64, return_seq)` → `SelfAttention(128)` → `GAP` → `Dense(64)` → `Dense(32)` → `Dense(H)` |

Cada arquitectura se entrena en dos variantes de salida:
- **Variante A** — diferencia logarítmica: el modelo predice `Δlog(1+x)`
- **Variante B** — valor logarítmico: el modelo predice `log(1+x)` directamente *(recomendada)*

---

## Framework jerárquico de dos niveles

```
Level 1 (horario)
  └─ Entrena Arch2_B con W=24 sobre la serie horaria (375 días)
  └─ Genera predicciones hourly → guardadas en Results_horas/

Level 2 (minuto)
  └─ Carga la predicción horaria como señal macro (anchor ω)
  └─ Entrena ArqX con ventana W=60 y peso ω ∈ {0%, 25%, 50%, 75%}
  └─ Horizontes: T+5 min y T+15 min
```

El anchor `ω` se concatena a la ventana de entrada como característica adicional:
`u_t = [x_{t-W+1:t}, ω · ŷ_hourly]`

Cuando `ω = 0`, el modelo Level 2 opera sin información jerárquica.

---

## Dataset

El dataset **BRAIN** está disponible públicamente en el repositorio [SNDlib](http://sndlib.zib.de/) (Zuse Institute Berlin).

Tras descargarlo, ejecutar `PreProceDatos.ipynb` para generar los ficheros:
- `brain_FINAL_1h.csv` — serie horaria por enlace con features temporales (hour_sin, hour_cos, day_sin, day_cos, is_weekend)
- `brain_FINAL_1min.csv` — serie de 5 minutos por enlace con las mismas features

La red BRAIN tiene 9 nodos principales (ADH, CVK, HTW, HU, SPK, TU, UP, WIAS, ZIB), resultando en **81 enlaces** tras la agregación jerárquica (incluyendo self-loops intra-nodo).

---

## Uso

### 1. Instalar el entorno

```bash
conda create --name sdn_gru --file requirements.txt
conda activate sdn_gru
```

O con pip (dependencias mínimas):

```bash
pip install tensorflow>=2.10 numpy pandas scikit-learn statsmodels matplotlib joblib
```

### 2. Preprocesar los datos

Ejecutar el notebook `PreProceDatos.ipynb` con los ficheros raw del dataset BRAIN para generar los CSV procesados.

### 3. Entrenar modelos horarios (Level 1)

Editar en `TrainArq_h.py` los parámetros deseados:

```python
ARQUITECTURA = 2      # 1, 2, 3 o 4
MODO         = "B"    # "A" o "B"
W            = 24     # 24 o 168
```

```bash
python TrainArq_h.py
```

Los modelos se guardan en `Models_horas/models_Arq{N}_1h_{modo}_W{W}/`.

### 4. Evaluar modelos horarios

```bash
python TestArq_h.py
```

Los resultados CSV se guardan en `Results_horas/results_Arq{N}_1h_{modo}_W{W}/`.

### 5. Entrenar modelos de minuto (Level 2)

Editar en `TrainArq_min.py`:

```python
ARQUITECTURA = 4      # 1, 2, 3 o 4
MODO         = "B"    # "A" o "B"
OMEGA        = 0.25   # 0.0, 0.25, 0.50, 0.75
HORIZONTE    = 5      # 5 o 15 (minutos)
W            = 60
```

```bash
python TrainArq_min.py
```

### 6. Evaluar baselines (ARIMA + Naive)

```bash
# Para granularidad horaria:
# Editar TIPO = "1h" en TestBaseline.py
python TestBaseline.py

# Para granularidad minuto:
# Editar TIPO = "1min" en TestBaseline.py
python TestBaseline.py
```

### 7. Visualizar resultados

Abrir los notebooks:
- `MostrarResults_h.ipynb` — tablas y gráficas de resultados horarios
- `MostrarResults_min.ipynb` — tablas, gráficas MAPE vs ω y win-rate vs ARIMA

---

## Convención de nombres

Los directorios siguen el patrón:

```
{Models|Results}_{horas|min}/
  {models|results}_Arq{1|2|3|4}_{1h|5m|15m}_{A|B}_W{window}_[p{omega}_H{horizon}]/
```

Ejemplos:
- `models_Arq4_5m_B_W60_p25_H5` → Arch4, Variant B, T+5 min, W=60, ω=25%
- `results_Arq2_1h_B_W24` → Arch2, Variant B, horario, W=24

---

## Entorno

- Python 3.11
- TensorFlow / Keras 3.x
- CUDA 12.9 + cuDNN 9.x
- Entrenado en GPU (NVIDIA)

Ver `requirements.txt` para el entorno conda completo reproducible.

---

## Créditos

Francisco Ortega-Zamorano · Esteban J. Palomo · Leonardo Franco  
Universidad de Málaga — Dpto. Lenguajes y Ciencias de la Computación / ITIS Software  
Financiado por MICINN · PID2024-155334OB-I00 (con fondos FEDER)
