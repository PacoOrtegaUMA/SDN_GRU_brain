# ============================================================
# TestBaseline_abilene.py — v1.1
# v1.1: carpetas de salida iguales a las que leen las figuras, tablas y tests:
#       1h → Results_horas_abilene/results_baselines_1h ; 5min → Results_min_Abilene/results_baselines_5min
# Baselines Naive, Media Móvil y ARIMA(0,1,1) para Abilene.
# Misma lógica que TestBaseline.py (BRAIN), con estos cambios:
#   1. Ficheros, fechas y horizontes de Abilene (pasos de 5 min en el nivel fino).
#   2. Las columnas 'hour' y 'day_of_week' ya no se tratan como enlaces
#      (en la versión de BRAIN se colaban en la lista).
#   3. Los baselines ven la historia ANTERIOR al corte (antes empezaban
#      "en frío" en el primer instante del test); solo se guarda el test.
#   4. Un proceso por enlace en paralelo (N_JOBS) para que ARIMA no tarde horas.
#
# Uso:  python TestBaseline_abilene.py 1h
#       python TestBaseline_abilene.py 5min
# ============================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
TIPO = sys.argv[1] if len(sys.argv) > 1 else "1h"     # "1h" o "5min"
assert TIPO in ("1h", "5min"), "TIPO debe ser '1h' o '5min'"

ARCHIVO_CSV_DATOS  = f"abilene_FINAL_{TIPO}.csv"
CARPETA_RESULTADOS = (os.path.join("Results_horas_abilene", "results_baselines_1h") if TIPO == "1h"
                      else os.path.join("Results_min_Abilene", "results_baselines_5min"))

if TIPO == "1h":
    HORIZONTES  = [1]                       # T+1 h
    FECHA_CORTE = "2004-07-11 00:00:00"     # mismo corte que TrainArq_h (Abilene)
    INICIO      = None
else:
    HORIZONTES  = [1, 3]                    # T+5 min y T+15 min (pasos de 5 min)
    FECHA_CORTE = "2004-08-08 00:00:00"     # mismo corte que TrainArq_min (Abilene)
    INICIO      = "2004-07-11 00:00:00"     # el nivel fino solo usa datos desde aquí

M_MA      = 24     # ventana de la media móvil (pasos)
K_ARIMA   = 100    # historia usada en cada ajuste ARIMA (pasos)
SALTO_ARIMA = 1
N_JOBS    = max(1, (os.cpu_count() or 2) - 2)   # procesos en paralelo (uno por enlace)

COLS_NO_ENLACE = ["hour", "day_of_week", "hour_sin", "hour_cos",
                  "day_sin", "day_cos", "is_weekend"]

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ==========================================
# 2. LÓGICA (igual que en BRAIN, pero solo predice desde el corte)
# ==========================================
def procesar_baselines(series_vals, t_ini, offsets, m_ma, k_arima, step_arima):
    """series_vals: serie completa (historia + test).
       t_ini: índice del primer instante de test. Se rellenan solo predicciones
       cuyo instante objetivo es >= t_ini, usando solo datos anteriores."""
    from statsmodels.tsa.arima.model import ARIMA
    warnings.filterwarnings("ignore")      # también dentro de cada proceso paralelo

    n   = len(series_vals)
    n_h = len(offsets)
    max_h = max(offsets)

    res = {k: np.full((n, n_h), np.nan, dtype=np.float32) for k in ("naive", "ma", "arima")}

    # Naive y Media Móvil
    for idx_h, h in enumerate(offsets):
        for t in range(max(0, t_ini - h), n - h):
            res["naive"][t + h, idx_h] = series_vals[t]
            inicio_ma = max(0, t - m_ma + 1)
            res["ma"][t + h, idx_h] = np.mean(series_vals[inicio_ma:t + 1])

    # ARIMA(0,1,1): origen t0, predice t0+1 ... t0+max_h
    for t0 in range(max(0, t_ini - max_h), n - 1, step_arima):
        history = series_vals[max(0, t0 - k_arima + 1): t0 + 1]
        if len(history) <= 10:
            continue
        try:
            forecast = ARIMA(history, order=(0, 1, 1)).fit().forecast(steps=max_h)
            for idx_h, h in enumerate(offsets):
                if t0 + h < n:
                    res["arima"][t0 + h, idx_h] = forecast[h - 1]
        except Exception:
            pass

    return res


def procesar_enlace(enlace, serie):
    ruta_csv = os.path.join(CARPETA_RESULTADOS, f"test_results_{enlace}_baselines.csv")
    if os.path.exists(ruta_csv):
        return f"⏩ {enlace} — ya existe"

    t_ini = int(np.searchsorted(serie.index.values, np.datetime64(FECHA_CORTE)))
    if len(serie) - t_ini < max(HORIZONTES) + 10:
        return f"⚠️ {enlace} — datos insuficientes para test"

    vals = serie.to_numpy(dtype=np.float32)
    res  = procesar_baselines(vals, t_ini, HORIZONTES, M_MA, K_ARIMA, SALTO_ARIMA)

    df_final = pd.DataFrame(index=serie.index[t_ini:])
    df_final["Realidad"] = vals[t_ini:]
    for idx_h, h in enumerate(HORIZONTES):
        suf = f"_t+{h}"
        df_final[f"Naive{suf}"] = res["naive"][t_ini:, idx_h]
        df_final[f"MA{suf}"]    = res["ma"][t_ini:, idx_h]
        df_final[f"ARIMA{suf}"] = res["arima"][t_ini:, idx_h]
    df_final.index.name = "timestamp"
    df_final.to_csv(ruta_csv)
    return f"✅ {enlace}"


# ==========================================
# 3. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(ARCHIVO_CSV_DATOS):
        sys.exit(f"❌ No existe {ARCHIVO_CSV_DATOS}")

    df = pd.read_csv(ARCHIVO_CSV_DATOS, index_col="timestamp", parse_dates=True)
    if INICIO is not None:
        df = df.loc[INICIO:]
    enlaces = [c for c in df.columns if c not in COLS_NO_ENLACE and not c.startswith("MACRO_")]

    print(f"📂 Abilene {TIPO}: {len(enlaces)} flujos | {df.index.min()} → {df.index.max()}")
    print(f"   Test desde {FECHA_CORTE} | horizontes (pasos): {HORIZONTES} | {N_JOBS} procesos")
    print(f"   Resultados en: {CARPETA_RESULTADOS}")

    t0 = time.time()
    mensajes = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(procesar_enlace)(e, df[e]) for e in enlaces
    )
    for m in mensajes:
        print(m)
    print(f"\n🚀 BASELINES {TIPO} FINALIZADOS en {(time.time() - t0)/60:.1f} min.")
