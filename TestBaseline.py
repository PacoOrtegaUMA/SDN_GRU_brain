import os
import pandas as pd
import numpy as np
import sys
import time
import warnings
from statsmodels.tsa.arima.model import ARIMA

# Forzar uso de GPU 2 (aunque ARIMA usa CPU, mantenemos coherencia)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN DINÁMICA
# ==========================================
TIPO = "1min"  # <--- OPCIONES: "1h" o "1min"

ARCHIVO_CSV_DATOS = f"brain_FINAL_{TIPO}.csv"
CARPETA_RESULTADOS = os.path.join("Results", f"results_baselines_{TIPO}")

if TIPO == "1h":
    HORIZONTES = [1]
    FECHA_CORTE = "2013-02-01" # Fecha para 1h
else:
    HORIZONTES = [1, 5, 15]           # para 1min
    FECHA_CORTE = "2013-03-14 00:00:00" # Fecha para 1min

M_MA = 24                
K_ARIMA = 100            
SALTO_ARIMA = 1          

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
if not os.path.exists(ARCHIVO_CSV_DATOS):
    print(f"❌ No existe {ARCHIVO_CSV_DATOS}")
    sys.exit()

# Cargamos el CSV. Nota: El CSV de 1min suele ser mucho más pesado
df = pd.read_csv(ARCHIVO_CSV_DATOS, index_col='timestamp', parse_dates=True)
COLS_TIEMPO = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend']
TODOS_LOS_ENLACES = [c for c in df.columns if c not in COLS_TIEMPO and not c.startswith('MACRO_')]

print(f"📂 Dataset BRAIN {TIPO} cargado. {len(TODOS_LOS_ENLACES)} enlaces.")
print(f"📂 Los resultados se guardarán en: {CARPETA_RESULTADOS}")
print(f"⏳ Horizontes configurados: {HORIZONTES}")

# ==========================================
# 3. LÓGICA DE PROCESAMIENTO
# ==========================================
def procesar_baselines(series_vals, offsets, m_ma, k_arima, step_arima):
    n = len(series_vals)
    n_h = len(offsets)
    max_h = max(offsets)

    res = {
        "naive": np.full((n, n_h), np.nan, dtype=np.float32),
        "ma":    np.full((n, n_h), np.nan, dtype=np.float32),
        "arima": np.full((n, n_h), np.nan, dtype=np.float32),
    }

    # Naive y Media Móvil
    for idx_h, h in enumerate(offsets):
        for t in range(n - h):
            res["naive"][t + h, idx_h] = series_vals[t]
            inicio_ma = max(0, t - m_ma + 1)
            res["ma"][t + h, idx_h] = np.mean(series_vals[inicio_ma:t+1])

    # ARIMA (0,1,1)
    print(f"Calculando ARIMA...")
    pasos_total = len(range(0, n - max_h, step_arima))
    pasos_hechos = 0
    start_time = time.time()

    for t0 in range(0, n - max_h, step_arima):
        inicio_hist = max(0, t0 - k_arima + 1)
        history = series_vals[inicio_hist : t0 + 1]

        if len(history) > 10:
            try:
                model = ARIMA(history, order=(0, 1, 1)).fit()
                forecast = model.forecast(steps=max_h)
                for idx_h, h in enumerate(offsets):
                    if t0 + h < n:
                        res["arima"][t0 + h, idx_h] = forecast[h - 1]
            except:
                pass

        pasos_hechos += 1
        if pasos_hechos % 50 == 0 or pasos_hechos == pasos_total:
            prog = 100.0 * pasos_hechos / pasos_total
            elapsed = time.time() - start_time
            avg_time = elapsed / pasos_hechos
            eta = avg_time * (pasos_total - pasos_hechos)
            sys.stdout.write(f"\r   > Progreso: {prog:4.1f}% | ETA: {eta/60:4.2f} min")
            sys.stdout.flush()

    return res

# ==========================================
# 4. EJECUCIÓN POR ENLACE
# ==========================================
for idx, ENLACE in enumerate(TODOS_LOS_ENLACES, start=1):
    print(f"\n" + "-"*40)
    print(f"📊 [{idx}/{len(TODOS_LOS_ENLACES)}] ENLACE: {ENLACE} ({TIPO})")
    
    ruta_csv = os.path.join(CARPETA_RESULTADOS, f"test_results_{ENLACE}_baselines.csv")
    
    # IMPORTANTE: Filtrar el TEST por fecha
    try:
        test_serie = df.loc[FECHA_CORTE:, ENLACE]
    except KeyError:
        print(f"⚠️ La fecha de corte {FECHA_CORTE} no existe en el índice. Saltando...")
        continue
    
    if len(test_serie) < max(HORIZONTES) + 10:
        print("⚠️ Datos insuficientes para test. Saltando...")
        continue

    valores = test_serie.to_numpy(dtype=np.float32)
    resultados = procesar_baselines(valores, HORIZONTES, M_MA, K_ARIMA, SALTO_ARIMA)

    # DataFrame final para este enlace
    df_final = pd.DataFrame(index=test_serie.index)
    df_final['Realidad'] = test_serie

    for idx_h, h in enumerate(HORIZONTES):
        suf = f"_t+{h}"
        df_final[f"Naive{suf}"] = resultados["naive"][:, idx_h]
        df_final[f"MA{suf}"] = resultados["ma"][:, idx_h]
        df_final[f"ARIMA{suf}"] = resultados["arima"][:, idx_h]

    df_final.to_csv(ruta_csv)
    print(f"\n✅ Guardado: {ruta_csv}")

print(f"\n🚀 PROCESO DE BASELINES {TIPO} FINALIZADO.")