# ============================================================
# Test_Chronos_Abilene.py — v1.0  (copia de Test_Chronos.py v1.2 con DATASET = "abilene")
#   Abilene: pasos de 5 min → contexto W = 12 pasos (60 min, igual que las GRU), T+5 = 1 paso, T+15 = 3 pasos.
#   Test desde 2004-08-08 (como los baselines). Salida en Results_min_Abilene/results_chronos-bolt-small_C12_{5,15}m/
# v1.2: CUDA_VISIBLE_DEVICES = "2" y TF_CPP_MIN_LOG_LEVEL = "2" al principio, como en el resto de scripts.
# v1.1: aviso si va en CPU.
# Predicción zero-shot (sin entrenar) con Chronos-Bolt sobre el MISMO test que las GRU,
# para comparar con ellas y con ARIMA (respuesta a R2, comentario mayor 1).
#   - Para cada enlace e instante del test, se le da la historia observada hasta t − H
#     y se pide la predicción. Una sola llamada da T+5 y T+15 a la vez.
#   - CONTEXTOS: "W" = la misma ventana que las GRU (60 min) ; 512 = contexto largo.
#   - Guarda un CSV por enlace con el mismo formato que los tests de las GRU:
#       <ROOT>/results_<modelo>_C<ctx>_<H>m/test_<modelo>_C<ctx>_<H>m_<enlace>.csv
#       columnas: timestamp, Realidad, Prediccion
#   - Al final imprime parámetros del modelo y latencia media por predicción.
# Requiere: pip install chronos-forecasting  (y torch con GPU si es posible)
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from chronos import BaseChronosPipeline

DATASET  = "abilene"                      # "brain" | "abilene"
MODELO   = "amazon/chronos-bolt-small"    # tiny | mini | small | base
CONTEXTOS = ["W"]                         # "W" = misma ventana que las GRU (60 min = 12 pasos)
HORIZONTES_MIN = [5, 15]
BATCH    = 1024
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
REHACER  = False

if DATASET == "brain":
    CSV, FECHA_CORTE, PASO_MIN, ROOT = ("brain_FINAL_1min_con_MEDIAH.csv", "2013-03-14 00:00:00",
                                        1, "Results_min_Base")
else:
    CSV, FECHA_CORTE, PASO_MIN, ROOT = ("abilene_FINAL_5min.csv", "2004-08-08 00:00:00",
                                        5, "Results_min_Abilene")
W_MIN = 60
COLS_NO = ["timestamp", "hour", "day_of_week", "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
TAG = MODELO.split("/")[-1]


def predecir(pipe, ventanas, h):
    """Mediana de la predicción para un lote de ventanas → array [B, h]."""
    t = torch.tensor(np.ascontiguousarray(ventanas), dtype=torch.float32)
    try:
        q, _ = pipe.predict_quantiles(t, prediction_length=h, quantile_levels=[0.5])
        return q[..., 0].float().cpu().numpy()
    except (AttributeError, TypeError):
        f = pipe.predict(t, prediction_length=h)                 # [B, n_cuantiles, h]
        return f[:, f.shape[1] // 2, :].float().cpu().numpy()


pipe = BaseChronosPipeline.from_pretrained(MODELO, device_map=DEVICE, torch_dtype=torch.float32)
n_param = sum(p.numel() for p in pipe.model.parameters()) if hasattr(pipe, "model") else None

df = pd.read_csv(CSV, parse_dates=["timestamp"])
enlaces = [c for c in df.columns if c not in COLS_NO and not c.startswith(("MEDIAH_", "MEDIAHU_"))]
ts = df["timestamp"].values
i0 = int((df["timestamp"] < FECHA_CORTE).sum())                  # primer índice del test
pasos = {H: H // PASO_MIN for H in HORIZONTES_MIN}
hmin, hmax = min(pasos.values()), max(pasos.values())
print(f"{TAG} | {DATASET} | {len(enlaces)} enlaces | test desde {FECHA_CORTE} | {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "  ⚠️ sin GPU: irá mucho más lento"))

t_total, n_total = 0.0, 0
for ctx in CONTEXTOS:
    C = W_MIN // PASO_MIN if ctx == "W" else int(ctx)
    carpetas = {H: os.path.join(ROOT, f"results_{TAG}_C{C}_{H}m") for H in HORIZONTES_MIN}
    for d in carpetas.values():
        os.makedirs(d, exist_ok=True)
    print(f"\n=== contexto C = {C} pasos ({C * PASO_MIN} min) ===")
    for k, e in enumerate(enlaces, 1):
        rutas = {H: os.path.join(carpetas[H], f"test_{TAG}_C{C}_{H}m_{e}.csv") for H in HORIZONTES_MIN}
        if not REHACER and all(os.path.exists(r) for r in rutas.values()):
            print(f"⏩ [{k}/{len(enlaces)}] {e}"); continue
        x = df[e].values.astype(np.float32)
        # último observado o: desde (i0 − hmax) hasta (len − 1 − hmin), y con historia suficiente
        o_ini, o_fin = max(C - 1, i0 - hmax), len(x) - 1 - hmin
        o = np.arange(o_ini, o_fin + 1)
        V = sliding_window_view(x, C)[o - C + 1]                  # ventana que termina en o
        P = np.empty((len(o), hmax), dtype=np.float32)
        t0 = time.perf_counter()
        for b in range(0, len(o), BATCH):
            P[b:b + BATCH] = predecir(pipe, V[b:b + BATCH], hmax)
        t_total += time.perf_counter() - t0; n_total += len(o)
        for H, h in pasos.items():
            j = o + h
            m = (j >= i0) & (j < len(x))
            pd.DataFrame({"timestamp": ts[j[m]], "Realidad": x[j[m]],
                          "Prediccion": np.clip(P[m, h - 1], 0, None)}).to_csv(rutas[H], index=False)
        print(f"✅ [{k}/{len(enlaces)}] {e} | {len(o)} predicciones")

print(f"\nParámetros de {TAG}: {n_param:,}" if n_param else "")
if n_total:
    print(f"Latencia media por predicción ({DEVICE}, lotes de {BATCH}): {1000 * t_total / n_total:.3f} ms")