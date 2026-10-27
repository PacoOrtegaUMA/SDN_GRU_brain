# ============================================================
# Test_Chronos_h.py — v1.0  (NIVEL HORARIO)
# Predicción zero-shot con Chronos-Bolt sobre el MISMO test horario que las GRU (T+1 h),
# para comparar con ellas y con ARIMA.
#   - Para cada enlace e instante del test se le da la historia observada hasta t − 1 h.
#   - CONTEXTOS = mismas ventanas que las GRU horarias (24 y 168 h).
#   - Un CSV por enlace con el formato de los tests horarios:
#       <ROOT>/results_<modelo>_C<ctx>_1h/test_<modelo>_C<ctx>_1h_<enlace>.csv
#       columnas: timestamp, Realidad, Prediccion
#   - Al final imprime parámetros del modelo y latencia media por predicción.
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

DATASET   = "brain"                        # "brain" | "abilene"
MODELO    = "amazon/chronos-bolt-small"
CONTEXTOS = [24, 168]                      # horas (mismas W que las GRU horarias)
H         = 1                              # T+1 h
BATCH     = 1024
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
REHACER   = False

if DATASET == "brain":       # como TestArq_h.py v1.1
    CSV, FECHA_CORTE, ROOT = "brain_FINAL_1h.csv", "2013-02-01 00:00:00", "Results_horas"
else:                        # como TrainArq_h_abilene.py v1.0 (revisa el nombre del CSV horario)
    CSV, FECHA_CORTE, ROOT = "abilene_FINAL_1h.csv", "2004-07-11 00:00:00", "Results_horas_abilene"
COLS_NO = ["timestamp", "hour", "day_of_week", "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
TAG = MODELO.split("/")[-1]


def predecir(pipe, ventanas, h):
    """Mediana de la predicción para un lote de ventanas → array [B, h]."""
    t = torch.tensor(np.ascontiguousarray(ventanas), dtype=torch.float32)
    try:
        q, _ = pipe.predict_quantiles(t, prediction_length=h, quantile_levels=[0.5])
        return q[..., 0].float().cpu().numpy()
    except (AttributeError, TypeError):
        f = pipe.predict(t, prediction_length=h)
        return f[:, f.shape[1] // 2, :].float().cpu().numpy()


pipe = BaseChronosPipeline.from_pretrained(MODELO, device_map=DEVICE, torch_dtype=torch.float32)
n_param = sum(p.numel() for p in pipe.model.parameters()) if hasattr(pipe, "model") else None

df = pd.read_csv(CSV, parse_dates=["timestamp"])
enlaces = [c for c in df.columns if c not in COLS_NO]
ts = df["timestamp"].values
i0 = int((df["timestamp"] < FECHA_CORTE).sum())                  # primer índice del test
print(f"{TAG} | {DATASET} horario | {len(enlaces)} enlaces | test desde {FECHA_CORTE} | {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "  ⚠️ sin GPU"))

t_total, n_total = 0.0, 0
for C in CONTEXTOS:
    carpeta = os.path.join(ROOT, f"results_{TAG}_C{C}_1h")
    os.makedirs(carpeta, exist_ok=True)
    print(f"\n=== contexto C = {C} h ===")
    for k, e in enumerate(enlaces, 1):
        ruta = os.path.join(carpeta, f"test_{TAG}_C{C}_1h_{e}.csv")
        if not REHACER and os.path.exists(ruta):
            print(f"⏩ [{k}/{len(enlaces)}] {e}"); continue
        x = df[e].values.astype(np.float32)
        o = np.arange(max(C - 1, i0 - H), len(x) - H)             # último observado
        V = sliding_window_view(x, C)[o - C + 1]                  # ventana que termina en o
        P = np.empty(len(o), dtype=np.float32)
        t0 = time.perf_counter()
        for b in range(0, len(o), BATCH):
            P[b:b + BATCH] = predecir(pipe, V[b:b + BATCH], H)[:, H - 1]
        t_total += time.perf_counter() - t0; n_total += len(o)
        j = o + H
        m = j >= i0
        pd.DataFrame({"timestamp": ts[j[m]], "Realidad": x[j[m]],
                      "Prediccion": np.clip(P[m], 0, None)}).to_csv(ruta, index=False)
        print(f"✅ [{k}/{len(enlaces)}] {e} | {m.sum()} predicciones")

print(f"\nParámetros de {TAG}: {n_param:,}" if n_param else "")
if n_total:
    print(f"Latencia media por predicción ({DEVICE}, lotes de {BATCH}): {1000 * t_total / n_total:.3f} ms")