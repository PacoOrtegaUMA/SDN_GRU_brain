# ============================================================
# Coste_modelos.py — v1.2
# v1.2: "red" en HACER: latencia REAL de la red completa con modelos por flujo. Se cargan los 81
#       modelos de la carpeta (uno por flujo) y se mide el tiempo de predecir una ventana con cada
#       uno, uno detrás de otro (mediana de REPS_RED repeticiones), en GPU y en CPU →
#       columnas Lat_red_GPU_ms, Lat_red_CPU_ms y N_modelos_red. Es la latencia comparable con la
#       de Chronos (un único modelo, lote de 81) y la de ARIMA (81 reajustes en secuencia).
#       Como el coste no depende del horizonte ni de la variante (misma red, solo cambia el objetivo),
#       se mide en RED_CONFIGS × MODOS_RED (por defecto minutos W=60 T+5 y horas W=24, variante A).
#       El CSV se ACTUALIZA columna a columna (las columnas que no se miden se conservan), así puedes
#       lanzar solo HACER = ["red"] sin perder lo que ya tenías.
#       Al final se imprime la tabla de latencia de la red completa (81 flujos) para el paper.
# v1.1: HACER = ["arq", "chronos", "arima"] elige qué se mide. ARIMA(0,1,1) como en el script de
#       baselines (reajuste en cada paso con los últimos 100 valores; solo CPU; sin entrenamiento previo).
#       Chronos en minutos (contexto 60, T+1…15) y en horas (contexto 24, T+1 h).
# Coste computacional para R5.9 / R4.10, todo en la MISMA máquina:
#   - Parámetros de cada modelo (Arch1–4, A y B; minutos W=60 T+5/T+15 y horas W=24/168).
#   - Muestras de entrenamiento por flujo.
#   - Tiempo por época de entrenamiento: 1 época de un clon del modelo con datos aleatorios del
#     MISMO tamaño que el train real.
#   - Latencia de UN modelo: lote de 81 ventanas (Lat_81) y una sola ventana (Lat_1).
#   - Latencia de la RED (v1.2): los 81 modelos por flujo, una ventana cada uno (Lat_red).
#   - Chronos-Bolt small: parámetros y latencia con el lote de 81 flujos.
# Ejecutar en una terminal, entorno GRU, desde la carpeta de los modelos:
#     conda activate GRU
#     cd ~/SDN/LSTM_brain_revision
#     python Coste_modelos.py
# Salida: tabla por pantalla y Coste_modelos.csv
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob
import time
import numpy as np
import pandas as pd

# ---------------- configuración ----------------
HACER = ["red"]          # qué medir: "arq" (GRU, un modelo), "red" (81 modelos), "chronos", "arima"
ROOT_MODELS_MIN = "Models_min_Base"
ROOT_MODELS_H   = "Models_horas"            # cambia a "Models_horas_v11" si no la renombraste
CSV_MIN, CORTE_MIN = "brain_FINAL_1min_con_MEDIAH.csv", "2013-03-14 00:00:00"
CSV_H,   CORTE_H   = "brain_FINAL_1h.csv",              "2013-02-01 00:00:00"
IDS        = [1, 2, 3, 4]
MODOS      = ["A", "B"]
CONFIGS    = ([("min", f"models_Arq{{ID}}_{H}m_{{M}}_W60_w00_H{H}", 60, H) for H in (5, 15)] +
              [("hora", f"models_Arq{{ID}}_1h_{{M}}_W{W}", W, 1) for W in (24, 168)])
BATCH_TRAIN = 128           # ⚠️ pon el batch_size que usas en TrainArq_*.py
N_FLUJOS    = 81
REPS        = 50
DISPOSITIVOS = ["GPU", "CPU"]
# Latencia de la red completa (v1.2): (nivel, W, H) y variantes en las que se mide
RED_CONFIGS = [("min", 60, 5), ("hora", 24, 1)]
MODOS_RED   = ["A"]
REPS_RED    = 20
CHRONOS_MODELO = "amazon/chronos-bolt-small"
CHRONOS_CONFIGS = [("min", 60, 15), ("hora", 24, 1)]      # (nivel, contexto, horizonte máx.)
# ARIMA como en el script de baselines
ARIMA_ORDEN, ARIMA_K, ARIMA_REPS = (0, 1, 1), 100, 5
ARIMA_NIVELES = {"min": ("brain_FINAL_1min.csv", CORTE_MIN, 15), "hora": (CSV_H, CORTE_H, 1)}
CSV_SALIDA = "Coste_modelos.csv"

if "arq" in HACER or "red" in HACER:
    import tensorflow as tf
    for _g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_g, True)


def n_train(csv, corte, W, H):
    n = int((pd.read_csv(csv, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"] < corte).sum())
    return max(n - W - H + 1, 0)


def latencia(f, X, reps=REPS):
    for _ in range(3):
        f(X)                                         # calentar
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(X); t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))               # ms


def modelos_de(d):
    return sorted(glob.glob(os.path.join(d, "Model_*.keras")) + glob.glob(os.path.join(d, "Model_*.h5")))


def etiqueta_h(nivel, H):
    return f"T+{H} min" if nivel == "min" else "T+1 h"


def medir_keras(ruta, n_tr):
    res = {}
    with tf.device("/CPU:0"):
        m0 = tf.keras.models.load_model(ruta, compile=False)
    shape = m0.input_shape[1:]
    res["Params"] = m0.count_params()
    rng = np.random.default_rng(0)
    epoca_hecha = False
    for dev in DISPOSITIVOS:
        if dev == "GPU" and not tf.config.list_physical_devices("GPU"):
            continue
        with tf.device(f"/{dev}:0"):
            m = tf.keras.models.load_model(ruta, compile=False)
            X81 = tf.constant(rng.standard_normal((N_FLUJOS, *shape)), tf.float32)
            X1 = X81[:1]
            fwd = tf.function(lambda x: m(x, training=False))
            res[f"Lat_81_{dev}_ms"] = latencia(fwd, X81)
            res[f"Lat_1_{dev}_ms"] = latencia(fwd, X1)
            if not epoca_hecha:                     # tiempo por época en el primer dispositivo disponible
                epoca_hecha = True
                c = tf.keras.models.clone_model(m)
                c.compile(optimizer="adam", loss="mse")
                Xt = rng.standard_normal((n_tr, *shape)).astype("float32")
                yt = rng.standard_normal((n_tr, *c.output_shape[1:])).astype("float32")
                c.fit(Xt[:BATCH_TRAIN * 2], yt[:BATCH_TRAIN * 2], batch_size=BATCH_TRAIN, epochs=1, verbose=0)
                t0 = time.perf_counter()
                c.fit(Xt, yt, batch_size=BATCH_TRAIN, epochs=1, verbose=0)
                res["s_por_epoca"] = time.perf_counter() - t0
                del c
            del m
        tf.keras.backend.clear_session()
    return res


def medir_red(rutas):
    """Latencia de la red completa: cada uno de los modelos por flujo predice su ventana, en secuencia."""
    rutas = rutas[:N_FLUJOS]
    res = {"N_modelos_red": len(rutas)}
    rng = np.random.default_rng(0)
    for dev in DISPOSITIVOS:
        if dev == "GPU" and not tf.config.list_physical_devices("GPU"):
            continue
        with tf.device(f"/{dev}:0"):
            ms = [tf.keras.models.load_model(r, compile=False) for r in rutas]
            fs = [tf.function(lambda x, m=m: m(x, training=False)) for m in ms]
            X = tf.constant(rng.standard_normal((1, *ms[0].input_shape[1:])), tf.float32)

            def red(x):
                for f in fs:
                    f(x)

            res[f"Lat_red_{dev}_ms"] = latencia(red, X, reps=REPS_RED)
            del ms, fs
        tf.keras.backend.clear_session()
    return res


filas = []
cache_n = {}
for nivel, plantilla, W, H in CONFIGS:
    medir_arq = "arq" in HACER
    medir_r   = "red" in HACER and (nivel, W, H) in RED_CONFIGS
    if not (medir_arq or medir_r):
        continue
    csv, corte = (CSV_MIN, CORTE_MIN) if nivel == "min" else (CSV_H, CORTE_H)
    root = ROOT_MODELS_MIN if nivel == "min" else ROOT_MODELS_H
    for ID in IDS:
        for M in MODOS:
            if not medir_arq and M not in MODOS_RED:
                continue
            d = os.path.join(root, plantilla.format(ID=ID, M=M))
            mods = modelos_de(d)
            if not mods:
                print(f"⚠️ sin modelos en {d}"); continue
            fila = {"Nivel": nivel, "Modelo": f"Arch{ID}_{M}", "W": W, "H": etiqueta_h(nivel, H)}
            try:
                if medir_arq:
                    if (csv, W, H) not in cache_n:
                        cache_n[(csv, W, H)] = n_train(csv, corte, W, H)
                    fila["Muestras_train"] = cache_n[(csv, W, H)]
                    fila.update(medir_keras(mods[0], cache_n[(csv, W, H)]))
                if medir_r and M in MODOS_RED:
                    fila.update(medir_red(mods))
                    print(f"✅ red {nivel} {fila['H']} W={W} Arch{ID}_{M}: {fila['N_modelos_red']} modelos, "
                          + " | ".join(f"{dv} {fila[f'Lat_red_{dv}_ms']:.1f} ms"
                                       for dv in DISPOSITIVOS if f"Lat_red_{dv}_ms" in fila))
            except Exception as e:
                print(f"❌ {d}: {e}"); continue
            filas.append(fila)
            if medir_arq:
                print(f"✅ {nivel} {fila['H']} W={W} Arch{ID}_{M}: {fila['Params']:,} parámetros")

if "chronos" in HACER:
    try:
        import torch
        from chronos import BaseChronosPipeline
        for dev in DISPOSITIVOS:
            tdev = "cuda" if dev == "GPU" else "cpu"
            if tdev == "cuda" and not torch.cuda.is_available():
                continue
            pipe = BaseChronosPipeline.from_pretrained(CHRONOS_MODELO, device_map=tdev, torch_dtype=torch.float32)
            n_par = sum(p.numel() for p in pipe.model.parameters())
            for nivel, ctx, hmax in CHRONOS_CONFIGS:
                X81 = torch.rand(N_FLUJOS, ctx)
                f = lambda x: pipe.predict_quantiles(x, prediction_length=hmax, quantile_levels=[0.5])
                etiqueta = f"T+1…{hmax} min" if nivel == "min" else "T+1 h"
                fila = next((x for x in filas if x["Modelo"] == "Chronos-Bolt small" and x["Nivel"] == nivel), None)
                if fila is None:
                    fila = {"Nivel": nivel, "Modelo": "Chronos-Bolt small", "W": ctx, "H": etiqueta,
                            "Muestras_train": 0, "Params": n_par}
                    filas.append(fila)
                fila[f"Lat_81_{dev}_ms"] = latencia(f, X81, reps=20)
                fila[f"Lat_1_{dev}_ms"] = latencia(f, X81[:1], reps=20)
            del pipe
        print("✅ Chronos medido")
    except Exception as e:
        print(f"⚠️ Chronos no medido: {e}")

if "arima" in HACER:
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    warnings.filterwarnings("ignore")
    no_enl = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "hour", "day_of_week"]
    for nivel, (csv, corte, hmax) in ARIMA_NIVELES.items():
        if not os.path.exists(csv):
            print(f"⚠️ ARIMA: no existe {csv}"); continue
        d = pd.read_csv(csv, index_col="timestamp", parse_dates=True)
        enl = [c for c in d.columns if c not in no_enl and not c.startswith(("MACRO_", "MEDIAH_", "MEDIAHU_"))]
        test = d.loc[corte:, enl]
        rng = np.random.default_rng(0)
        t_flujo, n_par = [], None
        for e in enl:
            x = test[e].to_numpy(dtype=np.float32)
            ts = []
            for t0 in rng.integers(ARIMA_K, len(x) - hmax, size=ARIMA_REPS):
                t1 = time.perf_counter()
                try:
                    m = ARIMA(x[t0 - ARIMA_K + 1: t0 + 1], order=ARIMA_ORDEN).fit()
                    m.forecast(steps=hmax)
                    n_par = n_par or len(m.params)
                except Exception:
                    continue
                ts.append(time.perf_counter() - t1)
            if ts:
                t_flujo.append(np.median(ts))
        t_flujo = np.array(t_flujo)
        filas.append({"Nivel": nivel, "Modelo": f"ARIMA{ARIMA_ORDEN}", "W": ARIMA_K,
                      "H": f"T+1…{hmax} min" if nivel == "min" else "T+1 h", "Muestras_train": 0,
                      "Params": n_par, "Lat_1_CPU_ms": 1000 * float(np.median(t_flujo)),
                      "Lat_81_CPU_ms": 1000 * float(t_flujo.mean()) * N_FLUJOS})
        print(f"✅ ARIMA {nivel}: {1000 * np.median(t_flujo):.1f} ms por flujo (reajuste + forecast)")

# ---------------- guardar: se actualiza columna a columna ----------------
clave = ["Nivel", "Modelo", "W", "H"]
df = pd.DataFrame(filas)
if os.path.exists(CSV_SALIDA) and not df.empty:
    viejo = pd.read_csv(CSV_SALIDA).set_index(clave)
    df = df.set_index(clave).combine_first(viejo).reset_index()
elif os.path.exists(CSV_SALIDA):
    df = pd.read_csv(CSV_SALIDA)
df.to_csv(CSV_SALIDA, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fmt = {c: "{:.2f}".format for c in df.columns if c.startswith("Lat_") or c == "s_por_epoca"}
print("\n" + df.to_string(index=False, na_rep="--",
                         formatters={**fmt, "Params": lambda v: f"{int(v):,}" if pd.notna(v) else "--"}))
print(f"\n💾 {CSV_SALIDA}")

# ---------------- latencia de la red completa (para el paper) ----------------
print("\n" + "=" * 78)
print(f"  LATENCIA DE LA RED COMPLETA ({N_FLUJOS} flujos), ms")
print("  GRU: 81 modelos por flujo en secuencia (Lat_red) | Chronos: lote de 81 | ARIMA: 81 reajustes")
print("=" * 78)
for _, r in df.iterrows():
    es_gru = str(r["Modelo"]).startswith("Arch")
    if es_gru and (r["Nivel"], r["W"], int(str(r["H"]).split("+")[1].split()[0])) not in RED_CONFIGS:
        continue
    pre = "Lat_red" if es_gru else "Lat_81"
    vals = {dv: r.get(f"{pre}_{dv}_ms") for dv in DISPOSITIVOS}
    if all(pd.isna(v) for v in vals.values()):
        continue
    print(f"  {r['Nivel']:>4}  {r['Modelo']:<20} W={int(r['W']):<4} {r['H']:<12} "
          + "  ".join(f"{dv} {v:9.2f}" if pd.notna(v) else f"{dv}       --" for dv, v in vals.items()))