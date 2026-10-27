# ============================================================
# Semillas_Arch1.py — v1.1  (BRAIN, nivel de minutos, GPU)
# v1.1: - Error 'JIT compilation failed [Op:Sign]': a TensorFlow le falta libdevice (el mismo
#         problema que en TestArq_h). El script lo busca solo (paquete nvidia-cuda-nvcc del entorno,
#         $CONDA_PREFIX/nvvm o /usr/local/cuda*) y pone XLA_FLAGS antes de importar TensorFlow.
#         Si no lo encuentra, para y dice cómo instalarlo (pip install nvidia-cuda-nvcc-cu12).
#       - DISPOSITIVO = "CPU" como alternativa (más lento).
#       - Si fallan 3 flujos seguidos, para (antes seguía y recorría los 81 sin hacer nada).
# Estabilidad frente a la semilla aleatoria (R4.8): se reentrena Arch1_A (ω = 0, W = 60) con
# SEMILLAS distintas en T+5 y T+15, se testea y se resume con media ± desviación típica,
# incluyendo la ejecución original (Results_min_Base).
#
# Entrenamiento IDÉNTICO al de los modelos base (TrainArq_min_mezcla v1.3 con ω = 0):
#   brain_FINAL_1min_con_MEDIAH.csv, corte 2013-03-14, validación = último 15 %, RobustScaler(10–90)
#   sobre log(1+x) diferenciado, 5 variables de calendario, objetivo log1p(x_{t+H}) − log1p(x_t)
#   escalado, Huber, Adam (lr 5e-4, clipnorm 1), batch 128, 100 épocas, EarlyStopping(val_mae,
#   patience 20, restore_best_weights), ReduceLROnPlateau(val_mae, factor 0.5, patience 7).
# Lo único que cambia es la semilla (tf.keras.utils.set_random_seed antes de cada modelo):
#   inicialización de los pesos, dropout y orden de los lotes.
#
# HACER = ["entrenar", "resumen"]
#   "entrenar": entrena y testea cada (H, semilla). Si se corta, al relanzar sigue donde iba
#               (salta los modelos y CSV que ya existen).
#       Modelos:    Models_min_semillas/models_Arq1_{H}m_A_W60_w00_H{H}_s{semilla}/
#       Resultados: Results_min_semillas/results_Arq1_{H}m_A_W60_w00_H{H}_s{semilla}/  (mismo formato)
#   "resumen" (sin GPU): para la original y cada semilla, en los instantes de las Tablas 4–6
#       (Realidad > 1000, con ARIMA y Chronos): WAPE global, MAE/RMSE/MAPE medios por flujo,
#       % de flujos que gana a ARIMA y a Chronos, y con la combinación horaria (Arch1_A W24):
#       WAPE global y % de flujos que gana a Chronos para cada ω de OMEGAS.
#       Imprime cada ejecución y media ± desviación típica, y guarda Semillas_Arch1_resumen.csv.
# Tiempo: con Arch1 (~1.1 s/época) son ~1–2.5 h por horizonte y semilla → toda la noche.
# Ejecutar en la carpeta de BRAIN:  python Semillas_Arch1.py
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import glob


def buscar_libdevice():
    """Carpeta CUDA que contiene nvvm/libdevice/libdevice.10.bc (la que necesita TF para tf.sign en GPU)."""
    base = os.environ.get("CONDA_PREFIX", sys.prefix)
    cand = (glob.glob(os.path.join(base, "lib", "python3*", "site-packages", "nvidia", "cuda_nvcc")) + [base] +
            sorted(glob.glob("/usr/local/cuda*"), reverse=True) + ["/usr/lib/cuda", "/opt/cuda"])
    return next((c for c in cand if os.path.exists(os.path.join(c, "nvvm", "libdevice", "libdevice.10.bc"))), None)


DISPOSITIVO = "GPU"         # "GPU" | "CPU" (CPU: mucho más lento, solo si no hay forma de arreglar libdevice)
if DISPOSITIVO == "GPU" and "XLA_FLAGS" not in os.environ:
    _d = buscar_libdevice()
    if _d:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_d}"
        print(f"✅ libdevice encontrado en {_d}")
    else:
        raise SystemExit(
            "❌ TensorFlow no encuentra libdevice (error 'JIT compilation failed [Op:Sign]').\n"
            "   En el entorno GRU:  pip install nvidia-cuda-nvcc-cu12   y vuelve a lanzar el script.\n"
            "   (o pon DISPOSITIVO = \"CPU\", que funciona pero tarda varias veces más)")

import numpy as np
import pandas as pd

# ---------------- configuración ----------------
HACER        = ["entrenar", "resumen"]
SEMILLAS     = [1, 2, 3,4,5,6,7,8,9]
HORIZONTES   = [5, 15]
W            = 60
CSV_1MIN     = "brain_FINAL_1min_con_MEDIAH.csv"
FECHA_CORTE  = "2013-03-14 00:00:00"
VAL_SPLIT    = 0.15
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
BATCH_SIZE, EPOCHS = 128, 100
ROOT_MODELS  = "Models_min_semillas"
ROOT_RES     = "Results_min_semillas"
NOMBRE       = "Arq1_{H}m_A_W60_w00_H{H}"
# resumen (igual que las tablas y figuras del paper)
ROOT_BASE    = "Results_min_Base"
DIR_BASELINES = "Results_min_Base/results_baselines_1min"
DIR_CHRONOS  = "Results_min_Base/results_chronos-bolt-small_C60_{H}m"
DIR_HORARIO  = "Results_horas/results_Arq1_1h_A_W24"
TRIGGER      = 1000
OMEGAS       = [0.05, 0.10, 0.15]
SALIDA       = "Semillas_Arch1_resumen.csv"


# ============================================================
# ENTRENAR Y TESTEAR
# ============================================================
def entrenar_y_testear():
    import joblib
    import tensorflow as tf
    from sklearn.preprocessing import RobustScaler
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Input, GRU, Dense, Dropout

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    def crear_modelo(input_shape):                    # Arch1, igual que en TrainArq_min_mezcla
        inp = Input(shape=input_shape)
        x = GRU(64, return_sequences=False)(inp)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu")(x)
        out = Dense(1, activation="linear")(x)
        m = Model(inp, out)
        m.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), loss="huber", metrics=["mae"])
        return m

    def entrada(df_part, e, scaler=None):
        log1p_x = np.log1p(df_part[e].values.astype(float))
        feat = np.diff(log1p_x, prepend=log1p_x[0]).reshape(-1, 1)
        if scaler is None:
            scaler = RobustScaler(quantile_range=(10, 90)).fit(feat)
        return np.hstack([scaler.transform(feat), df_part[COLS_TIEMPO].values]), log1p_x, scaler

    df = pd.read_csv(CSV_1MIN, parse_dates=["timestamp"])
    flujos = [c for c in df.columns
              if c not in COLS_EXCLUIR + COLS_TIEMPO and not c.startswith(("MEDIAH_", "MEDIAHU_"))]
    df_tr = df[df["timestamp"] < FECHA_CORTE].reset_index(drop=True)
    df_te = df[df["timestamp"] >= FECHA_CORTE].reset_index(drop=True)
    print(f"{len(flujos)} flujos | train {df_tr['timestamp'].min()} → {df_tr['timestamp'].max()} | "
          f"test {df_te['timestamp'].min()} → {df_te['timestamp'].max()}")

    disp = "/GPU:0" if DISPOSITIVO == "GPU" else "/CPU:0"
    errores_seguidos = 0
    for H in HORIZONTES:
        for s in SEMILLAS:
            n = NOMBRE.format(H=H)
            dm = os.path.join(ROOT_MODELS, f"models_{n}_s{s}")
            dr = os.path.join(ROOT_RES, f"results_{n}_s{s}")
            os.makedirs(dm, exist_ok=True); os.makedirs(dr, exist_ok=True)
            print(f"\n🚀 T+{H} | semilla {s}")
            for k, e in enumerate(flujos, 1):
                f_mod = os.path.join(dm, f"Model_{e}.keras")
                f_sc = os.path.join(dm, f"Scaler_{e}.bin")
                f_out = os.path.join(dr, f"test_{n}_s{s}_{e}.csv")
                if os.path.exists(f_out):
                    print(f"⏩ [{k}/{len(flujos)}] {e} — ya existe"); continue
                try:
                    tf.keras.backend.clear_session()
                    data, lx, sc = entrada(df_tr, e)
                    if os.path.exists(f_mod) and os.path.exists(f_sc):
                        sc = joblib.load(f_sc)
                        model = tf.keras.models.load_model(f_mod, compile=False)
                    else:
                        X, Y = [], []
                        for i in range(W, len(data) - H + 1):
                            X.append(data[i - W:i, :])
                            Y.append(lx[i + H - 1] - lx[i - 1])
                        X = np.array(X)
                        Y = sc.transform(np.array(Y).reshape(-1, 1)).ravel()
                        tf.keras.utils.set_random_seed(1000 * s + H)          # lo único que cambia
                        with tf.device(disp):
                            model = crear_modelo((W, X.shape[2]))
                            h = model.fit(X, Y, validation_split=VAL_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                          callbacks=[EarlyStopping(monitor="val_mae", mode="min", patience=20,
                                                                   restore_best_weights=True),
                                                     ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5,
                                                                       patience=7)],
                                          verbose=0)
                        model.save(f_mod); joblib.dump(sc, f_sc)
                        print(f"   [{k}/{len(flujos)}] {e}: {len(h.history['loss'])} épocas, "
                              f"val_mae {min(h.history['val_mae']):.4f}")
                    # test (igual que TestArq_min_mezcla)
                    dte, lxe, _ = entrada(df_te, e, sc)
                    Xt, ref = [], []
                    for i in range(W, len(dte) - H + 1):
                        Xt.append(dte[i - W:i, :]); ref.append(lxe[i - 1])
                    Xt, ref = np.array(Xt), np.array(ref)
                    p = sc.inverse_transform(model.predict(Xt, verbose=0).reshape(-1, 1)).ravel()
                    pred = np.clip(np.expm1(np.array(ref) + p), 0, None)
                    sl = slice(W + H - 1, W + H - 1 + len(Xt))
                    pd.DataFrame({"timestamp": df_te["timestamp"].values[sl], "Realidad": df_te[e].values[sl],
                                  "Prediccion": pred}).to_csv(f_out, index=False)
                    errores_seguidos = 0
                except Exception as ex:
                    print(f"❌ {e}: {ex}")
                    errores_seguidos += 1
                    if errores_seguidos >= 3:
                        raise SystemExit("❌ 3 errores seguidos: paro para no recorrer todos los flujos en vano.")
                finally:
                    tf.keras.backend.clear_session()


# ============================================================
# RESUMEN (sin GPU)
# ============================================================
def resumen():
    filas = []
    for H in HORIZONTES:
        n = NOMBRE.format(H=H)
        ejec = {"original": (os.path.join(ROOT_BASE, f"results_{n}"), f"test_{n}_")}
        for s in SEMILLAS:
            ejec[f"semilla {s}"] = (os.path.join(ROOT_RES, f"results_{n}_s{s}"), f"test_{n}_s{s}_")
        dc = DIR_CHRONOS.format(H=H)
        pc = "test_" + os.path.basename(dc).removeprefix("results_") + "_"
        ph = "test_" + os.path.basename(DIR_HORARIO).removeprefix("results_") + "_"
        flujos = sorted(f[len(pc):-4] for f in os.listdir(dc) if f.startswith(pc) and f.endswith(".csv"))
        for nombre, (d, pref) in ejec.items():
            if not os.path.isdir(d):
                print(f"⚠️ falta {d}"); continue
            acc = {"y": 0.0, "m": 0.0, "a": 0.0, "c": 0.0, **{w: 0.0 for w in OMEGAS}}
            mae, rmse, mape, gana_a, gana_c, n_f = [], [], [], 0, 0, 0
            gana_w = {w: 0 for w in OMEGAS}
            for e in flujos:
                f = os.path.join(d, f"{pref}{e}.csv")
                fb = os.path.join(DIR_BASELINES, f"test_results_{e}_baselines.csv")
                fh = os.path.join(DIR_HORARIO, f"{ph}{e}.csv")
                if not all(os.path.exists(x) for x in (f, fb, fh)):
                    continue
                j = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")[["Realidad", "Prediccion"]]
                mh = pd.read_csv(fh, parse_dates=["timestamp"]).set_index("timestamp")["Prediccion"]
                j["Hora"] = mh.reindex((j.index - pd.Timedelta(minutes=H)).ceil("h")).ffill().bfill().values
                j = j.join(pd.read_csv(fb, parse_dates=["timestamp"]).set_index("timestamp")[[f"ARIMA_t+{H}"]]
                             .rename(columns={f"ARIMA_t+{H}": "ARIMA"}), how="inner")
                j = j.join(pd.read_csv(os.path.join(dc, f"{pc}{e}.csv"), parse_dates=["timestamp"])
                             .set_index("timestamp")[["Prediccion"]].rename(columns={"Prediccion": "Chronos"}),
                           how="inner").dropna()
                j = j[j["Realidad"] > TRIGGER]
                if j.empty:
                    continue
                y, p = j["Realidad"].values, j["Prediccion"].values
                em, ea, ec = np.abs(y - p), np.abs(y - j["ARIMA"].values), np.abs(y - j["Chronos"].values)
                acc["y"] += y.sum(); acc["m"] += em.sum(); acc["a"] += ea.sum(); acc["c"] += ec.sum()
                mae.append(em.mean()); rmse.append(np.sqrt((em ** 2).mean())); mape.append(100 * (em / y).mean())
                gana_a += em.sum() < ea.sum(); gana_c += em.sum() < ec.sum(); n_f += 1
                for w in OMEGAS:
                    ew = np.abs(y - ((1 - w) * p + w * j["Hora"].values))
                    acc[w] += ew.sum(); gana_w[w] += ew.sum() < ec.sum()
            if n_f == 0:
                continue
            fila = {"H": f"T+{H}", "Ejecución": nombre, "Flujos": n_f,
                    "WAPE": 100 * acc["m"] / acc["y"], "MAE": np.mean(mae), "RMSE": np.mean(rmse),
                    "MAPE": np.mean(mape), "%gana_ARIMA": 100 * gana_a / n_f, "%gana_Chronos": 100 * gana_c / n_f,
                    "WAPE_ARIMA": 100 * acc["a"] / acc["y"], "WAPE_Chronos": 100 * acc["c"] / acc["y"]}
            for w in OMEGAS:
                fila[f"WAPE_w{w:.2f}"] = 100 * acc[w] / acc["y"]
                fila[f"%gana_Chronos_w{w:.2f}"] = 100 * gana_w[w] / n_f
            filas.append(fila)

    df = pd.DataFrame(filas)
    df.to_csv(SALIDA, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    cols = [c for c in df.columns if c not in ("H", "Ejecución", "Flujos", "WAPE_ARIMA", "WAPE_Chronos")]
    for H, g in df.groupby("H", sort=False):
        print(f"\n{'=' * 100}\n  Arch1_A, {H}: original + semillas {SEMILLAS}  "
              f"(referencias: WAPE ARIMA {g['WAPE_ARIMA'].iloc[0]:.1f} | Chronos {g['WAPE_Chronos'].iloc[0]:.1f})\n{'=' * 100}")
        t = g.set_index("Ejecución")[cols]
        print(t.round(2).to_string())
        print("\n  media ± desv. típica (todas las ejecuciones):")
        print("  " + " | ".join(f"{c} {t[c].mean():.2f} ± {t[c].std(ddof=1):.2f}" for c in cols))
    print(f"\n💾 {SALIDA}")


if __name__ == "__main__":
    if "entrenar" in HACER:
        entrenar_y_testear()
    if "resumen" in HACER:
        resumen()