# ============================================================
# TestArq_min_Abilene.py — v1.2  (ABILENE, datos cada 5 min)
# v1.2: REHACER = True vuelve a generar los CSV aunque ya existan (si reentrenas un
#       modelo, el test antiguo no se actualizaba). Imprime qué fichero de modelo carga.
# v1.1: FECHA_CORTE = 2004-08-08 (igual que los baselines) y DIR_MACRO apunta al modelo
#       horario de Abilene (TestArq_h_abilene.py), así que la fuente 'pred' ya tiene valores.
# Test de los modelos de TrainArq_min_Abilene.py. Igual que TestArq_min_mezcla.py v1.1
# (BRAIN), adaptado a Abilene:
#   - Horizonte y ventana en MINUTOS → pasos de 5 min (T+5 = 1 paso, T+15 = 3, W60 = 12).
#     Mismos nombres de carpeta/fichero que en BRAIN (Arq1_5m_A_W60_w00_H5).
#   - Fuente 'pred' = predicción del modelo horario de Abilene (DIR_MACRO). None = vacía.
#   - No hay columna MEDIAH: la media real de la hora del objetivo (fuente "real") se
#     calcula aquí con los datos de 5 min: media de (hora−60 min, hora] que contiene T+H.
#   - movil60 con al menos MIN_MUESTRAS_60 de las 12 muestras de la última hora.
#   - Entrada igual que en el train: W minutos anteriores + hora del día.
#   - Se compara SIEMPRE con el valor real del minuto T+H.
# v1.1: además de la predicción del modelo, guarda el valor horario de cada
#   fuente y la predicción COMBINADA para cada ω:
#       Pred_<fuente>_w<ω> = (1 − ω)·Prediccion + ω·Hora_<fuente>
#   Fuentes (todas referidas al último minuto observado t_obs = t − H):
#       pred     → predicción del modelo horario para la hora que contiene t_obs   (sin fuga)
#       movil60  → media real de los 60 minutos anteriores a t_obs (incluido)     (sin fuga)
#       hora_ant → media real de la última hora de reloj ya terminada en t_obs     (sin fuga)
#       real     → media real de la hora del minuto objetivo (Media_hora)          (CON FUGA: cota)
#   Si un CSV ya existe pero no tiene estas columnas, se rehace.
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# PARÁMETROS GLOBALES (iguales que en TrainArq_min_mezcla.py)
# ============================================================
CSV_1MIN     = "abilene_FINAL_5min.csv"
PASO_MIN     = 5                                         # minutos entre muestras
FECHA_CORTE  = "2004-08-08 00:00:00"
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_min_Abilene"
ROOT_RESULTS = "Results_min_Abilene"
DIR_MACRO    = "Results_horas_abilene/results_Arq1_1h_A_W24"   # salida de TestArq_h_abilene.py (None = sin pred)
MIN_MUESTRAS_60 = 6                                      # de 12 muestras en 60 min
REHACER      = False                                     # True = rehace los CSV aunque existan
OMEGAS_COMB  = [0.25, 0.5, 0.75, 1.0]                    # ω de la combinación que se guardan
FUENTES      = ["pred", "movil60", "hora_ant", "real"]
# ============================================================


def nombre_config(ID, H, W, modo, omega):
    return f"Arq{ID}_{H}m_{modo}_W{W}_w{int(round(omega * 100)):02d}_H{H}"


def preprocesar_test(df_test, enlace, scaler, modo):
    log1p_x = np.log1p(df_test[enlace].values)
    feat = np.diff(log1p_x, prepend=log1p_x[0]) if modo == "A" else log1p_x
    data = np.hstack([scaler.transform(feat.reshape(-1, 1)), df_test[COLS_TIEMPO].values])
    return data, log1p_x


def crear_ventanas_test(data_scaled, log1p_x, W, H):
    X_seq, ref = [], []
    for i in range(W, len(data_scaled) - H + 1):
        X_seq.append(data_scaled[i - W : i, :])
        ref.append(log1p_x[i - 1])                     # último valor observado
    return np.array(X_seq), np.array(ref)


def reconstruir(preds_sc, ref_log1p, scaler, modo):
    p = scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    return np.clip(np.expm1(ref_log1p + p if modo == "A" else p), 0, None)


def calcular_metricas(y_real, y_pred):
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mask = y_real > 1e-6
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    wape = np.abs(y_real - y_pred).sum() / max(y_real.sum(), 1e-9) * 100
    return mae, rmse, mape, wape


_macro = {}


def prediccion_horaria(enlace):
    if DIR_MACRO is None:
        return None
    if enlace not in _macro:
        base = os.path.basename(DIR_MACRO.rstrip("/")).removeprefix("results_").removeprefix("1h_")
        f = os.path.join(DIR_MACRO, f"test_{base}_{enlace}.csv")
        _macro[enlace] = (pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")["Prediccion"]
                          if os.path.exists(f) else None)
    return _macro[enlace]


def fuentes_horarias(serie_completa, enlace, t_obs, media_real):
    """Valor horario de cada fuente para cada instante (t_obs = último minuto observado)."""
    t_obs = pd.DatetimeIndex(t_obs)
    out = {}
    m = prediccion_horaria(enlace)
    out["pred"] = (m.reindex(t_obs.ceil("h")).ffill().bfill().values
                   if m is not None else np.full(len(t_obs), np.nan))
    movil = serie_completa.rolling("60min", min_periods=MIN_MUESTRAS_60).mean()
    out["movil60"] = movil.reindex(t_obs).values
    media_h = serie_completa.resample("1h", label="right", closed="right").mean()
    out["hora_ant"] = media_h.reindex(t_obs.floor("h")).values
    out["real"] = media_real
    return out


def test(ID, horizonte, ventana, omega, modo="A"):
    assert horizonte % PASO_MIN == 0 and ventana % PASO_MIN == 0, f"horizonte y ventana múltiplos de {PASO_MIN} min"
    W, H = ventana // PASO_MIN, horizonte // PASO_MIN          # en pasos de 5 min
    raiz = nombre_config(ID, horizonte, ventana, modo, omega)
    path_models  = os.path.join(ROOT_MODELS,  f"models_{raiz}")
    path_results = os.path.join(ROOT_RESULTS, f"results_{raiz}")

    if not os.path.exists(path_models):
        print(f"❌ No existe: {path_models}")
        return
    os.makedirs(path_results, exist_ok=True)

    df = pd.read_csv(CSV_1MIN, parse_dates=["timestamp"])
    enlaces = [c for c in df.columns
               if c not in COLS_EXCLUIR + COLS_TIEMPO and not c.startswith(("MEDIAH_", "MEDIAHU_"))]
    df_test = df[df["timestamp"] >= FECHA_CORTE].copy().reset_index(drop=True)
    serie_df = df.set_index("timestamp")

    print(f"\n🔍 TEST | {raiz} | ω={omega:.2f} | {len(enlaces)} enlaces")
    resumen = []

    for idx, enlace in enumerate(enlaces, 1):
        ruta_scaler = os.path.join(path_models, f"Scaler_{enlace}.bin")
        ruta_out    = os.path.join(path_results, f"test_{raiz}_{enlace}.csv")
        ruta_keras  = os.path.join(path_models, f"Model_{enlace}.keras")
        ruta_h5     = os.path.join(path_models, f"Model_{enlace}.h5")

        if   os.path.exists(ruta_keras): ruta_modelo = ruta_keras
        elif os.path.exists(ruta_h5):    ruta_modelo = ruta_h5
        else:
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — falta modelo"); continue
        if not REHACER and os.path.exists(ruta_out) and "Hora_pred" in pd.read_csv(ruta_out, nrows=1).columns:
            print(f"⏩ [{idx}/{len(enlaces)}] {enlace} — ya existe"); continue

        try:
            scaler = joblib.load(ruta_scaler)
            data, log1p_x = preprocesar_test(df_test, enlace, scaler, modo)
            X_te, ref = crear_ventanas_test(data, log1p_x, W, H)
            n  = len(X_te)
            sl = slice(W + H - 1, W + H - 1 + n)
            y_real = df_test[enlace].values[sl]
            ts     = df_test["timestamp"].values[sl]
            t_obs  = df_test["timestamp"].values[W - 1 : W - 1 + n]          # último minuto observado
            if f"MEDIAH_{enlace}" in df_test:
                media = df_test[f"MEDIAH_{enlace}"].values[sl]
            else:                                                       # media real de la hora del objetivo
                media_h = serie_df[enlace].resample("1h", label="right", closed="right").mean()
                media = media_h.reindex(pd.DatetimeIndex(ts).ceil("h")).values

            model = tf.keras.models.load_model(ruta_modelo, compile=False)
            preds = reconstruir(model.predict(X_te, verbose=0), ref, scaler, modo)
            del model; tf.keras.backend.clear_session()

            mae, rmse, mape, wape = calcular_metricas(y_real, preds)

            salida = {"timestamp": ts, "Realidad": y_real, "Prediccion": preds, "Media_hora": media}
            serie_completa = serie_df[enlace]                                # solo se usa el pasado de t_obs
            for fu, valores in fuentes_horarias(serie_completa, enlace, t_obs, media).items():
                salida[f"Hora_{fu}"] = valores
                for w in OMEGAS_COMB:
                    salida[f"Pred_{fu}_w{int(round(w * 100)):02d}"] = (1 - w) * preds + w * valores
            pd.DataFrame(salida).to_csv(ruta_out, index=False)

            print(f"✅ [{idx}/{len(enlaces)}] {enlace} | {os.path.basename(ruta_modelo)} | MAE:{mae:.2e} WAPE:{wape:.1f}%")
            resumen.append({"enlace": enlace, "MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape})

        except Exception as e:
            print(f"❌ {enlace}: {e}")
            import traceback; traceback.print_exc()

    if resumen:
        df_res = pd.DataFrame(resumen).round(4)
        df_res.to_csv(os.path.join(path_results, f"resumen_{raiz}.csv"), index=False)
        print(df_res[["MAE", "RMSE", "MAPE", "WAPE"]].describe().round(2))


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    for Id in [1, 2, 3, 4]:
        for modo in ["A", "B"]:
            for horizon in [15, 5]:
                for omega in [0.0]:
                    test(ID=Id, horizonte=horizon, ventana=60, omega=omega, modo=modo)