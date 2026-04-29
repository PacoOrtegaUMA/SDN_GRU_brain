import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.layers import (Input, GRU, Dense, Dropout,
                                     Conv1D, LeakyReLU, Bidirectional,
                                     Attention, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
CSV_1MIN     = "brain_FINAL_1min.csv"
DIR_MACRO    = "Results_min/results_1h_Arq1_W24"
FECHA_CORTE  = "2013-03-14 00:00:00"
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_min"
ROOT_RESULTS = "Results_min"
# ============================================================


def cargar_macro_1h(enlace):
    ruta = os.path.join(DIR_MACRO, f"test_results_Arq1_GRU_1h_{enlace}.csv")
    if not os.path.exists(ruta):
        return None
    df = pd.read_csv(ruta, parse_dates=["timestamp"])
    return (df[["timestamp", "Prediccion_GRU"]]
              .rename(columns={"Prediccion_GRU": "macro"})
              .set_index("timestamp").sort_index())


def alinear_macro(timestamps_1min, df_macro):
    ts_siguiente_hora = pd.DatetimeIndex(timestamps_1min).ceil("h")
    return np.clip(
        df_macro.reindex(ts_siguiente_hora).ffill().fillna(0)["macro"].values,
        0, None
    )


def preprocesar_test(df_test, enlace, scaler, modo):
    if modo == "A":
        log_vals = np.log1p(df_test[enlace].values)
        feat     = np.diff(log_vals, prepend=log_vals[0]).reshape(-1, 1)
        log1p_vals = log_vals
    elif modo == "B":
        log_vals   = np.log1p(df_test[enlace].values)
        feat       = log_vals.reshape(-1, 1)
        log1p_vals = log_vals
    else:
        raise ValueError(f"modo='{modo}' no reconocido. Usar 'A' o 'B'.")

    feat_sc     = scaler.transform(feat)
    test_scaled = np.hstack([feat_sc, df_test[COLS_TIEMPO].values])
    return test_scaled, log1p_vals


def crear_ventanas_test(data_scaled, log1p_real, macro_array, W, H):
    X_seq, ref_log1p, ref_macro = [], [], []
    for i in range(W, len(data_scaled) - H):
        X_seq.append(data_scaled[i - W : i, :])
        ref_log1p.append(log1p_real[i])
        ref_macro.append(macro_array[i])
    return np.array(X_seq), np.array(ref_log1p), np.array(ref_macro)


def reconstruir(preds_sc, ref_log1p, scaler, modo):
    """
    Modo A → pred es delta escalado → invertir scaler + sumar ref_log1p
    Modo B → pred es log1p escalado → invertir scaler directamente
    """
    pred_orig = scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    if modo == "A":
        return np.clip(np.expm1(ref_log1p + pred_orig), 0, None)
    else:  # B
        return np.clip(np.expm1(pred_orig), 0, None)


def calcular_metricas(y_real, y_pred):
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mask = y_real > 1e-6
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    return mae, rmse, mape


def test(ID, horizonte, ventana, peso_macro, modo="A"):
    assert modo in ("A", "B"), "modo debe ser 'A' o 'B'"

    W, H = ventana, horizonte

    peso_str = str(int(round(peso_macro * 100))).zfill(2)

    # Nombre de carpeta: modo A conserva nombre original (sin sufijo)

    nombre_raiz  = f"Arq{ID}_{H}m_{modo}_W{W}_p{peso_str}_H{horizonte}"
    nombre_arq   = f"models_{nombre_raiz}"
    path_models  = os.path.join(ROOT_MODELS,  nombre_arq)
    path_results = os.path.join(ROOT_RESULTS, f"results_{nombre_raiz}")

    if not os.path.exists(path_models):
        print(f"❌ No existe: {path_models}")
        return
    os.makedirs(path_results, exist_ok=True)

    df      = pd.read_csv(CSV_1MIN, parse_dates=["timestamp"])
    enlaces = [c for c in df.columns if c not in COLS_EXCLUIR + COLS_TIEMPO]
    df_test = df[df["timestamp"] >= FECHA_CORTE].copy().reset_index(drop=True)

    print(f"\n🔍 TEST | {nombre_arq} | modo={modo} | {len(enlaces)} enlaces")
    print(f"   Test: {df_test['timestamp'].min()} → {df_test['timestamp'].max()}")

    resumen = []

    for idx, enlace in enumerate(enlaces, 1):
        ruta_scaler  = os.path.join(path_models, f"Scaler_{enlace}.bin")
        ruta_csv_out = os.path.join(path_results, f"test_{nombre_raiz}_{enlace}.csv")
        ruta_keras   = os.path.join(path_models, f"Model_{enlace}.keras")
        ruta_h5      = os.path.join(path_models, f"Model_{enlace}.h5")

        if   os.path.exists(ruta_keras): ruta_modelo = ruta_keras
        elif os.path.exists(ruta_h5):    ruta_modelo = ruta_h5
        else:
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — falta modelo"); continue

        if not os.path.exists(ruta_scaler):
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — falta scaler"); continue

        if os.path.exists(ruta_csv_out):
            print(f"⏩ [{idx}/{len(enlaces)}] {enlace} — ya existe"); continue

        df_macro = cargar_macro_1h(enlace)
        if df_macro is None:
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — sin macro 1h"); continue

        try:
            scaler     = joblib.load(ruta_scaler)
            macro_test = alinear_macro(df_test["timestamp"].values, df_macro)
            test_s, log1p_te = preprocesar_test(df_test.copy(), enlace, scaler, modo)
            X_te, ref_log1p, ref_macro = crear_ventanas_test(test_s, log1p_te, macro_test, W, H)

            y_real = df_test[enlace].values[W + H - 1 : W + H - 1 + len(X_te)]

            model    = tf.keras.models.load_model(ruta_modelo, compile=False)
            preds_sc = model.predict(X_te, verbose=0)
            del model; tf.keras.backend.clear_session()

            preds_final     = reconstruir(preds_sc, ref_log1p, scaler, modo)
            mae, rmse, mape = calcular_metricas(y_real, preds_final)

            n          = len(X_te)
            timestamps = df_test["timestamp"].values[W + H - 1 : W + H - 1 + n]
            pd.DataFrame({
                "timestamp" : timestamps,
                "Realidad"  : y_real,
                "Prediccion": preds_final,
                "Macro_1h"  : ref_macro,
            }).to_csv(ruta_csv_out, index=False)

            print(f"✅ [{idx}/{len(enlaces)}] {enlace} | MAE:{mae:.2e} RMSE:{rmse:.2e} MAPE:{mape:.1f}%")
            resumen.append({"enlace": enlace,
                            "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)})

        except Exception as e:
            print(f"❌ {enlace}: {e}")
            import traceback; traceback.print_exc()

    if resumen:
        df_res   = pd.DataFrame(resumen)
        ruta_res = os.path.join(path_results, f"resumen_{nombre_arq}.csv")
        df_res.to_csv(ruta_res, index=False)
        print(f"\n📊 {ruta_res}")
        print(df_res[["MAE", "RMSE", "MAPE"]].describe().round(4))

    print(f"\n{'='*40}\n  TEST COMPLETADO: {nombre_arq}\n{'='*40}")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    for modo in ["A", "B"]:
        for ID in [1, 2, 3, 4]:
            for peso in [0.0, 0.25, 0.5, 0.75]:
                for horizon in [5, 15]:
                    test(ID=ID, horizonte=horizon, ventana=60, peso_macro=peso, modo=modo)