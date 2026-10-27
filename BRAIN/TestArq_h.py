# ============================================================
# TestArq_h.py — v1.1  (BRAIN, nivel horario)
# Cambios respecto a v1.0:
#   1. Lee los modelos de TrainArq_h.py v1.1 (Models_horas_v11) y escribe en
#      Results_horas. Estas predicciones son las anclas del nivel de minutos.
#   2. Las ventanas de test usan como contexto las W horas ANTERIORES al corte:
#      hay predicción desde el primer instante del test. Solo datos pasados.
#   3. Modo A: reconstrucción desde el último valor observado x_{i-1}
#      (v1.0 usaba x_i, que es el propio valor a predecir → fuga).
#   4. Se añade WAPE por enlace.
#   5. CUDA_VISIBLE_DEVICES antes de importar TensorFlow.
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
CSV_1H       = "brain_FINAL_1h.csv"
FECHA_CORTE  = "2013-02-01 00:00:00"
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_horas_v11"
ROOT_RESULTS = "Results_horas"
H            = 1
# ============================================================


def preprocesar_test(df_test, enlace, scaler, modo):
    log_vals = np.log1p(df_test[enlace].values)
    if modo == "A":
        feat = np.diff(log_vals, prepend=log_vals[0]).reshape(-1, 1)
    elif modo == "B":
        feat = log_vals.reshape(-1, 1)
    else:
        raise ValueError(f"modo='{modo}' no reconocido. Usar 'A' o 'B'.")
    feat_sc     = scaler.transform(feat)
    test_scaled = np.hstack([feat_sc, df_test[COLS_TIEMPO].values])
    return test_scaled, log_vals


def crear_ventanas_test(data_scaled, log1p_real, W):
    # Entrada: filas i-W ... i-1 ; objetivo: fila i+H-1 ; último observado: i-1
    X_seq, ref_log1p = [], []
    for i in range(W, len(data_scaled) - H + 1):
        X_seq.append(data_scaled[i - W : i, :])
        ref_log1p.append(log1p_real[i - 1])          # v1.0: último valor observado
    return np.array(X_seq), np.array(ref_log1p)


def reconstruir(preds_sc, ref_log1p, scaler, modo):
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
    wape = np.abs(y_real - y_pred).sum() / max(y_real.sum(), 1e-9) * 100
    return mae, rmse, mape, wape


def test(ID, ventana, modo="B"):
    assert modo in ("A", "B"), "modo debe ser 'A' o 'B'"
    W = ventana

    nombre_arq   = f"models_Arq{ID}_1h_{modo}_W{W}"
    path_models  = os.path.join(ROOT_MODELS,  nombre_arq)
    path_results = os.path.join(ROOT_RESULTS, f"results_Arq{ID}_1h_{modo}_W{W}")

    if not os.path.exists(path_models):
        print(f"❌ No existe: {path_models}")
        return
    os.makedirs(path_results, exist_ok=True)

    df      = pd.read_csv(CSV_1H, parse_dates=["timestamp"])
    enlaces = [c for c in df.columns if c not in COLS_EXCLUIR + COLS_TIEMPO]

    # Test + las W horas anteriores como contexto de la primera ventana
    i_corte = int((df["timestamp"] < FECHA_CORTE).sum())
    df_test = df.iloc[max(0, i_corte - W):].copy().reset_index(drop=True)

    print(f"\n🔍 TEST | {nombre_arq} | modo={modo} | {len(enlaces)} enlaces")
    print(f"   Contexto desde {df_test['timestamp'].iloc[0]} | "
          f"predicciones {FECHA_CORTE} → {df_test['timestamp'].max()}")

    resumen = []

    for idx, enlace in enumerate(enlaces, 1):
        ruta_scaler  = os.path.join(path_models, f"Scaler_{enlace}.bin")
        ruta_csv_out = os.path.join(path_results, f"test_Arq{ID}_1h_{modo}_W{W}_{enlace}.csv")
        ruta_keras   = os.path.join(path_models,  f"Model_{enlace}.keras")
        ruta_h5      = os.path.join(path_models,  f"Model_{enlace}.h5")

        if   os.path.exists(ruta_keras): ruta_modelo = ruta_keras
        elif os.path.exists(ruta_h5):    ruta_modelo = ruta_h5
        else:
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — falta modelo"); continue

        if not os.path.exists(ruta_scaler):
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — falta scaler"); continue

        if os.path.exists(ruta_csv_out):
            print(f"⏩ [{idx}/{len(enlaces)}] {enlace} — ya existe"); continue

        try:
            scaler           = joblib.load(ruta_scaler)
            test_s, log1p_te = preprocesar_test(df_test.copy(), enlace, scaler, modo)
            X_te, ref_log1p  = crear_ventanas_test(test_s, log1p_te, W)

            n          = len(X_te)
            y_real     = df_test[enlace].values[W + H - 1 : W + H - 1 + n]
            timestamps = df_test["timestamp"].values[W + H - 1 : W + H - 1 + n]

            model    = tf.keras.models.load_model(ruta_modelo, compile=False)
            preds_sc = model.predict(X_te, verbose=0)
            del model; tf.keras.backend.clear_session()

            preds_final = reconstruir(preds_sc, ref_log1p, scaler, modo)
            mae, rmse, mape, wape = calcular_metricas(y_real, preds_final)

            pd.DataFrame({
                "timestamp" : timestamps,
                "Realidad"  : y_real,
                "Prediccion": preds_final,
            }).to_csv(ruta_csv_out, index=False)

            print(f"✅ [{idx}/{len(enlaces)}] {enlace} | MAE:{mae:.2e} RMSE:{rmse:.2e} "
                  f"MAPE:{mape:.1f}% WAPE:{wape:.1f}%")
            resumen.append({"enlace": enlace, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                            "MAPE": round(mape, 4), "WAPE": round(wape, 4)})

        except Exception as e:
            print(f"❌ {enlace}: {e}")
            import traceback; traceback.print_exc()

    if resumen:
        df_res   = pd.DataFrame(resumen)
        ruta_res = os.path.join(path_results, f"resumen_{nombre_arq}.csv")
        df_res.to_csv(ruta_res, index=False)
        print(f"\n📊 {ruta_res}")
        print(df_res[["MAE", "RMSE", "MAPE", "WAPE"]].describe().round(4))

    print(f"\n{'='*40}\n  TEST COMPLETADO: {nombre_arq}\n{'='*40}")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    for ventana in [24, 168]:
        for ID in [1, 2, 3, 4]:
            for modo in ["A", "B"]:
                 test(ID, ventana=ventana, modo=modo)
