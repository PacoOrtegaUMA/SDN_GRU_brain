import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Input, GRU, Dense, Dropout,
                                      Conv1D, LeakyReLU, Bidirectional,
                                      Attention, GlobalAveragePooling1D)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
CSV_1MIN     = "brain_FINAL_1min.csv"
DIR_MACRO    = "Results_min/results_1h_Arq3_1h_B_W24"
FECHA_CORTE  = "2013-03-14 00:00:00"
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_min"
BATCH_SIZE   = 128
EPOCHS       = 100
# ============================================================


# ------------------------------------------------------------
# MACRO — sin cambios
# ------------------------------------------------------------
def cargar_macro_1h(enlace):
    # "results_1h_Arq3_1h_B_W24" → "test_1h_Arq3_1h_B_W24"
    prefijo = "test" + os.path.basename(DIR_MACRO).removeprefix("results_1h")
    ruta    = os.path.join(DIR_MACRO, f"{prefijo}_{enlace}.csv")

    if not os.path.exists(ruta):
        print(f"⚠️ No encontrado: {ruta}")
        return None

    df = pd.read_csv(ruta, parse_dates=["timestamp"])
    return (df[["timestamp", "Prediccion"]]
              .rename(columns={"Prediccion": "macro"})
              .set_index("timestamp")
              .sort_index())


def alinear_macro(timestamps_1min, df_macro):
    ts_siguiente_hora = pd.DatetimeIndex(timestamps_1min).ceil("h")
    return np.clip(
        df_macro.reindex(ts_siguiente_hora).ffill().fillna(0)["macro"].values,
        0, None
    )


# ------------------------------------------------------------
# PREPROCESAMIENTO — modo A (delta log) o B (log directo)
# ------------------------------------------------------------
def preprocesar(df_train, df_test, enlace, modo):
    if modo == "A":
        def calcular_delta(series):
            log_vals = np.log1p(series.values)
            return np.diff(log_vals, prepend=log_vals[0])
        feat_train = calcular_delta(df_train[enlace]).reshape(-1, 1)
        feat_test  = calcular_delta(df_test[enlace]).reshape(-1, 1)

    elif modo == "B":
        feat_train = np.log1p(df_train[enlace].values).reshape(-1, 1)
        feat_test  = np.log1p(df_test[enlace].values).reshape(-1, 1)

    else:
        raise ValueError(f"modo='{modo}' no reconocido. Usar 'A' o 'B'.")

    scaler = RobustScaler(quantile_range=(10, 90))
    feat_train_sc = scaler.fit_transform(feat_train)
    feat_test_sc  = scaler.transform(feat_test)

    train_scaled = np.hstack([feat_train_sc, df_train[COLS_TIEMPO].values])
    test_scaled  = np.hstack([feat_test_sc,  df_test[COLS_TIEMPO].values])

    log1p_train = np.log1p(df_train[enlace].values)
    log1p_test  = np.log1p(df_test[enlace].values)

    return train_scaled, test_scaled, scaler, log1p_train, log1p_test


# ------------------------------------------------------------
# VENTANAS
#
# Modo A → Y_ext (N,2): col0=delta_target_sc, col1=delta_macro_sc
# Modo B → Y_ext (N,2): col0=log_target_sc,   col1=delta_macro_sc
#          (col1 siempre se calcula en espacio delta para la penalización)
# ------------------------------------------------------------
def crear_ventanas(data_scaled, log1p_real, macro_array, scaler, W, H, modo):
    X_seq, Y_ext, ref_log1p = [], [], []

    for i in range(W, len(data_scaled) - H):
        X_seq.append(data_scaled[i - W : i, :])

        target_sc        = data_scaled[i + H - 1, 0]           # ya escalado (A o B)
        delta_macro_orig = np.log1p(macro_array[i]) - log1p_real[i]
        delta_macro_sc   = scaler.transform([[delta_macro_orig]])[0, 0]

        Y_ext.append([target_sc, delta_macro_sc])
        ref_log1p.append(log1p_real[i])

    return np.array(X_seq), np.array(Y_ext), np.array(ref_log1p)


# ------------------------------------------------------------
# LOSS CUSTOM — sin cambios
# ------------------------------------------------------------
def hacer_loss(peso_macro):
    def loss_con_anclaje(y_true_combined, y_pred):
        y_true    = y_true_combined[:, 0:1]
        macro_ref = y_true_combined[:, 1:2]
        base_loss = tf.keras.losses.huber(y_true, y_pred)
        penalty   = tf.reduce_mean(tf.abs(y_pred - macro_ref), axis=-1)
        return base_loss + peso_macro * penalty
    return loss_con_anclaje


# ------------------------------------------------------------
# ARQUITECTURAS — sin cambios
# ------------------------------------------------------------
def crear_modelo(ID, input_shape, num_outputs, peso_macro):
    inp = Input(shape=input_shape)

    if ID == 1:
        x = GRU(64, return_sequences=False)(inp)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)

    elif ID == 2:
        x = Conv1D(filters=64, kernel_size=3, padding='same')(inp)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.2)(x)
        x = GRU(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='swish')(x)

    elif ID == 3:
        x = Conv1D(filters=64, kernel_size=3, padding='same')(inp)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(GRU(64, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='swish')(x)

    elif ID == 4:
        x = Conv1D(filters=64, kernel_size=3, padding='same')(inp)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.2)(x)
        gru_out = Bidirectional(GRU(64, return_sequences=True))(x)
        query   = Dense(128)(gru_out)
        value   = Dense(128)(gru_out)
        x       = Attention()([query, value])
        x       = GlobalAveragePooling1D()(x)
        x       = Dense(64, activation='swish')(x)
        x       = Dropout(0.3)(x)
        x       = Dense(32, activation='swish')(x)

    else:
        raise ValueError(f"ID={ID} no reconocido. Usar 1, 2, 3 o 4.")

    out = Dense(num_outputs, activation='linear')(x)
    lr      = 0.0004 if ID == 4 else 0.0005
    loss_fn = hacer_loss(peso_macro)

    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss=loss_fn, metrics=['mae'])
    return model


# ------------------------------------------------------------
# ENTRENAMIENTO PRINCIPAL
# ------------------------------------------------------------
def entrenar(ID, horizonte, ventana, peso_macro, modo="A"):
    W, H = ventana, horizonte

    assert modo in ("A", "B"), "modo debe ser 'A' o 'B'"

    peso_str   = str(int(round(peso_macro * 100))).zfill(2)
    nombre_arq = f"models_Arq{ID}_{H}m_{modo}_W{W}_p{peso_str}_H{horizonte}"
    path_models = os.path.join(ROOT_MODELS, nombre_arq)
    os.makedirs(path_models, exist_ok=True)

    if not os.path.exists(CSV_1MIN):
        print(f"❌ No se encuentra {CSV_1MIN}")
        return

    df = pd.read_csv(CSV_1MIN, parse_dates=["timestamp"])
    cols_excluir = COLS_EXCLUIR + COLS_TIEMPO
    enlaces = [c for c in df.columns if c not in cols_excluir]

    df_train = df[df["timestamp"] <  FECHA_CORTE].copy().reset_index(drop=True)
    df_test  = df[df["timestamp"] >= FECHA_CORTE].copy().reset_index(drop=True)

    print(f"\n🚀 {nombre_arq} | modo={modo} | {len(enlaces)} enlaces")
    print(f"   Train: {df_train['timestamp'].min()} → {df_train['timestamp'].max()}")
    print(f"   Test:  {df_test['timestamp'].min()}  → {df_test['timestamp'].max()}")

    for idx, enlace in enumerate(enlaces, 1):
        ext         = ".keras" if ID == 4 else ".h5"
        ruta_modelo = os.path.join(path_models, f"Model_{enlace}{ext}")
        ruta_scaler = os.path.join(path_models, f"Scaler_{enlace}.bin")

        if os.path.exists(ruta_modelo):
            print(f"⏩ [{idx}/{len(enlaces)}] {enlace} — ya existe")
            continue

        df_macro = cargar_macro_1h(enlace)
        if df_macro is None:
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — sin macro 1h, saltando")
            continue

        print(f"\n📍 [{idx}/{len(enlaces)}] {enlace}")
        tf.keras.backend.clear_session()

        try:
            macro_train = alinear_macro(df_train["timestamp"].values, df_macro)
            macro_test  = alinear_macro(df_test["timestamp"].values,  df_macro)

            train_s, test_s, scaler, log1p_tr, log1p_te = preprocesar(
                df_train.copy(), df_test.copy(), enlace, modo
            )
            joblib.dump(scaler, ruta_scaler)

            X_tr, Y_tr, _ = crear_ventanas(train_s, log1p_tr, macro_train, scaler, W, H, modo)
            X_te, Y_te, _ = crear_ventanas(test_s,  log1p_te, macro_test,  scaler, W, H, modo)

            n_features = X_tr.shape[2]
            print(f"   Samples — Train: {len(X_tr)} | Test: {len(X_te)}")
            print(f"   Target train — mean: {Y_tr[:,0].mean():.4f} | std: {Y_tr[:,0].std():.4f}")

            model = crear_modelo(ID, (W, n_features), 1, peso_macro)
            model.fit(
                X_tr, Y_tr,
                validation_data=(X_te, Y_te),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[
                    EarlyStopping(monitor="val_mae", patience=20,
                                  restore_best_weights=True, verbose=1),
                    ReduceLROnPlateau(monitor="val_mae", factor=0.5,
                                     patience=7, verbose=1),
                ],
                verbose=2,
            )
            model.save(ruta_modelo)
            print(f"✅ Guardado: {ruta_modelo}")

        except Exception as e:
            print(f"❌ Error en {enlace}: {e}")
            import traceback; traceback.print_exc()
        finally:
            tf.keras.backend.clear_session()

    print("\n" + "=" * 40)
    print(f"  COMPLETADO: {nombre_arq}")
    print("=" * 40)


# ============================================================
# PUNTO DE ENTRADA — prueba reducida A vs B
# ============================================================
if __name__ == "__main__":
    for horizon in [5, 15]:
        for modo in ["A", "B"]:
            for ID in [1, 2, 3, 4]:
                for peso in [0.0, 0.25, 0.5, 0.75]:
                    entrenar(ID=ID, horizonte=horizon, ventana=60, peso_macro=peso, modo=modo)


                    