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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
CSV_1H       = "brain_FINAL_1h.csv"
FECHA_CORTE  = "2013-02-01 00:00:00"
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_horas"
BATCH_SIZE   = 128
EPOCHS       = 100
H            = 1
# ============================================================
#
# MODOS DE PREPROCESAMIENTO:
#   "A" → delta log: scaler sobre Δlog1p(x), reconstrucción acumulada
#   "B" → log directo: scaler sobre log1p(x), reconstrucción directa
# ============================================================


# ------------------------------------------------------------
# PREPROCESAMIENTO
# ------------------------------------------------------------
def preprocesar(df_train, df_test, enlace, modo):
    if modo == "A":
        # Diferencia del logaritmo (como en minutos)
        def calcular_delta(series):
            log_vals = np.log1p(series.values)
            return np.diff(log_vals, prepend=log_vals[0])

        feat_train = calcular_delta(df_train[enlace]).reshape(-1, 1)
        feat_test  = calcular_delta(df_test[enlace]).reshape(-1, 1)

    elif modo == "B":
        # Logaritmo directo (como en el train vainilla)
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
# ------------------------------------------------------------
def crear_ventanas(data_scaled, W):
    X_seq, Y_seq = [], []
    for i in range(W, len(data_scaled) - H):
        X_seq.append(data_scaled[i - W : i, :])
        Y_seq.append(data_scaled[i + H - 1, 0])
    return np.array(X_seq), np.array(Y_seq)


# ------------------------------------------------------------
# ARQUITECTURAS
# ------------------------------------------------------------
def crear_modelo(ID, input_shape):
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
        raise ValueError(f"ID={ID} no reconocido.")

    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                  loss='huber', metrics=['mae'])
    return model


# ------------------------------------------------------------
# ENTRENAMIENTO PRINCIPAL
# ------------------------------------------------------------
def entrenar(ID, ventana, modo="A"):
    W = ventana

    assert modo in ("A", "B"), "modo debe ser 'A' o 'B'"

    nombre_arq  = f"models_Arq{ID}_1h_{modo}_W{W}"
    path_models = os.path.join(ROOT_MODELS, nombre_arq)
    os.makedirs(path_models, exist_ok=True)

    if not os.path.exists(CSV_1H):
        print(f"❌ No se encuentra {CSV_1H}")
        return

    df = pd.read_csv(CSV_1H, parse_dates=["timestamp"])

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

        print(f"\n📍 [{idx}/{len(enlaces)}] {enlace}")
        tf.keras.backend.clear_session()

        try:
            train_s, test_s, scaler, _, _ = preprocesar(
                df_train.copy(), df_test.copy(), enlace, modo
            )
            joblib.dump(scaler, ruta_scaler)

            X_tr, Y_tr = crear_ventanas(train_s, W)
            X_te, Y_te = crear_ventanas(test_s,  W)

            n_features = X_tr.shape[2]
            print(f"   Samples — Train: {len(X_tr)} | Test: {len(X_te)}")

            model = crear_modelo(ID, (W, n_features))
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
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    for ventana in [24, 168]:
        for ID in [1, 2, 3, 4]:
            for modo in ["A", "B"]:
                entrenar(ID=ID, ventana=ventana, modo=modo)