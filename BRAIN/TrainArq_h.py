# ============================================================
# TrainArq_h.py — v1.1  (BRAIN, nivel horario)
# Cambios respecto a v1.0:
#   1. Validación: último 15 % del train (validation_split). Antes se usaba
#      el test como validación (early stopping mirando el test → fuga).
#   2. Early stopping / ReduceLROnPlateau con mode="min".
#   3. Carpeta de modelos nueva (Models_horas_v11) para no pisar los antiguos.
#   4. CUDA_VISIBLE_DEVICES antes de importar TensorFlow.
#   5. Por defecto solo se entrena el modelo que da el ancla (Arq3, W24, B).
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
CSV_1H       = "brain_FINAL_1h.csv"
FECHA_CORTE  = "2013-02-01 00:00:00"
VAL_SPLIT    = 0.15                     # último 15 % del train, sin barajar
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_horas_v11"
BATCH_SIZE   = 128
EPOCHS       = 100
H            = 1
# ============================================================
#   "A" → delta log: scaler sobre Δlog1p(x)
#   "B" → log directo: scaler sobre log1p(x)
# ============================================================


def preprocesar(df_train, enlace, modo):
    log_vals = np.log1p(df_train[enlace].values)
    if modo == "A":
        feat = np.diff(log_vals, prepend=log_vals[0]).reshape(-1, 1)
    elif modo == "B":
        feat = log_vals.reshape(-1, 1)
    else:
        raise ValueError(f"modo='{modo}' no reconocido. Usar 'A' o 'B'.")

    scaler = RobustScaler(quantile_range=(10, 90))
    feat_sc = scaler.fit_transform(feat)
    return np.hstack([feat_sc, df_train[COLS_TIEMPO].values]), scaler


def crear_ventanas(data_scaled, W):
    # Entrada: filas i-W ... i-1 ; objetivo: fila i+H-1
    X_seq, Y_seq = [], []
    for i in range(W, len(data_scaled) - H + 1):
        X_seq.append(data_scaled[i - W : i, :])
        Y_seq.append(data_scaled[i + H - 1, 0])
    return np.array(X_seq), np.array(Y_seq)


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


def entrenar(ID, ventana, modo="B"):
    assert modo in ("A", "B"), "modo debe ser 'A' o 'B'"
    W = ventana

    nombre_arq  = f"models_Arq{ID}_1h_{modo}_W{W}"
    path_models = os.path.join(ROOT_MODELS, nombre_arq)
    os.makedirs(path_models, exist_ok=True)

    if not os.path.exists(CSV_1H):
        print(f"❌ No se encuentra {CSV_1H}")
        return

    df = pd.read_csv(CSV_1H, parse_dates=["timestamp"])
    enlaces = [c for c in df.columns if c not in COLS_EXCLUIR + COLS_TIEMPO]

    # Solo el train; el test no se toca en este script
    df_train = df[df["timestamp"] < FECHA_CORTE].copy().reset_index(drop=True)

    print(f"\n🚀 {nombre_arq} | modo={modo} | {len(enlaces)} enlaces")
    print(f"   Train+val: {df_train['timestamp'].min()} → {df_train['timestamp'].max()}"
          f" (val = último {int(VAL_SPLIT*100)} %)")

    ruta_val = os.path.join(path_models, "val_mae.csv")

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
            train_s, scaler = preprocesar(df_train.copy(), enlace, modo)
            joblib.dump(scaler, ruta_scaler)

            X_tr, Y_tr = crear_ventanas(train_s, W)
            n_val = int(len(X_tr) * VAL_SPLIT)
            print(f"   Samples — Train: {len(X_tr) - n_val} | Val: {n_val}")

            model = crear_modelo(ID, (W, X_tr.shape[2]))
            hist = model.fit(
                X_tr, Y_tr,
                validation_split=VAL_SPLIT,          # Keras toma el ÚLTIMO tramo
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[
                    EarlyStopping(monitor="val_mae", mode="min", patience=20,
                                  restore_best_weights=True, verbose=1),
                    ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5,
                                      patience=7, verbose=1),
                ],
                verbose=2,
            )
            model.save(ruta_modelo)

            val_mae = min(hist.history["val_mae"])
            with open(ruta_val, "a") as f:
                f.write(f"{enlace},{val_mae}\n")
            print(f"✅ Guardado: {ruta_modelo} | val_mae={val_mae:.4f}")

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
    # Barrido completo (solo si se quiere rehacer la tabla horaria):
    for ventana in [24, 168]:
        for ID in [1, 2, 3, 4]:
            for modo in ["A", "B"]:
                 entrenar(ID=ID, ventana=ventana, modo=modo)