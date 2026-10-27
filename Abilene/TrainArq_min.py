# ============================================================
# TrainArq_min_Abilene.py — v1.0  (ABILENE, datos cada 5 min)
# Igual que TrainArq_min_mezcla.py v1.0 (BRAIN), adaptado a Abilene:
#   - CSV abilene_FINAL_5min.csv (110 enlaces, 2004-05-01 → 2004-08-19, paso de 5 min).
#   - Horizonte y ventana se dan en MINUTOS y se pasan a pasos de 5 min:
#       horizonte 5 → 1 paso, 15 → 3 pasos ; ventana 60 → 12 pasos.
#     Los nombres de carpeta llevan los minutos (Arq1_5m_A_W60_w00_H5), como en BRAIN.
#   - FECHA_CORTE 2004-07-29 → ~80 % train / 20 % test.
#   - Con ω = 0 (modelo base) no hace falta la columna MEDIAH_<enlace>.
#
# Objetivo de entrenamiento = mezcla entre el valor real del minuto T+H y la
# media REAL de la hora en la que cae ese minuto:
#
#     objetivo = (1 − ω) · x_{T+H}  +  ω · media_hora(T+H)
#
#   ω = 0    → objetivo = valor real (no se toca nada)
#   ω = 1    → objetivo = media de la hora
#   Ej.: x = 10.000, media = 20.000, ω = 0.25 → objetivo = 12.500
#
# La media sale de la columna MEDIAH_<enlace> de brain_FINAL_1min_con_MEDIAH.csv
# (celda MediaHoraria_desde1min): media de los minutos (T-60, T] de cada hora.
# La red NO recibe la media como entrada: solo los W minutos anteriores y la
# hora del día. La mezcla se hace en bps y después se pasa a log.
#
# Modo A → objetivo = log1p(mezcla) − log1p(x_{i-1})   (respecto al último observado)
# Modo B → objetivo = log1p(mezcla)
# Validación: último 15 % del train. Loss Huber.
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
CSV_1MIN     = "abilene_FINAL_5min.csv"
PASO_MIN     = 5                         # minutos entre muestras
FECHA_CORTE  = "2004-07-29 00:00:00"
VAL_SPLIT    = 0.15
COLS_TIEMPO  = ["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]
COLS_EXCLUIR = ["timestamp", "hour", "day_of_week"]
ROOT_MODELS  = "Models_min_Abilene"
BATCH_SIZE   = 128
EPOCHS       = 100
# ============================================================


def nombre_config(ID, H, W, modo, omega):
    return f"Arq{ID}_{H}m_{modo}_W{W}_w{int(round(omega * 100)):02d}_H{H}"


def preprocesar(df_train, enlace, modo):
    log1p_x = np.log1p(df_train[enlace].values)
    if modo == "A":
        feat = np.diff(log1p_x, prepend=log1p_x[0]).reshape(-1, 1)
    elif modo == "B":
        feat = log1p_x.reshape(-1, 1)
    else:
        raise ValueError(f"modo='{modo}' no reconocido. Usar 'A' o 'B'.")
    scaler = RobustScaler(quantile_range=(10, 90))
    data = np.hstack([scaler.fit_transform(feat), df_train[COLS_TIEMPO].values])
    return data, scaler, log1p_x


def crear_ventanas(data_scaled, x, log1p_x, media, scaler, W, H, modo, omega):
    # Entrada: filas i-W ... i-1 ; objetivo: minuto j = i+H-1
    X_seq, Y = [], []
    for i in range(W, len(data_scaled) - H + 1):
        X_seq.append(data_scaled[i - W : i, :])
        j = i + H - 1
        mezcla = (1 - omega) * x[j] + omega * media[j]
        y = np.log1p(mezcla)
        Y.append(y - log1p_x[i - 1] if modo == "A" else y)
    Y = scaler.transform(np.array(Y).reshape(-1, 1)).ravel()
    return np.array(X_seq), Y


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
        raise ValueError(f"ID={ID} no reconocido. Usar 1, 2, 3 o 4.")

    out = Dense(1, activation='linear')(x)
    lr  = 0.0004 if ID == 4 else 0.0005
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='huber', metrics=['mae'])
    return model


def entrenar(ID, horizonte, ventana, omega, modo="A"):
    assert modo in ("A", "B"), "modo debe ser 'A' o 'B'"
    assert 0.0 <= omega <= 1.0, "omega entre 0 y 1"
    assert horizonte % PASO_MIN == 0 and ventana % PASO_MIN == 0, f"horizonte y ventana múltiplos de {PASO_MIN} min"
    W, H = ventana // PASO_MIN, horizonte // PASO_MIN          # en pasos de 5 min

    nombre_arq  = "models_" + nombre_config(ID, horizonte, ventana, modo, omega)
    path_models = os.path.join(ROOT_MODELS, nombre_arq)
    os.makedirs(path_models, exist_ok=True)

    if not os.path.exists(CSV_1MIN):
        print(f"❌ No se encuentra {CSV_1MIN}")
        return

    df = pd.read_csv(CSV_1MIN, parse_dates=["timestamp"])
    enlaces = [c for c in df.columns
               if c not in COLS_EXCLUIR + COLS_TIEMPO and not c.startswith("MEDIAH_")]
    df_train = df[df["timestamp"] < FECHA_CORTE].copy().reset_index(drop=True)

    print(f"\n🚀 {nombre_arq} | ω={omega:.2f} | modo={modo} | {len(enlaces)} enlaces | W={W} pasos, H={H} pasos")
    print(f"   Train+val: {df_train['timestamp'].min()} → {df_train['timestamp'].max()}")

    ruta_val = os.path.join(path_models, "val_mae.csv")

    for idx, enlace in enumerate(enlaces, 1):
        ext         = ".keras" if ID == 4 else ".h5"
        ruta_modelo = os.path.join(path_models, f"Model_{enlace}{ext}")
        ruta_scaler = os.path.join(path_models, f"Scaler_{enlace}.bin")

        if os.path.exists(ruta_modelo):
            print(f"⏩ [{idx}/{len(enlaces)}] {enlace} — ya existe")
            continue
        col_media = f"MEDIAH_{enlace}"
        if omega > 0 and col_media not in df_train.columns:
            print(f"⚠️  [{idx}/{len(enlaces)}] {enlace} — falta {col_media} en el CSV, saltando")
            continue

        print(f"\n📍 [{idx}/{len(enlaces)}] {enlace}")
        tf.keras.backend.clear_session()

        try:
            x     = df_train[enlace].values.astype(float)
            media = (df_train[col_media].ffill().bfill().fillna(0).values.astype(float)
                     if col_media in df_train.columns else x)          # con ω = 0 no se usa

            data, scaler, log1p_x = preprocesar(df_train, enlace, modo)
            joblib.dump(scaler, ruta_scaler)

            X_tr, Y_tr = crear_ventanas(data, x, log1p_x, media, scaler, W, H, modo, omega)
            n_val = int(len(X_tr) * VAL_SPLIT)
            print(f"   Samples — Train: {len(X_tr) - n_val} | Val: {n_val}")

            model = crear_modelo(ID, (W, X_tr.shape[2]))
            hist = model.fit(
                X_tr, Y_tr,
                validation_split=VAL_SPLIT,
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
    for Id in [1,2,3,4]:
        for modo in ["A","B"]:
            for horizon in [15, 5]:
                for omega in [0.0]:
                    entrenar(ID=Id, horizonte=horizon, ventana=60, omega=omega, modo=modo)