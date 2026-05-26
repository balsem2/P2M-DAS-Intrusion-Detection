##lstm_vf
import os
import random
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIG
# =========================
DATA_DIR = r"D:\Telechargement\DAS_dataset_active\data" # <-- à modifier
SEQ_LEN = 8192
PRED_LEN = 1024
N_KEEP = 8
STRIDE = 2048
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# UTILITIES
# =========================
def find_h5_files(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    if not files:
        raise FileNotFoundError(f"Aucun fichier .h5 trouvé dans: {root_dir}")
    return files

def find_rawdata_dataset(h5_file):
    candidates = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.lower().endswith("rawdata"):
            candidates.append(obj)

    h5_file.visititems(visitor)

    if not candidates:
        raise KeyError("Dataset RawData introuvable dans ce fichier HDF5.")

    # On prend le plus probable
    candidates = sorted(candidates, key=lambda d: len(d.shape), reverse=True)
    return candidates[0]

def load_raw_signal(h5_path, n_keep):
    with h5py.File(h5_path, "r") as f:
        ds = find_rawdata_dataset(f)
        n_total = ds.shape[1] if len(ds.shape) > 1 else 1

        if n_total == 1:
            raw = ds[()]  # cas 1D
            raw = raw[:, None]
        else:
            # --- CORRECTION : Canaux contigus au lieu de linspace ---
            # Si c'est le fichier filtré (ex: 170 canaux), on prend les 16 du milieu
            if n_total > n_keep:
                debut_idx = (n_total - n_keep) // 2
                raw = ds[:, debut_idx:debut_idx + n_keep]
            else:
                raw = ds[:, :n_total]

    raw = np.asarray(raw, dtype=np.float32)

    if raw.shape[0] < raw.shape[1]:
        raw = raw.T

    # --- CORRECTION : Passer à l'enveloppe du signal (Valeur Absolue) ---
    # Permet de prédire l'intensité de l'énergie acoustique, ce qui est prédictible
    raw = np.abs(raw) 

    return raw

def normalize_x_y(x, y):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    y = (y - mean) / std
    return x, y
def normalize_file(raw):
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-6
    return (raw - mean) / std, mean, std

def count_windows_in_file(h5_path, seq_len, pred_len, stride):
    with h5py.File(h5_path, "r") as f:
        ds = find_rawdata_dataset(f)
        shape = ds.shape

    if len(shape) == 1:
        time_len = shape[0]
    else:
        # Heuristique simple: la plus grande dimension est le temps
        time_len = max(shape)

    last_start = time_len - seq_len - pred_len + 1
    if last_start <= 0:
        return 0
    return ((last_start - 1) // stride) + 1

def split_files(file_list):
    if len(file_list) < 3:
        raise ValueError("Il faut au moins 3 fichiers pour faire train/val/test proprement.")

    train_files, temp_files = train_test_split(
        file_list, test_size=0.2, random_state=SEED, shuffle=True
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=SEED, shuffle=True
    )
    return train_files, val_files, test_files

# =========================
# GENERATOR
# =========================
def compute_envelope(raw_signal, window_size=512): # Gardez bien window_size=512 ici !
    abs_sig = np.abs(raw_signal)
    envelope = np.zeros_like(abs_sig)
    
    # Un simple filtre de moyenne mobile par canal
    kernel = np.ones(window_size) / window_size
    for c in range(abs_sig.shape[1]):
        envelope[:, c] = np.convolve(abs_sig[:, c], kernel, mode='same')
        
    return envelope
def window_generator(file_paths, seq_len, pred_len, n_keep, stride):
    for path in file_paths:
        # 1. Chargement des données brutes
        raw = load_raw_signal(path, n_keep)
        
        # 2. Normalisation de X (le signal en entrée)
        raw, mean, std = normalize_file(raw)
        
        # 3. NOUVEAU : Calcul de l'enveloppe cible Y très lissée (fenêtre de 512)
        envelope = compute_envelope(raw, window_size=256)

        n_samples = raw.shape[0]
        last_start = n_samples - seq_len - pred_len + 1
        if last_start <= 0:
            continue

        for start in range(0, last_start, stride):
            # X reste le signal normalisé
            x = raw[start : start + seq_len]
            
            # Y devient l'enveloppe ultra-lissée du futur
            y = envelope[start + seq_len : start + seq_len + pred_len]
            
            yield x.astype(np.float32), y.astype(np.float32)
def window_generator(file_paths, seq_len, pred_len, n_keep, stride):
    for path in file_paths:
        # 1. Chargement des données brutes
        raw = load_raw_signal(path, n_keep)
        
        # 2. Normalisation de X (le signal en entrée)
        raw, mean, std = normalize_file(raw)
        
        # 3. NOUVEAU : Calcul de l'enveloppe cible Y très lissée (fenêtre de 512)
        envelope = compute_envelope(raw, window_size=256)

        n_samples = raw.shape[0]
        last_start = n_samples - seq_len - pred_len + 1
        if last_start <= 0:
            continue

        for start in range(0, last_start, stride):
            # X reste le signal normalisé
            x = raw[start : start + seq_len]
            
            # Y devient l'enveloppe ultra-lissée du futur
            y = envelope[start + seq_len : start + seq_len + pred_len]
            
            yield x.astype(np.float32), y.astype(np.float32)

def make_dataset(file_paths, training=False):
    output_signature = (
        tf.TensorSpec(shape=(SEQ_LEN, N_KEEP), dtype=tf.float32),
        tf.TensorSpec(shape=(PRED_LEN, N_KEEP), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: window_generator(file_paths, SEQ_LEN, PRED_LEN, N_KEEP, STRIDE),
        output_signature=output_signature
    )

    if training:
        # 256 ou 512 suffisent largement à mélanger localement sans saturer le CPU/RAM au départ
        ds = ds.shuffle(512, reshuffle_each_iteration=True)
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# =========================
# MODEL
# =========================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, N_KEEP)),
        
        # 1. Réduction 8192 -> 4096
        tf.keras.layers.Conv1D(32, kernel_size=7, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # 2. Réduction 4096 -> 2048
        tf.keras.layers.Conv1D(64, kernel_size=5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # 3. Réduction 2048 -> 1024
        tf.keras.layers.Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # 4. Compréhension Temporelle
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        
        # 5. Lissage (Taille de noyau plus grande pour éviter les pics de bruit)
        tf.keras.layers.Conv1D(32, kernel_size=7, padding='same', activation='relu'),
        
        # 6. Sortie (CRUCIAL : activation 'linear', pas de BatchNormalization avant)
        tf.keras.layers.Conv1D(N_KEEP, kernel_size=3, padding='same', activation='linear') 
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0), 
        metrics=["mae"]
    )
    return model
# =========================
# MAIN
# =========================
def main():
    files = find_h5_files(DATA_DIR)
    print(f"Nombre de fichiers trouvés: {len(files)}")

    train_files, val_files, test_files = split_files(files)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Comptage approximatif des fenêtres pour définir steps_per_epoch
    train_windows = sum(count_windows_in_file(p, SEQ_LEN, PRED_LEN, STRIDE) for p in train_files)
    val_windows   = sum(count_windows_in_file(p, SEQ_LEN, PRED_LEN, STRIDE) for p in val_files)
    test_windows  = sum(count_windows_in_file(p, SEQ_LEN, PRED_LEN, STRIDE) for p in test_files)

    train_steps = max(1, train_windows // BATCH_SIZE)
    val_steps   = max(1, val_windows // BATCH_SIZE)
    test_steps  = max(1, test_windows // BATCH_SIZE)

    print(f"Fenêtres - Train: {train_windows}, Val: {val_windows}, Test: {test_windows}")
    print(f"Steps - Train: {train_steps}, Val: {val_steps}, Test: {test_steps}")

    train_ds = make_dataset(train_files, training=True)
    val_ds   = make_dataset(val_files, training=False)
    test_ds  = make_dataset(test_files, training=False)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_lstm_forecast.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    print("\nÉvaluation sur test:")
    test_metrics = model.evaluate(test_ds, steps=test_steps, verbose=1)
    print(dict(zip(model.metrics_names, test_metrics)))

    # =========================
    # Exemple de prédiction
    # =========================
    for x_batch, y_batch in test_ds.take(1):
        pred_batch = model.predict(x_batch, verbose=0)
        

        x0 = x_batch[0].numpy()
        y0 = y_batch[0].numpy()
        p0 = pred_batch[0]

        # Affichage du premier canal
        true_signal = y0[:, 0]
        pred_signal = p0[:, 0]

        print("\nMAE sur un exemple:", mean_absolute_error(true_signal, pred_signal))
        print("MSE sur un exemple:", mean_squared_error(true_signal, pred_signal))

        plt.figure(figsize=(10, 4))
        plt.plot(true_signal, label="True")
        plt.plot(pred_signal, label="Predicted")
        plt.title("Prédiction du futur signal - canal 1")
        plt.legend()
        plt.tight_layout()
        plt.show()
    model.save("lstm_das_model.h5")
    print("Modèle sauvegardé")
    # =========================
    # EVALUATION GLOBALE
    # =========================
    all_true = []
    all_pred = []

    for x_batch, y_batch in test_ds.take(50):  # 50 batches
        pred = model.predict(x_batch, verbose=0)

        all_true.append(y_batch.numpy())
        all_pred.append(pred)

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    mae = np.mean(np.abs(all_true - all_pred))
    mse = np.mean((all_true - all_pred)**2)

    print("\n===== RESULTATS FINAUX =====")
    print(f"MAE global: {mae:.4f}")
    print(f"MSE global: {mse:.4f}")

if __name__ == "__main__":
    main()