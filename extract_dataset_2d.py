from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder




DATASET_ROOT = Path(r"D:\DAS-dataset\DAS-dataset\data")
H5_KEY = "Acquisition/Raw[0]/RawData"

CLASS_NAMES = [
    "car",
    "construction",
    "fence",
    "longboard",
    "manipulation",
    "openclose",
    "regular",
    "running",
    "walk",
]

WINDOW_SIZE = 8192
SHIFT = 2048
OVERSAMPLING = 3
FFT_KEEP = 1024


PATCH_CHANNELS = 17
HALF_PATCH = PATCH_CHANNELS // 2

MAX_SAMPLES_PER_LABEL = 1500
USE_AUGMENTATION = False
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)





def load_bitmap(file_path: Path) -> np.ndarray:
    return np.load(file_path)


def hann_fft_features(signal_1d: np.ndarray, augment: bool = False) -> np.ndarray:
    x = signal_1d.astype(np.float32)
    x = x - np.mean(x)

    w = np.hanning(len(x)).astype(np.float32)
    xw = x * w

    nfft = len(x) * OVERSAMPLING
    spec = np.fft.rfft(xw, n=nfft)
    mag = np.abs(spec)

    mag = mag[:FFT_KEEP]
    mag = np.log1p(mag)
    mag = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)

    if augment:
        noise = np.random.normal(0.0, 0.01, size=mag.shape).astype(np.float32)
        mag = mag + noise

    return mag.astype(np.float32)


def extract_patch_2d(
    dset,
    center_channel: int,
    window_idx: int,
    num_channels: int,
) -> np.ndarray:
    start = window_idx * SHIFT
    end = start + WINDOW_SIZE

    patch = np.zeros((PATCH_CHANNELS, FFT_KEEP), dtype=np.float32)

    row = 0
    for ch in range(center_channel - HALF_PATCH, center_channel + HALF_PATCH + 1):
        if 0 <= ch < num_channels:
            signal = np.array(dset[start:end, ch], dtype=np.float32)
            if len(signal) == WINDOW_SIZE:
                feat = hann_fft_features(signal, augment=False)
                patch[row] = feat
        row += 1

    return patch





def build_dataset_from_h5_and_npy(dataset_root: Path):
    X_list = []
    y_list = []

    for label in CLASS_NAMES:
        label_dir = dataset_root / label
        if not label_dir.exists():
            print(f"[WARN] Dossier absent: {label_dir}")
            continue

        collected = 0

        for h5_path in sorted(label_dir.glob("*.h5")):
            npy_path = h5_path.with_suffix(".npy")
            if not npy_path.exists():
                print(f"[WARN] Bitmap absent pour {h5_path.name}")
                continue

            print(f"[INFO] Lecture: {h5_path.name}")

            bitmap = load_bitmap(npy_path)
            if bitmap.ndim != 2:
                print(f"[WARN] Bitmap inattendu: {bitmap.shape}")
                continue

            with h5py.File(h5_path, "r") as f:
                dset = f[H5_KEY]
                num_time, num_channels = dset.shape

                n_windows = (num_time - WINDOW_SIZE) // SHIFT + 1
                if n_windows <= 0:
                    print(f"[WARN] Trop petit: {h5_path.name}")
                    continue

                bmp = bitmap
                if bmp.shape[0] != n_windows and bmp.shape[1] == n_windows:
                    bmp = bmp.T

                min_windows = min(n_windows, bmp.shape[0])
                min_channels = min(num_channels, bmp.shape[1])

                if min_windows <= 0 or min_channels <= 0:
                    print(f"[WARN] Dimensions invalides pour {h5_path.name}: raw={dset.shape}, bmp={bmp.shape}")
                    continue

                if bmp.shape[0] != n_windows:
                    print(
                        f"[WARN] Fenêtres mismatch pour {h5_path.name}: "
                        f"raw={n_windows}, bmp={bmp.shape} -> min_windows={min_windows}"
                    )



                positive_positions = np.argwhere(bmp[:min_windows, :min_channels] > 0)

                if len(positive_positions) == 0:
                    continue


                np.random.shuffle(positive_positions)

                for w_idx, ch in positive_positions:
                    patch = extract_patch_2d(
                        dset=dset,
                        center_channel=int(ch),
                        window_idx=int(w_idx),
                        num_channels=num_channels,
                    )
                    X_list.append(patch)
                    y_list.append(label)
                    collected += 1

                    if USE_AUGMENTATION and collected < MAX_SAMPLES_PER_LABEL:
                        patch_aug = patch + np.random.normal(0, 0.01, patch.shape).astype(np.float32)
                        X_list.append(patch_aug)
                        y_list.append(label)
                        collected += 1

                    if collected >= MAX_SAMPLES_PER_LABEL:
                        break

            print(f"[OK] {label}: {collected} échantillons accumulés")

    if not X_list:
        raise ValueError("Aucun échantillon généré.")

    X = np.stack(X_list).astype(np.float32)
    X = X[..., np.newaxis]

    le = LabelEncoder()
    y = le.fit_transform(np.array(y_list))

    print("\n[INFO] Extraction 2D terminée")
    print("[INFO] X shape:", X.shape)
    print("[INFO] y shape:", y.shape)
    print("[INFO] Classes:", list(le.classes_))

    np.save("X_das_2d.npy", X)
    np.save("y_das_2d.npy", y)
    np.save("classes_das_2d.npy", le.classes_)

    print("[INFO] Fichiers sauvegardés :")
    print(" - X_das_2d.npy")
    print(" - y_das_2d.npy")
    print(" - classes_das_2d.npy")


if __name__ == "__main__":
    build_dataset_from_h5_and_npy(DATASET_ROOT)
