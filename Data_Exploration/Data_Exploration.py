import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ====================== CONFIGURATION ======================
data_dir = "*"   # ← À ADAPTER
label = "*"
filename = "*"       # ← Ton fichier

h5_path = f"{data_dir}/{label}/{filename}"
npy_path = h5_path.replace('.h5', '.npy')

# --- Nouveau : fichier texte pour logs ---
txt_path = h5_path.replace('.h5', '.txt')
sys.stdout = open(txt_path, "w", encoding="utf-8")

print(f"Chargement : {h5_path}")
# ===========================================================

# 1. Charger les données
with h5py.File(h5_path, 'r') as f:
    raw_data = f['/Acquisition/Raw[0]/RawData'][:]      # (n_samples, n_channels)
    print(f"Signal brut shape : {raw_data.shape}")

bitmap = np.load(npy_path)                              # (n_windows, n_channels)
print(f"Bitmap shape       : {bitmap.shape}")
print(f"Pourcentage d'événements : {bitmap.mean()*100:.3f}%")

# Paramètres de fenêtrage
window_size = 8192
shift = 2048

# 2. Visualisation améliorée
fig, axs = plt.subplots(3, 1, figsize=(15, 11), sharex=False)

mean_abs = np.mean(np.abs(raw_data), axis=1)
global_power = 20 * np.log10(mean_abs + 1e-8)
axs[0].plot(global_power, color='blue')
axs[0].set_title("Signal global - Puissance moyenne sur tous les canaux")
axs[0].set_ylabel("Amplitude (dB approx)")
axs[0].grid(True, alpha=0.3)

axs[1].imshow(bitmap.T, aspect='auto', cmap='gray_r', origin='lower', extent=[0, bitmap.shape[0]*shift, 0, bitmap.shape[1]])
axs[1].set_title("Bitmap de labels (.npy) - Blanc = Fenêtre contenant l'événement")
axs[1].set_ylabel("Canaux (Distance ≈ mètres)")
axs[1].set_xlabel("Numéro de fenêtre (chaque fenêtre = 2048 échantillons)")

active_channels = np.where(bitmap.sum(axis=0) > 0)[0]
if len(active_channels) > 0:
    best_channel = active_channels[len(active_channels)//2]
else:
    best_channel = raw_data.shape[1] // 2

print(f"Canal choisi pour visualisation détaillée : {best_channel}")

axs[2].plot(raw_data[:, best_channel], color='darkblue', linewidth=0.8)
axs[2].set_title(f"Signal brut - Canal {best_channel}")
axs[2].set_xlabel("Échantillons temporels")
axs[2].set_ylabel("Amplitude (valeur brute)")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

windows_with_event = np.where(bitmap.any(axis=1))[0]
if len(windows_with_event) > 0:
    print(f"\nFenêtres contenant l'événement : {windows_with_event}")
    print(f"Première fenêtre active : {windows_with_event[0]}")
    print(f"Dernière fenêtre active  : {windows_with_event[-1]}")
    active_channels = np.where(bitmap.any(axis=0))[0]
    
    print(f"\nNombre de canaux où l'événement a été détecté : {len(active_channels)}")
    print(f"Canaux actifs (distances approx) : {active_channels}")
    
    if len(active_channels) > 0:
        print(f"Plage des canaux actifs : de {active_channels.min()} à {active_channels.max()}")
        print(f"Canaux les plus actifs : {active_channels[:10].tolist()} ...")

    channel_activity = np.sum(bitmap, axis=0)
    most_active_channels = np.argsort(channel_activity)[::-1]
    
    print(f"\nTop 5 canaux les plus actifs :")
    for i in range(min(5, len(most_active_channels))):
        ch = most_active_channels[i]
        print(f"  Canal {ch:4d} → {channel_activity[ch]} fenêtres actives")
else:
    print("\nAucune fenêtre ne contient l'événement.")

# --- Fermer le fichier log ---
sys.stdout.close()
