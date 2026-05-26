import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = r"D:/P2M-DAS-Intrusion-Detection/data/raw/reconstruction1_2023-06-06T150714+0100.h5"

with h5py.File(file_path, "r") as f:
    ds = f["/Acquisition/Raw[0]/RawData"]
    print("Shape totale:", ds.shape)

    # Extrait (rapide) — ajuste si tu veux
    data = ds[:5000, :300]

# Courbe 1: amplitude moyenne absolue dans le temps
mean_signal = np.mean(np.abs(data), axis=1)
plt.figure(figsize=(10, 4))
plt.plot(mean_signal)
plt.xlabel("Temps (échantillons)")
plt.ylabel("Amplitude moyenne")
plt.title("Amplitude moyenne du signal DAS")
plt.grid(True)
plt.savefig("figures/mean_signal_time.png", dpi=300, bbox_inches="tight")
plt.close()

# Courbe 2: signal temporel à une position donnée
pos = 150
signal_time = data[:, pos]
plt.figure(figsize=(10, 4))
plt.plot(signal_time)
plt.xlabel("Temps (échantillons)")
plt.ylabel("Amplitude DAS")
plt.title(f"Signal temporel au capteur virtuel {pos}")
plt.grid(True)
plt.savefig("figures/local_time_signal.png", dpi=300, bbox_inches="tight")
plt.close()

# Courbe 3: coupe spatiale à l'instant le plus actif
t_max = int(np.argmax(mean_signal))
signal_space = data[t_max, :]
plt.figure(figsize=(10, 4))
plt.plot(signal_space)
plt.xlabel("Position sur la fibre")
plt.ylabel("Amplitude DAS")
plt.title(f"Coupe spatiale à l’instant t={t_max}")
plt.grid(True)
plt.savefig("figures/spatial_slice.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figures saved in: D:/P2M-DAS-Intrusion-Detection/figures/")
