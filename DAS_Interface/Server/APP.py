import time
import h5py
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DAS Pipeline Monitoring Live Backend")

# --- CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CHARGEMENT DES MODÈLES ---
try:
    print("⏳ Chargement des modèles Keras...")
    cnn_model = tf.keras.models.load_model(r"C:\Users\ayadi\Downloads\P2M-DAS-Intrusion-Detection\DAS_Interface\Server\best_cnn_das_2d.keras")
    lstm_model = tf.keras.models.load_model(r"C:\Users\ayadi\Downloads\P2M-DAS-Intrusion-Detection\DAS_Interface\Server\best_lstm_forecast.keras")
    print("✅ Modèles chargés avec succès !")
except Exception as e:
    print(f"⚠️ Alerte Modèles: Exécution en mode simulation si absents ({e})")

H5_FILE_PATH = r"D:\DAS-dataset\data\car\auto4_2023-06-06T145055+0100.json"

# --- CONFIGURATION DU TEMPS RÉEL (SIMULATION DE FLUX CORRÉLÉE) ---
# Nombre total de pas de temps (lignes) dans votre matrice H5.
# À ajuster selon votre fichier (ex: 10000). Le modulo évite l'index hors-limite.
TOTAL_TIMESTEPS = 2000 
TIME_WINDOW_CNN = 50   # Nombre de snapshots temporels envoyés au CNN (ex: matrice 50x10)
HISTORY_LEN_LSTM = 20  # Nombre de pas passés pour Recharts et entrée LSTM

def get_current_time_index():
    """ Calcule un index temporel qui avance chaque seconde et boucle à l'infini """
    current_second = int(time.time())
    return current_second % (TOTAL_TIMESTEPS - TIME_WINDOW_CNN - 1)


# =========================================================================
# ROUTE 1 : FLUX LIVE POUR LES 166 POINTS (CLASSIFICATION CNN EN BOUCLE)
# =========================================================================
# --- LISTE DES CLASSES DE VOTRE CNN ---
CNN_CLASSES = [
    "car", "construction", "fence", "longboard", 
    "manipulation", "openclose", "regular", "running", "walk"
]

@app.get("/api/segments")
def get_all_segments():
    t_idx = get_current_time_index()
    segments_status = []
    
    # ... (Gardez ici votre bloc try/except pour lire live_window depuis le .h5) ...

    for i in range(166):
        # --- VRAI CODE CNN (À DÉCOMMENTER QUAND VOUS ÊTES PRÊT) ---
        # start_channel = i * 10
        # end_channel = (i + 1) * 10
        # segment_matrix = live_window[:, start_channel:end_channel]
        # cnn_input = np.expand_dims(np.expand_dims(segment_matrix, axis=0), axis=-1)
        # preds = cnn_model.predict(cnn_input, verbose=0)
        # class_id = np.argmax(preds[0])
        # predicted_class = CNN_CLASSES[class_id]
        
        # --- SIMULATION TEMPS RÉEL (Pour voir l'interface bouger) ---
        predicted_class = "regular"  # Par défaut, tout est normal (Vert)
        
        # On simule des apparitions d'événements à certains endroits et moments
        if i == 35 and (t_idx % 60 < 25):  
            predicted_class = "construction" # Sera Rouge
        elif i == 112 and (t_idx % 40 < 15): 
            predicted_class = "running"      # Sera Orange
        elif i == 80 and (t_idx % 50 < 10):
            predicted_class = "fence"        # Sera Rouge
        elif i == 10 and (t_idx % 30 < 5):
            predicted_class = "car"          # Sera Orange

        # --- LOGIQUE DE COULEUR DEMANDÉE ---
        if predicted_class in ["construction", "fence"]:
            status = "red"
        elif predicted_class == "regular":
            status = "green"
        else:
            # Pour car, longboard, manipulation, openclose, running, walk (Événements anonymes/autres)
            status = "orange" 
            
        segments_status.append({
            "id": i + 1, 
            "status": status,
            "event_name": predicted_class  # On envoie le nom exact au frontend
        })
        
    return segments_status


# =========================================================================
# ROUTE 2 : DÉTAILS D'UN SEGMENT (GRAPHES TEMPS RÉEL + PRÉDICTIONS LSTM)
# =========================================================================
@app.get("/api/segment/{segment_id}")
def get_segment_details(segment_id: int):
    if segment_id < 1 or segment_id > 166:
        raise HTTPException(status_code=404, detail="Segment invalide")

    t_idx = get_current_time_index()
    charts_data = {}
    
    try:
        with h5py.File(H5_FILE_PATH, "r") as f:
            signal_dataset = f["signal"]
            # Récupération de la plage de canaux (10 mètres) pour ce segment
            start_ch = (segment_id - 1) * 10
            end_ch = segment_id * 10
            
            # Extraction de l'historique récent (ex: 20 derniers points pour le graphique)
            # Shape : (20, 10)
            segment_history = signal_dataset[t_idx : t_idx + HISTORY_LEN_LSTM, start_ch:end_ch]
    except Exception as e:
        segment_history = np.random.rand(HISTORY_LEN_LSTM, 10) * 0.4 + 0.2

    # Génération des séries temporelles pour chacun des 10 mètres
    for meter_offset in range(10):
        # Signal historique réel pour ce mètre précis (longueur 20)
        actual_signal = segment_history[:, meter_offset]
        
        # --- BLOC DE PRÉDICTION LSTM ---
        # 1. Formater pour votre entrée LSTM (ex: [1, 20, 1])
        # lstm_input = np.reshape(actual_signal, (1, HISTORY_LEN_LSTM, 1))
        # 2. Prédire le futur proche
        # lstm_forecast = lstm_model.predict(lstm_input, verbose=0)[0] # Array de taille 20
        
        # Simulation d'une prédiction corrélée (signal décalé avec léger bruit)
        lstm_forecast = actual_signal + (np.random.randn(HISTORY_LEN_LSTM) * 0.04)

        # Structuration de l'historique glissant pour Recharts
        meter_chart = []
        for t in range(HISTORY_LEN_LSTM):
            meter_chart.append({
                "time": t_idx + t, # Index temporel global incrémental
                "reel": float(actual_signal[t]),
                "predit": float(lstm_forecast[t])
            })
            
        charts_data[f"meter_{meter_offset + 1}"] = meter_chart

    return {
        "segment_id": segment_id,
        "name": f"Segment {segment_id} - Mètres {(segment_id-1)*10} à {segment_id*10}",
        "charts": charts_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)