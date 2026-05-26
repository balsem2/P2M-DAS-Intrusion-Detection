from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Trained_Model" / "best_cnn_das_2d.keras"
SUMMARY_PATH = ROOT / "Results" / "model_summary.txt"

model = tf.keras.models.load_model(MODEL_PATH)

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

model.summary()
print(f"\nRésumé sauvegardé dans {SUMMARY_PATH}")
