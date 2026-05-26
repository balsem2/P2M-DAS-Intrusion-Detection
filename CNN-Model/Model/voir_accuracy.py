from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_ROOT = ROOT / "Data_Preparation"
TRAINED_ROOT = ROOT / "Trained_Model"
RESULTS_ROOT = ROOT / "Results"

x_path = DATA_ROOT / "X_das_2d.npy"
y_path = DATA_ROOT / "y_das_2d.npy"
if not x_path.exists():
    x_path = PROJECT_ROOT / "X_das_2d.npy"
if not y_path.exists():
    y_path = PROJECT_ROOT / "y_das_2d.npy"

X = np.load(x_path)
y = np.load(y_path)
classes = np.load(TRAINED_ROOT / "classes_das_2d.npy", allow_pickle=True)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_SEED,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=RANDOM_SEED,
)

model = tf.keras.models.load_model(TRAINED_ROOT / "best_cnn_das_2d.keras")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

report = classification_report(
    y_test,
    y_pred,
    target_names=classes,
    digits=4,
    zero_division=0,
)
matrix = confusion_matrix(y_test, y_pred)

result = []
result.append(f"Dataset total : {X.shape}")
result.append(f"Train         : {X_train.shape}")
result.append(f"Validation    : {X_val.shape}")
result.append(f"Test          : {X_test.shape}")
result.append("")
result.append(f"Test loss     : {test_loss:.4f}")
result.append(f"Test accuracy : {test_acc:.4f}")
result.append(f"Accuracy %    : {test_acc * 100:.2f}%")
result.append("")
result.append("Classification report:")
result.append(report)
result.append("Confusion matrix:")
result.append(str(matrix))

text = "\n".join(result)

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_ROOT / "accuracy_result.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(text)
print(f"\nRésultats sauvegardés dans {output_path}")
