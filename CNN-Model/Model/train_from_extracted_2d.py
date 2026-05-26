from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models




RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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

print("[INFO] X shape:", X.shape)
print("[INFO] y shape:", y.shape)
print("[INFO] Classes:", classes)




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

print(f"[INFO] Train: {X_train.shape}")
print(f"[INFO] Val  : {X_val.shape}")
print(f"[INFO] Test : {X_test.shape}")




class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights_values)}
print("[INFO] Class weights:", class_weights)




def build_cnn_2d(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 7), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 5), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_cnn_2d(X_train.shape[1:], len(classes))
model.summary()




cbs = [
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
    ),
    callbacks.ModelCheckpoint(
        TRAINED_ROOT / "best_cnn_das_2d.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
]




history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=cbs,
    shuffle=True,
    verbose=1,
)




test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[RESULT] Test loss     : {test_loss:.4f}")
print(f"[RESULT] Test accuracy : {test_acc:.4f}")

y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\n=== Classification Report ===")
report = classification_report(y_test, y_pred, target_names=classes, digits=4)
print(report)

print("\n=== Confusion Matrix ===")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
with open(RESULTS_ROOT / "accuracy_result.txt", "w", encoding="utf-8") as f:
    f.write(f"Test loss     : {test_loss:.4f}\n")
    f.write(f"Test accuracy : {test_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))
