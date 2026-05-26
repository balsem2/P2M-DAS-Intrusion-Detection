import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

X = np.load("X_das_2d.npy")
y = np.load("y_das_2d.npy")
classes = np.load("classes_das_2d.npy", allow_pickle=True)

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

model = tf.keras.models.load_model("best_cnn_das_2d.keras")

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

with open("accuracy_result.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(text)
print("\nRésultats sauvegardés dans accuracy_result.txt")
