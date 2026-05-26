import tensorflow as tf

model = tf.keras.models.load_model("best_cnn_das_2d.keras")

with open("model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

model.summary()
print("\nRésumé sauvegardé dans model_summary.txt")
