
import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
VAL_DIR = r"D:\AnemiaDetekt\data\test"

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", val_ds.class_names)

model = tf.keras.models.load_model(
    "best_mobilenetv2_v2.keras"
)

predictions = model.predict(val_ds, verbose=0).flatten()

actual = np.concatenate([
    labels.numpy().flatten()
    for _, labels in val_ds
]).astype(int)

predicted = (predictions >= 0.5).astype(int)

print("\nActual Anemic:", np.sum(actual == 0))
print("Actual Non-anemic:", np.sum(actual == 1))
print("Predicted Anemic:", np.sum(predicted == 0))
print("Predicted Non-anemic:", np.sum(predicted == 1))

print("\nConfusion matrix:")
print(tf.math.confusion_matrix(
    actual, predicted, num_classes=2
).numpy())