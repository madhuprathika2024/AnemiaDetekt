import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

TRAIN_DIR = r"D:\AnemiaDetekt\data\train"
VAL_DIR = r"D:\AnemiaDetekt\data\val"

# -----------------------------
# Load datasets
# -----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# print("Classes:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# Data augmentation
# -----------------------------

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# -----------------------------
# MobileNetV2
# -----------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze everything
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

# -----------------------------
# Phase 1: train classifier
# -----------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

# Class weights
# Anemic = 296
# Non-anemic = 200

class_weight = {
    0: 1.0,
    1: 296 / 200
}

print("\nClass weights:", class_weight)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best_mobilenetv2_v2.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

print("\n========== PHASE 1 ==========")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weight,
    callbacks=callbacks
)

# -----------------------------
# Phase 2: fine-tuning
# -----------------------------

print("\n========== PHASE 2: FINE-TUNING ==========")

base_model.trainable = True

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Keep BatchNorm layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=8,
    class_weight=class_weight,
    callbacks=callbacks
)
import numpy as np

print("\n========== VALIDATION CHECK ==========")

print("Classes:", train_ds.class_names)

all_predictions = []
all_labels = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0).flatten()
    all_predictions.extend(predictions)
    all_labels.extend(labels.numpy().flatten())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels).astype(int)

predicted_classes = (all_predictions >= 0.5).astype(int)

print("Actual Anemic:", np.sum(all_labels == 0))
print("Actual Non-anemic:", np.sum(all_labels == 1))
print("Predicted Anemic:", np.sum(predicted_classes == 0))
print("Predicted Non-anemic:", np.sum(predicted_classes == 1))

print("\nFirst 10 probabilities:")
print(all_predictions[:10])
# -----------------------------
# Save model
# -----------------------------

model.save("anemia_mobilenetv2_v2.keras")

print("\n✅ Experiment 2 completed!")
print("Model saved as: anemia_mobilenetv2_v2.keras")