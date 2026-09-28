import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# ==================================================
# SETTINGS
# ==================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

TRAIN_DIR = r"D:\AnemiaDetekt\data\augmented_train"
VAL_DIR = r"D:\AnemiaDetekt\data\val"

# ==================================================
# LOAD DATA
# ==================================================

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

print("Classes:", train_ds.class_names)

# Verify dataset size
total_images = 0
for images, labels in train_ds:
    total_images += len(images)

print("\nTRAIN IMAGES =", total_images)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==================================================
# DATA AUGMENTATION
# ==================================================

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# ==================================================
# CLASS WEIGHTS
# ==================================================

anemic_count = 1147
non_anemic_count = 790

class_weight = {
    0: 1.0,
    1: anemic_count / non_anemic_count
}

print("\nClass weights:", class_weight)

# ==================================================
# MODEL
# ==================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)

model.summary()

# ==================================================
# PHASE 1
# ==================================================

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_efficientnet_aug.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

print("\n========== PHASE 1 ==========")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weight,
    callbacks=callbacks
)

# ==================================================
# PHASE 2 - FINE TUNING
# ==================================================

print("\n========== PHASE 2 ==========")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=8,
    class_weight=class_weight,
    callbacks=callbacks
)

# ==================================================
# SAVE MODEL
# ==================================================

model.save("anemia_efficientnet_augmented.keras")

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")
print("Saved: anemia_efficientnet_augmented.keras")