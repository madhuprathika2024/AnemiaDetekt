import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# =============================
# CONFIG
# =============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20

TRAIN_DIR = r"D:\AnemiaDetekt\data\train"
VAL_DIR = r"D:\AnemiaDetekt\data\val"

# =============================
# LOAD DATASETS
# =============================

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

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =============================
# DATA AUGMENTATION
# =============================

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# =============================
# EFFICIENTNETB0
# =============================

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
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

# =============================
# COMPILE
# =============================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

# =============================
# CLASS WEIGHTS
# =============================

class_weight = {
    0: 1.0,
    1: 296 / 200
}

print("\nClass weights:", class_weight)

# =============================
# CALLBACKS
# =============================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_efficientnetb0.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# =============================
# MODEL SUMMARY
# =============================

model.summary()

# =============================
# PHASE 1
# =============================

print("\n========== EFFICIENTNET PHASE 1 ==========")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weight,
    callbacks=callbacks
)

# =============================
# PHASE 2
# =============================

print("\n========== EFFICIENTNET PHASE 2 ==========")

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
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

# =============================
# SAVE MODEL
# =============================

model.save("anemia_efficientnetb0.keras")

print("\n================================")
print("✅ EFFICIENTNET TRAINING COMPLETED!")
print("================================")
print("Model saved as: anemia_efficientnetb0.keras")