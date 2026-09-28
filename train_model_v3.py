import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

TRAIN_DIR = r"D:\AnemiaDetekt\data\train"
VAL_DIR = r"D:\AnemiaDetekt\data\val"


# ==========================================
# LOAD DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=False
)
print("Classes:", train_ds.class_names)


# ==========================================
# ASPECT-RATIO PRESERVING PREPROCESSING
# ==========================================

def resize_and_pad(images, labels):

    images = tf.cast(images, tf.float32)

    images = tf.image.resize_with_pad(
        images,
        target_height=224,
        target_width=224
    )

    return images, labels


train_ds = train_ds.map(
    resize_and_pad,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    resize_and_pad,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==========================================
# PERFORMANCE
# ==========================================

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ==========================================
# DATA AUGMENTATION
# ==========================================

augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])


# ==========================================
# MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# MODEL
# ==========================================

inputs = keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)


# ==========================================
# PHASE 1
# ==========================================

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


# ==========================================
# CLASS WEIGHTS
# ==========================================

class_weight = {
    0: 1.0,
    1: 296 / 200
}

print("\nClass weights:", class_weight)


# ==========================================
# CALLBACKS
# ==========================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_mobilenetv2_v3.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# ==========================================
# PHASE 1 TRAINING
# ==========================================

print("\n========== V3 PHASE 1 ==========")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weight,
    callbacks=callbacks
)


# ==========================================
# PHASE 2 — FINE TUNING
# ==========================================

print("\n========== V3 PHASE 2: FINE-TUNING ==========")

base_model.trainable = True


# Freeze early layers
for layer in base_model.layers[:-40]:
    layer.trainable = False


# Keep Batch Normalization frozen
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


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=8,
    class_weight=class_weight,
    callbacks=callbacks
)


# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "anemia_mobilenetv2_v3.keras"
)

print("\n================================")
print("✅ V3 TRAINING COMPLETED!")
print("================================")

print(
    "Model saved as: "
    "anemia_mobilenetv2_v3.keras"
)