import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


MODEL_PATH = r"D:\AnemiaDetekt\anemia_efficientnet_augmented.keras"
TEST_DIR = r"D:\AnemiaDetekt\data\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:", class_names)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Get predictions
y_true = []
y_prob = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_prob.extend(predictions.flatten())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Convert probabilities to classes
y_pred = (y_prob >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n========== BASELINE TEST RESULTS ==========")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"F1 Score      : {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_true, y_pred)

print(cm)

print("\nRows = Actual")
print("Columns = Predicted")
print(f"\n             {class_names}")