import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Original training data
SOURCE_DIR = r"D:\AnemiaDetekt\data\train"

# New augmented dataset
TARGET_DIR = r"D:\AnemiaDetekt\data\augmented_train"

# Create folders
classes = ["Anemic", "Non-anemic"]

for cls in classes:
    os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Number of augmented copies per image
AUG_PER_IMAGE = 3

for cls in classes:

    src_folder = os.path.join(SOURCE_DIR, cls)
    dst_folder = os.path.join(TARGET_DIR, cls)

    images = os.listdir(src_folder)

    print(f"\nProcessing {cls}...")

    for img_name in images:

        img_path = os.path.join(src_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(dst_folder, img_name))

            x = img.resize((224,224))
            x = x.copy()

            import numpy as np
            x = np.array(x)
            x = x.reshape((1,) + x.shape)

            i = 0

            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=dst_folder,
                save_prefix="aug",
                save_format="jpg"
            ):
                i += 1
                if i >= AUG_PER_IMAGE:
                    break

        except Exception as e:
            print("Error:", img_name, e)

print("\nDataset augmentation completed!")