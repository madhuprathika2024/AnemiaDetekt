import os
import random
import shutil

DATASET_PATH = r"C:\Users\New\Downloads\CP-AnemiC dataset"
OUTPUT_PATH = r"D:\AnemiaDetekt\data"

random.seed(42)

classes = ["Anemic", "Non-anemic"]
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Create output folders
for split in ["train", "val", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(OUTPUT_PATH, split, class_name),
            exist_ok=True
        )

for class_name in classes:

    source_folder = os.path.join(DATASET_PATH, class_name)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(extensions)
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    print(f"\n{class_name}")
    print(f"Total : {total}")

    for split, files in splits.items():

        print(f"{split}: {len(files)}")

        destination = os.path.join(
            OUTPUT_PATH,
            split,
            class_name
        )

        for filename in files:

            source = os.path.join(source_folder, filename)
            target = os.path.join(destination, filename)

            shutil.copy2(source, target)

print("\n✅ Dataset split completed!")
print(f"Created at: {OUTPUT_PATH}")