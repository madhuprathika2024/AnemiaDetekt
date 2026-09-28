import os
from PIL import Image

DATASET_PATH = r"C:\Users\New\Downloads\CP-AnemiC dataset"

folders = {
    "Anemic": os.path.join(DATASET_PATH, "Anemic"),
    "Non-anemic": os.path.join(DATASET_PATH, "Non-anemic")
}

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for label, folder in folders.items():

    print(f"\n========== {label.upper()} ==========")

    total = 0
    valid = 0
    corrupted = 0
    sizes = {}

    for filename in os.listdir(folder):

        if not filename.lower().endswith(extensions):
            continue

        total += 1
        filepath = os.path.join(folder, filename)

        try:
            with Image.open(filepath) as img:

                # Verify image
                img.verify()

            # Open again to get dimensions
            with Image.open(filepath) as img:
                width, height = img.size

            valid += 1
            sizes[(width, height)] = sizes.get((width, height), 0) + 1

        except Exception:
            corrupted += 1
            print("Corrupted:", filename)

    print("Total images     :", total)
    print("Valid images     :", valid)
    print("Corrupted images :", corrupted)

    print("\nImage sizes:")
    for size, count in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{size}: {count} images")