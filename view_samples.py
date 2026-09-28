import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = r"C:\Users\New\Downloads\CP-AnemiC dataset"

folders = {
    "Anemic": os.path.join(DATASET_PATH, "Anemic"),
    "Non-anemic": os.path.join(DATASET_PATH, "Non-anemic")
}

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(42)

fig, axes = plt.subplots(4, 5, figsize=(15, 10))

for row, (label, folder) in enumerate(folders.items()):

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]

    samples = random.sample(files, 10)

    for i, filename in enumerate(samples):

        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)

        ax = axes[row * 2 + i // 5, i % 5]

        ax.imshow(img)
        ax.set_title(f"{label}\n{filename}")
        ax.axis("off")

plt.tight_layout()
plt.show()