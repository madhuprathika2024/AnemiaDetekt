import os
import pandas as pd

DATASET_PATH = r"C:\Users\New\Downloads\CP-AnemiC dataset"

ANEMIC_FOLDER = os.path.join(DATASET_PATH, "Anemic")
NON_ANEMIC_FOLDER = os.path.join(DATASET_PATH, "Non-anemic")

EXCEL_FILE = os.path.join(
    DATASET_PATH,
    "Anemia_Data_Collection_Sheet.xlsx"
)

# Read Excel
df = pd.read_excel(EXCEL_FILE)

# Clean column names
df.columns = df.columns.str.strip()

# Create image -> folder mapping
image_to_folder = {}

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for filename in os.listdir(ANEMIC_FOLDER):
    if filename.lower().endswith(extensions):
        image_id = os.path.splitext(filename)[0].strip()
        image_to_folder[image_id] = "Anemic"

for filename in os.listdir(NON_ANEMIC_FOLDER):
    if filename.lower().endswith(extensions):
        image_id = os.path.splitext(filename)[0].strip()
        image_to_folder[image_id] = "Non-anemic"

# Check every Excel row
mismatches = []
missing = []

for _, row in df.iterrows():

    image_id = str(row["IMAGE_ID"]).strip()
    excel_label = str(row["REMARK"]).strip()

    folder_label = image_to_folder.get(image_id)

    if folder_label is None:
        missing.append(image_id)

    elif folder_label.lower() != excel_label.lower():
        mismatches.append(
            (image_id, excel_label, folder_label)
        )

print("\n========== LABEL VERIFICATION ==========")

print(f"Total Excel records : {len(df)}")
print(f"Images found        : {len(image_to_folder)}")
print(f"Missing images      : {len(missing)}")
print(f"Label mismatches    : {len(mismatches)}")

if missing:
    print("\nMissing images:")
    for image in missing:
        print(image)

if mismatches:
    print("\nLabel mismatches:")
    for item in mismatches:
        print(item)

if not missing and not mismatches:
    print("\n✅ ALL 710 RECORDS MATCH CORRECTLY!")