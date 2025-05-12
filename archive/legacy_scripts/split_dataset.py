import os
import csv
import random

# === CONFIGURABLE CONSTANTS ===
INPUT_CSV_PATH = r"data\features\features_v4.csv"
OUTPUT_DIR = r"data\features\v7.0_split"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")

# === ENSURE OUTPUT DIRECTORY EXISTS ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === READ DATA ===
with open(INPUT_CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = reader[1:]

# === SHUFFLE DATA ===
random.shuffle(data)

# === SPLIT DATA ===
total = len(data)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# === WRITE CSV FILES ===
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

write_csv(TRAIN_CSV, train_data)
write_csv(VAL_CSV, val_data)
write_csv(TEST_CSV, test_data)

print(f"Split complete:")
print(f"  Training set:   {len(train_data)} rows → {TRAIN_CSV}")
print(f"  Validation set: {len(val_data)} rows → {VAL_CSV}")
print(f"  Test set:       {len(test_data)} rows → {TEST_CSV}")
