import pandas as pd

# === Config ===
CSV_PATH = r"C:\Users\aidan\Documents\VSCode Projects\music_genre_model\data\metadata\dataset_meta_v3.1.csv"        # Replace with your CSV path
OUTPUT_PATH = r"C:\Users\aidan\Documents\VSCode Projects\music_genre_model\data\metadata\dataset_meta_v3.3.csv"  # Where to save the modified CSV

# === Load, Modify, Save ===
df = pd.read_csv(CSV_PATH)

# Replace 'heavy metal' with 'heavy_metal' in 'genre' column
df['genre'] = df['genre'].replace('heavy metal', 'heavy_metal')

# Save updated CSV
df.to_csv(OUTPUT_PATH, index=False)

print(f"Updated CSV saved to: {OUTPUT_PATH}")
