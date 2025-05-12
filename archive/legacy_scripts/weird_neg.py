import pandas as pd
import numpy as np

# Genre list and mappings
GENRES = [
    "classic_rock", "alternative_rock", "alternative", "pop_punk", "punk", "soul", "motown", 
    "funk", "disco", "hip-hop", "rap", "folk", "country", "pop_country", "fusion", "jazz", 
    "classical", "blues", "metal", "heavy_metal", "rock", "pop", "electronic"
]
INDEX_TO_GENRE = {i: genre for i, genre in enumerate(GENRES)}

# Path to your CSV
csv_file_path = r'data\features\features_v3.csv'

# Read your CSV
df = pd.read_csv(csv_file_path)

# Check if 'genre' column exists in your CSV
if 'genre' in df.columns:
    # Check for any -1 labels
    invalid_labels = df[df['genre'] == -1]
    
    if not invalid_labels.empty:
        print(f"Found {len(invalid_labels)} rows with invalid label -1.")
        print(invalid_labels[['genre', 'genre_name']])
    
    # Decode the valid labels back to genres
    valid_labels = df[df['genre'] != -1]
    decoded_genres = [INDEX_TO_GENRE[label] if label != -1 else 'Invalid' for label in valid_labels['genre']]
    
    # Add a new column with the genre names
    valid_labels['genre_name'] = decoded_genres
    
    # Combine the valid and invalid data back together
    df = pd.concat([valid_labels, invalid_labels])
    
    # Output the first few rows to verify
    print(df[['genre', 'genre_name']].head())
else:
    print("Error: 'genre' column not found in the CSV.")
