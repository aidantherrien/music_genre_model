import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Genres list to match with the dataset
GENRES = [
    "classic_rock", "alternative_rock", "alternative", "pop_punk", "punk", 
    "soul", "motown", "funk", "disco", "hip-hop", "rap", "folk", 
    "country", "pop_country", "fusion", "jazz", "classical", "blues", 
    "metal", "heavy_metal", "rock", "pop", "electronic"
]

# Mapping from genre to class index
GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}

# Mapping from class index back to genre
INDEX_TO_GENRE = {i: genre for i, genre in enumerate(GENRES)}

# --- Load and shuffle dataset ---
df = pd.read_csv("data/features/features_v3.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Prepare input and labels ---
X = df.drop("genre", axis=1).values
y_genres = df["genre"].values

# Map genre names to class indices
y_indices = [GENRE_TO_INDEX.get(genre, -1) for genre in y_genres]

# Check unique values in y to identify any mismatched or unseen labels
print("Unique labels in y:", np.unique(y_indices))

# Make sure that all genres in y are accounted for in GENRES
missing_labels = set(np.unique(y_indices)) - set(range(len(GENRES)))
if missing_labels:
    print("Missing labels not found in GENRES:", missing_labels)

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y_indices, test_size=0.2, random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=int)



# --- Compute class weights to handle imbalance ---
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_indices), y=y_indices)
class_weight_dict = dict(enumerate(class_weights))

# --- Load existing model ---
model_path = os.path.join("models", "genre_classifier_model_v6.1.keras")
model = load_model(model_path)

# --- Set up callbacks ---
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
]

# --- Train model ---
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    shuffle=True,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# --- Save updated model ---
model.save(model_path)

# --- Evaluate performance ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
