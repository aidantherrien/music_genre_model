import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


"""
Audio Feature Extraction and Dataset Creation for Music Genre Classification
----------------------------------------------------------------------------

This script processes raw and augmented audio datasets to extract a rich set of features from each audio file.
It constructs a labeled dataset ready for training neural networks, visualizes feature distribution via t-SNE,
and saves the processed data in both CSV and NPZ formats.

Key Processing Stages:
----------------------

1. Feature Extraction (via librosa):
   - MFCC (13 coefficients)
   - Chroma STFT (12 bins)
   - Spectral Contrast (7 bands)
   - Zero Crossing Rate (1)
   - Spectral Rolloff (1)
   - Tonnetz (6 dimensions)
   - RMS Energy (1)
   → Combined: 41-dimensional feature vector per time frame

2. Shape Normalization:
   Each song is represented as a fixed-size matrix:
       (time_steps = 128, num_features = 41)
   ┌──────────────────────────┐
   │    Feature Vector (41)   │
   ├──────────────┬───────────┤
   │ Time Step 1  │ ...       │
   │ Time Step 2  │ ...       │
   │     ...      │ ...       │
   │ Time Step 128│ ...       │
   └──────────────┴───────────┘

   → Each song → (128, 41) array → stacked into (num_samples, 128, 41)

3. Label Encoding:
   - Each genre is assigned a unique integer `genre_idx`
   - Labels are one-hot encoded to shape: (num_samples, num_classes = 23)

4. Data Preprocessing:
   - Features are standardized (mean=0, std=1)
   - Dimensionality reduced via PCA (→ 50D), visualized using t-SNE (→ 2D)

5. Saving Outputs:
   - Flattened features + genre labels saved as CSV
   - Final processed dataset (X, y) saved in NPZ for ML use

Inputs:
-------
- Raw audio files in MP3 format, organized by genre
- Two folders: `data/raw` and `data/augmented`

Outputs:
--------
- `features_v8.csv` → flat feature vectors + genre labels
- `features_v8.npz` → structured dataset for training

Usage Notes:
------------
- Modify `time_steps`, `num_features`, or `num_classes` in the config section as needed
- Ideal for CNNs, LSTMs, or hybrid audio models

Dependencies:
-------------
librosa, numpy, pandas, matplotlib, sklearn, tensorflow

"""


# Constants for directories and file paths
RAW_DATA_DIR = 'data/raw'  # Directory for raw audio files
AUGMENTED_DATA_DIR = 'data/augmented'  # Directory for augmented audio files
CSV_SAVE_PATH = r'data\features\features_v8.csv'  # Path to save preprocessed features CSV
NPZ_SAVE_PATH = r'data\processed\features_v8.npz'  # Path to save features and labels as NPZ


# Config
time_steps = 128  # Adjust for desired time steps
num_features = 41  # Number of MFCC features
num_classes = 23  # Adjust for the number of genres
audio_folders = [RAW_DATA_DIR, AUGMENTED_DATA_DIR]  # Folders with raw and augmented data
genres = os.listdir(audio_folders[0])  # Assuming both folders have the same genre structure


# Function to extract multiple features from an audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Extract Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Extract Zero Crossing Rate
    zero_crossings = librosa.feature.zero_crossing_rate(y=y)
    
    # Extract Spectral Roll-Off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # Extract Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    # Extract RMS Energy
    rms_energy = librosa.feature.rms(y=y)
    
    # Stack all features into a single feature vector
    features = np.vstack([mfcc, chroma, contrast, zero_crossings, spectral_rolloff, tonnetz, rms_energy]).T
    return features


# Function to process all audio files in the dataset and create the dataset
def create_dataset(audio_folders, genres):
    all_features = []
    all_labels = []
    total_files = sum(len([f for f in os.listdir(os.path.join(folder, genre)) if f.endswith('.mp3')]) for folder in audio_folders for genre in genres)
    processed_files = 0

    for folder in audio_folders:
        print(f"Processing folder: {folder}")
        
        for genre_idx, genre in enumerate(genres):
            genre_folder = os.path.join(folder, genre)
            audio_files = [f for f in os.listdir(genre_folder) if f.endswith('.mp3')]

            for audio_file in audio_files:
                file_path = os.path.join(genre_folder, audio_file)
                features = extract_features(file_path)
                
                # Pad or truncate features to a consistent shape (time_steps, num_features)
                if features.shape[0] < time_steps:
                    features = np.pad(features, ((0, time_steps - features.shape[0]), (0, 0)), mode='constant')
                else:
                    features = features[:time_steps, :]

                all_features.append(features)
                all_labels.append(genre_idx)

                processed_files += 1
                if processed_files % 100 == 0 or processed_files == total_files:
                    print(f"Processed {processed_files}/{total_files} files")

    # Convert to numpy arrays
    X = np.array(all_features)  # Shape: (num_samples, time_steps, num_features)
    y = np.array(all_labels)

    # One-hot encode labels
    y = to_categorical(y, num_classes=num_classes)

    return X, y


# Create the dataset from raw and augmented data
X, y = create_dataset(audio_folders, genres)

# Check the shape of the dataset
print(f"Feature shape: {X.shape}")
print(f"Label shape: {y.shape}")

# Data Normalization (Standard scaling)
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])  # Flatten the time steps dimension for scaling
X_reshaped = scaler.fit_transform(X_reshaped)
X = X_reshaped.reshape(X.shape[0], time_steps, num_features)

# Apply PCA for feature selection (dimensionality reduction)
pca = PCA(n_components=50)  # Reduce to 50 components, adjust as needed
X_pca = pca.fit_transform(X_reshaped)

# Visualize data with t-SNE for dimensionality reduction
X_tsne = TSNE(n_components=2).fit_transform(X_pca)

# Plot the t-SNE result
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.argmax(y, axis=1), cmap='viridis', alpha=0.6)
plt.colorbar()
plt.title("t-SNE Visualization of the Audio Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

# Save the preprocessed data as CSV (optional)
features_df = pd.DataFrame(X_reshaped)
labels_df = pd.DataFrame(np.argmax(y, axis=1), columns=["genre"])
data = pd.concat([features_df, labels_df], axis=1)
data.to_csv(CSV_SAVE_PATH, index=False)

# Save the data and labels as NumPy arrays in a single .npz file
np.savez_compressed(NPZ_SAVE_PATH, X=X, y=y)

# Check the data
print(f"Preprocessed data saved! X shape: {X.shape}, y shape: {y.shape}")
