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
from multiprocessing import Pool, cpu_count


"""
Audio Feature Extraction and Dataset Creation for Music Genre Classification
----------------------------------------------------------------------------

This script processes raw and augmented audio datasets to extract a rich set of features from each audio file.
It constructs a labeled dataset ready for training neural networks, visualizes feature distribution via t-SNE,
and saves the processed data in both CSV and NPZ formats.

Note: This script is functionally the same as encode_vectors.py in terms of how it encodes vectors, however
      here we are able to use multiple cores/workers for faster processing.

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


# Constants
RAW_DATA_DIR = 'data/raw'
AUGMENTED_DATA_DIR = 'data/augmented'
CSV_SAVE_PATH = r'data\features\features_v8.csv'
NPZ_SAVE_PATH = r'data\processed\features_v8.npz'
NUM_CORES = 6  # Set the number of cores you want to use


# Config
time_steps = 128  # Number of time steps (frames)
num_features = 41  # Number of features per time step (MFCC, Chroma, etc.)
num_classes = 23  # Number of genre classes
audio_folders = [RAW_DATA_DIR, AUGMENTED_DATA_DIR]
genres = os.listdir(audio_folders[0])


# Function to extract and process features 
def extract_and_process(args):
    file_path, genre_idx = args
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Skip silent or very short clips
        if y is None or len(y) < sr * 3 or np.max(np.abs(y)) < 0.01:
            print(f"Skipping short or silent file: {file_path}")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossings = librosa.feature.zero_crossing_rate(y=y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)

        # Stack all features into one 2D array (time x features)
        features = np.vstack([
            mfcc, chroma, contrast,
            zero_crossings, spectral_rolloff, tonnetz, rms_energy
        ]).T

        # If there are NaNs or Infs, return None
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"Invalid values in features for {file_path}")
            return None

        # Ensure the feature array matches the desired time_steps
        if features.shape[0] < time_steps:
            features = np.pad(features, ((0, time_steps - features.shape[0]), (0, 0)), mode='constant')
        else:
            features = features[:time_steps, :]

        return (features, genre_idx)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Function to collect file paths 
def collect_filepaths(audio_folders, genres):
    tasks = []
    for genre_idx, genre in enumerate(genres):
        for folder in audio_folders:
            genre_path = os.path.join(folder, genre)
            if not os.path.isdir(genre_path):
                continue
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.mp3'):
                    full_path = os.path.join(genre_path, file_name)
                    tasks.append((full_path, genre_idx))
    return tasks


if __name__ == '__main__':
    print("Collecting audio file paths...")
    task_list = collect_filepaths(audio_folders, genres)
    print(f"Total files to process: {len(task_list)}")

    print(f"Using {NUM_CORES} CPU cores for parallel processing...")
    with Pool(NUM_CORES) as pool:
        results = pool.map(extract_and_process, task_list)

    # Filter out any None results due to errors or short files
    results = [res for res in results if res is not None]
    features, labels = zip(*results)

    # Convert features and labels to arrays
    X = np.array(features)  # Shape: (num_songs, time_steps, num_features)
    y = np.array(labels)    # Shape: (num_songs,)

    # One-hot encode labels
    y = to_categorical(y, num_classes=num_classes)

    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    # Step 1: Normalize the features (per feature, across time steps)
    # Flatten time and feature dimensions for scaling
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to (num_samples, time_steps, num_features)
    X = X_scaled.reshape(X.shape[0], time_steps, num_features)

    # Step 2: Optionally, apply PCA for dimensionality reduction (if desired)
    pca = PCA(n_components=min(X.shape[2], 20))  # Limit to 20 components, or based on features
    X_pca = pca.fit_transform(X_reshaped)

    # Reshape back for CNN-LSTM input (3D: num_samples, time_steps, reduced_num_features)
    X_pca = X_pca.reshape(X.shape[0], time_steps, -1)

    """    
    # Step 3: Visualize with t-SNE (optional)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca.reshape(X_pca.shape[0], -1))

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.argmax(y, axis=1), cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title("t-SNE Visualization of the Audio Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
    """

    # Save the features and labels in CSV and NPZ format
    features_df = pd.DataFrame(X_reshaped)
    labels_df = pd.DataFrame(np.argmax(y, axis=1), columns=["genre"])
    pd.concat([features_df, labels_df], axis=1).to_csv(CSV_SAVE_PATH, index=False)

    np.savez_compressed(NPZ_SAVE_PATH, X=X, y=y)

    print(f"Preprocessed data saved! X shape: {X.shape}, y shape: {y.shape}")

