import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import librosa.display
from multiprocessing import Pool, cpu_count


### This script is good for pre 8.0, 64-Neuron models. 


# === CONSTANTS ===
RAW_DATA_PATH = r'data/raw'
AUGMENTED_DATA_PATH = r'data/augmented'
SCALE_PATH = r'scalers'
OUTPUT_CSV = r'data\features\features_v4.csv'
OUTPUT_NPZ = r'data\processed\features_v4.npz'
SPEC_AUGMENT_FREQ_MASK = 4
SPEC_AUGMENT_TIME_MASK = 10
SAMPLE_RATE = 22050  # Common sample rate for librosa
N_MFCC = 13  # Number of MFCCs to compute
NUM_WORKERS = 6

GENRE_LIST = [
    "classic_rock",
    "alternative_rock",
    "alternative",
    "pop_punk",
    "punk",
    "soul",
    "motown",
    "funk",
    "disco",
    "hip-hop",
    "rap",
    "folk",
    "country",
    "pop_country",
    "fusion",
    "jazz",
    "classical",
    "blues",
    "metal",
    "heavy_metal",
    "rock",
    "pop",
    "electronic"
]

# === HELPER FUNCTIONS ===

# Apply SpecAugment on MFCC (modular for future expansion)
def apply_spec_augment(mfcc, freq_mask_param=SPEC_AUGMENT_FREQ_MASK, time_mask_param=SPEC_AUGMENT_TIME_MASK):
    # Frequency masking
    num_mfcc = mfcc.shape[0]
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, num_mfcc - f)
    mfcc[f0:f0+f, :] = 0

    # Time masking
    num_frames = mfcc.shape[1]
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, num_frames - t)
    mfcc[:, t0:t0+t] = 0

    return mfcc

# Function to extract MFCC, Chroma, and Spectral Contrast
def extract_features(audio_path, sr=SAMPLE_RATE):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr, duration=30)  # Ensuring consistent duration
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Apply SpecAugment to MFCC
    mfcc = apply_spec_augment(mfcc)
    
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    return mfcc, chroma, spectral_contrast

# Normalize features and return scaled features and scaler
def normalize_features(features):
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    return features_normalized, scaler

# for multicore support
def process_single_file(args):
    genre_folder, audio_file, audio_file_path, genre_map = args
    genre_label = genre_map[genre_folder]

    filename_no_ext = os.path.splitext(audio_file)[0]
    title = filename_no_ext.split(" - ", 1)[1] if " - " in filename_no_ext else filename_no_ext

    try:
        mfcc, chroma, spectral_contrast = extract_features(audio_file_path)
        feature_vector = np.hstack([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            spectral_contrast.mean(axis=1), spectral_contrast.std(axis=1)
        ])
        if feature_vector.shape[0] != 64:
            raise ValueError(f"Invalid feature vector shape: {feature_vector.shape[0]}")
    except Exception as e:
        print(f"Error: Skipping {title} due to: {e}")
        return None

    return (feature_vector, genre_label, audio_file_path)


# Function to process data from each folder (both raw and augmented)
def process_data(data_path, genre_map):
    all_features = []
    all_labels = []
    all_file_paths = []

    audio_files_info = []
    for genre_folder in os.listdir(data_path):
        genre_folder_path = os.path.join(data_path, genre_folder)
        if not os.path.isdir(genre_folder_path):
            continue

        for audio_file in os.listdir(genre_folder_path):
            if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
                audio_file_path = os.path.join(genre_folder_path, audio_file)
                audio_files_info.append((genre_folder, audio_file, audio_file_path, genre_map))

    total_files = len(audio_files_info)
    print(f"\nFound {total_files} audio files in '{data_path}' to process using {NUM_WORKERS} workers.\n")

    with Pool(processes=NUM_WORKERS) as pool:
        for idx, result in enumerate(pool.imap_unordered(process_single_file, audio_files_info), start=1):
            if result is None:
                continue
            feature_vector, genre_label, audio_file_path = result
            all_features.append(feature_vector)
            all_labels.append(genre_label)
            all_file_paths.append(audio_file_path)
            print(f"[{idx}/{total_files}] Processed: {os.path.basename(audio_file_path)}")

    return np.array(all_features), np.array(all_labels), all_file_paths


# === MAIN PROCESSING PIPELINE ===

def main():
    genre_map = {genre: idx for idx, genre in enumerate(GENRE_LIST)}

    print("\nStarting raw data processing...")
    raw_features, raw_labels, raw_paths = process_data(RAW_DATA_PATH, genre_map)

    print("\nStarting augmented data processing...")
    augmented_features, augmented_labels, augmented_paths = process_data(AUGMENTED_DATA_PATH, genre_map)

    print("\nCombining and normalizing features...")
    all_features = np.vstack([raw_features, augmented_features])
    all_labels = np.hstack([raw_labels, augmented_labels])
    all_paths = raw_paths + augmented_paths
    
    # Assign genre names for later
    genre_names = [GENRE_LIST[label] for label in all_labels]
    
    features_normalized, scaler = normalize_features(all_features)

    if not os.path.exists(SCALE_PATH):
        os.makedirs(SCALE_PATH)
    scaler_file_path = os.path.join(SCALE_PATH, 'feature_scaler.pkl')
    with open(scaler_file_path, 'wb') as f:
        import pickle
        pickle.dump(scaler, f)

    df = pd.DataFrame(features_normalized)
    df['genre'] = all_labels
    df['genre_name'] = genre_names
    df['file_path'] = all_paths

    df.to_csv(OUTPUT_CSV, index=False)
    np.savez_compressed(OUTPUT_NPZ, features=features_normalized, labels=all_labels, genre_names=genre_names)

    print(f"\nData processing complete. Saved to:\n  - CSV: {OUTPUT_CSV}\n  - NPZ: {OUTPUT_NPZ}")

if __name__ == "__main__":
    main()
