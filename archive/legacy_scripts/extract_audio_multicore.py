import numpy as np
import pandas as pd
import os
import librosa
import random
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# === Configuration ===
RAW_DATA_PATH = r"data\raw"
AUGMENTED_DATA_PATH = r"data\augmented"
DATA_PATHS = [RAW_DATA_PATH, AUGMENTED_DATA_PATH]
PROCESSED_DATA_PATH = r"data\processed"
CSV_DIR = r"data/features"
DATASET_VERSION = "v3"
NUM_CORES = 6  # <-- Set this to the number of cores you want to use


def apply_specaugment(features, time_mask_param=10, freq_mask_param=4, num_time_masks=1, num_freq_masks=1):
    augmented = features.copy()
    
    num_mel_channels, num_time_steps = augmented.shape
    
    # Apply frequency masks
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, num_mel_channels - f)
        augmented[f0:f0 + f, :] = 0

    # Apply time masks
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, num_time_steps - t)
        augmented[:, t0:t0 + t] = 0

    return augmented


def extract_features_at_intervals(audio_path, frame_length=2048, hop_length=512, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=frame_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)

        mfcc = apply_specaugment(mfcc)
        chroma = apply_specaugment(chroma)
        spectral_contrast = apply_specaugment(spectral_contrast)

        features = np.vstack([mfcc, chroma, spectral_contrast])
        return np.mean(features.T, axis=0)
    except Exception as e:
        print(f"[ERROR] Failed to process {audio_path}: {e}")
        return None

def process_file(args):
    file_path, genre = args
    print(f"[INFO] Processing: {os.path.basename(file_path)} from genre: {genre}")
    return [extract_features_at_intervals(file_path), genre]

def load_and_process_data(data_paths):
    file_list = []
    genres_collected = set()

    print("[STEP] Scanning genre folders across multiple sources...")
    for path in data_paths:
        if not os.path.exists(path):
            print(f"[WARNING] Path not found: {path}")
            continue

        genres = os.listdir(path)
        genres_collected.update(genres)
        for genre in genres:
            genre_path = os.path.join(path, genre)
            if not os.path.isdir(genre_path):
                continue
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)
                file_list.append((file_path, genre))

    print(f"[INFO] Detected genres: {sorted(genres_collected)}")
    print(f"[STEP] Starting parallel feature extraction on {len(file_list)} files using {NUM_CORES} CPU cores...")

    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(process_file, file_list)

    data = [res for res in results if res[0] is not None]
    print(f"[STEP] Feature extraction complete. {len(data)} successful out of {len(file_list)}.")

    df = pd.DataFrame(data, columns=["features", "label"])
    df["features"] = df["features"].apply(np.array)

    print("[STEP] Encoding labels...")
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    print("[STEP] Preparing dataset splits and scaling...")
    X = np.array(df["features"].tolist())
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder

if __name__ == "__main__":
    print("Starting dataset processing...")
    X_train, X_test, y_train, y_test, encoder = load_and_process_data(DATA_PATHS)

    print("Saving dataset to .npz...")
    np.savez(
        os.path.join(PROCESSED_DATA_PATH, f"features_{DATASET_VERSION}.npz"),
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    print("Writing flat CSV...")
    flat_data = []
    for i in range(len(X_train)):
        row = list(X_train[i]) + [y_train[i]]
        flat_data.append(row)

    mfcc_cols = [f"mfcc_{i+1}" for i in range(13)]
    chroma_cols = [f"chroma_{i+1}" for i in range(12)]
    contrast_cols = [f"contrast_{i+1}" for i in range(7)]
    columns = mfcc_cols + chroma_cols + contrast_cols + ["genre"]

    df_flat = pd.DataFrame(flat_data, columns=columns)
    csv_path = os.path.join(CSV_DIR, f"features_{DATASET_VERSION}.csv")
    df_flat.to_csv(csv_path, index=False)

    print(f"Extraction complete, CSV saved to: {csv_path}")
