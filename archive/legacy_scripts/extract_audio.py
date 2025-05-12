# extract_features.py
import numpy as np
import pandas as pd
import os
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

UNPROCESSED_DATA_PATH = r"C:\Users\aidan\PycharmProjects\pythonProject19\current_data"
PROCESSED_DATA_PATH = r"C:\Users\aidan\PycharmProjects\pythonProject19\current_data"

DATASET_VERSION = "v2"

def extract_features_at_intervals(audio_path, frame_length=2048, hop_length=512, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=frame_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)

        features = np.vstack([mfcc, chroma, spectral_contrast])
        return features.T  # shape: (num_frames, total_features)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_and_process_data(data_path):
    genres = os.listdir(data_path)
    print(f"Found genres: {genres}\n")

    data = []
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        print(f"Processing genre: {genre}")
        files = os.listdir(genre_path)
        for idx, file in enumerate(files):
            file_path = os.path.join(genre_path, file)
            print(f"  → [{idx+1}/{len(files)}] Extracting: {file}")
            features = extract_features_at_intervals(file_path)
            if features is not None:
                features_mean = np.mean(features, axis=0)
                data.append([features_mean, genre])
            else:
                print(f"  ⚠️ Skipped {file} due to extraction error.")

    print("\nFinished feature extraction. Converting to DataFrame...")

    df = pd.DataFrame(data, columns=["features", "label"])
    df["features"] = df["features"].apply(lambda x: np.array(x))

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    print("Encoding labels and splitting dataset...")
    X = np.array(df["features"].tolist())
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data processing complete!\n")
    return X_train, X_test, y_train, y_test, encoder

if __name__ == "__main__":
    print("Starting feature extraction pipeline...\n")
    data_path = UNPROCESSED_DATA_PATH
    X_train, X_test, y_train, y_test, encoder = load_and_process_data(data_path)

    print(f"Saving processed data to: {PROCESSED_DATA_PATH}")
    np.savez(PROCESSED_DATA_PATH,
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Save to CSV
    print("Exporting feature matrix to CSV...")
    flat_data = []
    for i in range(len(X_train)):
        row = list(X_train[i]) + [y_train[i]]
        flat_data.append(row)

    mfcc_cols = [f"mfcc_{i+1}" for i in range(13)]
    chroma_cols = [f"chroma_{i+1}" for i in range(12)]
    contrast_cols = [f"contrast_{i+1}" for i in range(7)]
    columns = mfcc_cols + chroma_cols + contrast_cols + ["genre"]

    df_flat = pd.DataFrame(flat_data, columns=columns)
    csv_path = r"C:\Users\aidan\PycharmProjects\pythonProject19\csvs\features_" + DATASET_VERSION + ".csv"
    df_flat.to_csv(csv_path, index=False)

    print(f"\nFeature CSV saved to: {csv_path}")