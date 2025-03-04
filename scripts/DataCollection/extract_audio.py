# extract_features.py
import numpy as np
import pandas as pd
import os
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# path to your unprocessed data folder (You should see a bunch of empty folders with genre names)
UNPROCESSED_DATA_PATH = r"path_to_folders"

# where you are saving your processed songs to as a single .npz file
PROCESSED_DATA_PATH = r"path_to_save_location"


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    return np.hstack((
        mfccs.mean(axis=1),
        chroma.mean(axis=1),
        spectral_contrast.mean(axis=1)
    ))


def load_and_process_data(data_path):
    genres = os.listdir(data_path)
    print("Genres:", genres)

    data = []
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            data.append([features, genre])

    df = pd.DataFrame(data, columns=["features", "label"])
    df["features"] = df["features"].apply(lambda x: np.array(x))

    # Encode labels numerically
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    # Split dataset
    X = np.array(df["features"].tolist())
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder


if __name__ == "__main__":
    data_path = UNPROCESSED_DATA_PATH
    X_train, X_test, y_train, y_test, encoder = load_and_process_data(data_path)
    np.savez(PROCESSED_DATA_PATH,
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)