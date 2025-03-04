import numpy as np
import pandas as pd
import os
import librosa.display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Extract audio features"""
# Load and inspect data
data_path = "/data/archive/Data/genres_original"
genres = os.listdir(data_path)

print("Genres:", genres)
for genre in genres:
    files = os.listdir(os.path.join(data_path, genre))
    print(f"{genre}: {len(files)} files")

# Extract Audio features
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

# Apply feature extraction to entire dataset
data = []
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        features = extract_features(file_path)
        data.append([features, genre])

df = pd.DataFrame(data, columns=["features", "label"])
df["features"] = df["features"].apply(lambda x: np.array(x))

"""Prep data for training"""

# Encode labels numerically
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

# Split the dataset into training and testing sets
X = np.array(df["features"].tolist())
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Input(shape=(32,)),  # Input layer specifying shape
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


print("Model created successfully!")

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save Model
model.save('C:/Users/aidan\PycharmProjects\genreModel\models/genre_classifier_model_v1.0.keras')



