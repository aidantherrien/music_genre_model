import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


"""
Genre Classification Model Training Script
------------------------------------------
This script loads preprocessed audio feature data from a CSV file and trains a neural network
to classify music tracks into one of 23 genres. The model uses a hybrid CNN-LTSM 

Note - For use with .h5 models.

Features:
- Loads data and encodes genre labels
- Normalizes features using training set statistics
- Splits data into training, validation, and test sets (80/10/10 split)
- Trains or loads an existing Keras model
- Visualizes training and validation performance
- Evaluates test accuracy
- Supports temperature scaling for calibrated softmax predictions
- Saves model and label encoder for future inference

Dependencies:
- pandas, numpy, tensorflow, scikit-learn, matplotlib

"""


# Constants
DATA_NPZ_PATH = r'data\processed\features_v8.npz'
MODEL_SAVE_PATH = r'models\genre_classifier_model_v8.6.h5'
TEST_SPLIT_SAVE_PATH = r'data\processed\test_split_v8.npz'
TIME_STEPS = 128
NUM_FEATURES = 41
NUM_CLASSES = 23
BATCH_SIZE = 64
EPOCHS = 20
TEMPERATURE = 2.0

# Genre list (in order corresponding to neurons in the model)
GENRES = [
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

# Mapping from genre to class index
GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}

# Mapping from class index back to genre
INDEX_TO_GENRE = {i: genre for i, genre in enumerate(GENRES)}



# Load preprocessed data from NPZ
data = np.load(DATA_NPZ_PATH)
X = data['X']  # Shape: (samples, time_steps, num_features)
y = data['y']  # Genre labels as one hot vectors


# Reverse one-hot encoded labels to class indices
y_indices = np.argmax(y, axis=1)  # Shape: (16559,)

# Now y_indices is 1D and ready for stratification and splitting
X_train, X_temp, y_train_indices, y_temp_indices = train_test_split(
    X, y_indices, test_size=0.2, stratify=y_indices, random_state=42
)
X_val, X_test, y_val_indices, y_test_indices = train_test_split(
    X_temp, y_temp_indices, test_size=0.5, stratify=y_temp_indices, random_state=42
)

# Re-one-hot encode the labels for training
y_train = to_categorical(y_train_indices, num_classes=23)
y_val = to_categorical(y_val_indices, num_classes=23)
y_test = to_categorical(y_test_indices, num_classes=23)


# Cache test set
np.savez(TEST_SPLIT_SAVE_PATH, X_test=X_test, y_test=y_test)
print(f"Test set saved to {TEST_SPLIT_SAVE_PATH}")

# Normalize each set independently (safe division)
def normalize(data):
    max_vals = np.max(data, axis=(0, 1), keepdims=True)
    return data / np.where(max_vals == 0, 1, max_vals)

X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

# Load or build model
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)
else:
    print("Building new model...")
    model = Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(TIME_STEPS, NUM_FEATURES)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train model
    history = model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Plot training metrics
    def plot_metrics(history):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_metrics(history)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

# Temperature scaling
def temperature_scaling(logits, temperature=1.0):
    logits = logits / temperature
    return tf.nn.softmax(logits)

# Inference example
sample_input = X_test[0:1]
logits = model(sample_input)
scaled_probs = temperature_scaling(logits, TEMPERATURE)
predicted_class = np.argmax(scaled_probs)
predicted_genre = INDEX_TO_GENRE[predicted_class]

print("Predicted genre with temperature scaling:", predicted_genre)

