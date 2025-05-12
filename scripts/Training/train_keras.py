import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


"""
Genre Classification Model Training Script
------------------------------------------
This script loads preprocessed audio feature data from a CSV file and trains a neural network
to classify music tracks into one of 23 genres. The model uses a multilayer perceptron (MLP) 
architecture with L2 regularization, dropout, and batch normalization for improved generalization.

Note - For use with .keras models.

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
DATA_CSV_PATH = r'data\features\features_v4.csv'  # Specify the path to your CSV
MODEL_SAVE_PATH = r'models\genre_classifier_model_v7.2.keras'  # Path to save the trained model
INPUT_DIM = 64  # Number of input features
NUM_CLASSES = 23  # Change if needed (number of genres)
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
EPOCHS = 10
TEMPERATURE = 1.0  # Default temperature (can be changed)

# Load data
df = pd.read_csv(DATA_CSV_PATH)

# Drop unnecessary columns (genre_name and file_path)
df = df.drop(columns=['genre_name', 'file_path'], errors='ignore')

# Ensure the number of columns matches
if df.shape[1] != INPUT_DIM + 1:
    raise ValueError(f"Expected {INPUT_DIM + 1} columns in CSV, got {df.shape[1]}")

# Split features and target
X = df.iloc[:, :-1].values  # Features (first 64 columns)
y = df.iloc[:, -1].values  # Target (last column, genre)

# Label encoding for the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert string labels to integers

# Train/test split (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalize the data
X_train, X_val, X_test = map(lambda x: x / x.max(axis=0), [X_train, X_val, X_test])

# Build model
def build_genre_mlp():
    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Load existing model if available, otherwise build new
try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print("Loaded existing model.")
except:
    print("No existing model found. Building a new model.")
    model = build_genre_mlp()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and store history for plotting
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))


# Plot training & validation loss and accuracy
def plot_metrics(history):
    # Plot training & validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(history)


# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save(MODEL_SAVE_PATH)
print("Model saved to", MODEL_SAVE_PATH)

# Function for Temperature Scaling during Inference
def temperature_scaling(logits, temperature=1.0):
    """Apply temperature scaling to logits."""
    logits = logits / temperature  # Apply the temperature
    return tf.nn.softmax(logits)   # Softmax to get the probabilities

# Function to make predictions with temperature scaling
def predict_with_temperature(model, input_data, temperature=1.0):
    logits = model(input_data)  # Get raw logits from the model
    probs = temperature_scaling(logits, temperature)  # Apply temperature scaling
    return probs

# Example usage for prediction
input_data = np.array(X_test[0:1])  # Using one example from the test set
scaled_probs = predict_with_temperature(model, input_data, TEMPERATURE)
predicted_class = np.argmax(scaled_probs)  # Get the predicted class
predicted_genre = label_encoder.inverse_transform([predicted_class])  # Convert to genre name

print("Predicted genre with temperature scaling:", predicted_genre[0])
