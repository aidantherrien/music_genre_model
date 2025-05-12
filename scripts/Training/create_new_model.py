import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
import os

# ==============================================================================
# MODEL CREATION SCRIPT (Subject to change)
# ==============================================================================
# This script is my playground to design and compile models in different ways,
# as it stands, im experimenting with CNN-LTSM Hybrid models.
#

MODEL_PATH = "genre_classifier_model_v8.0.h5"

# Define model parameters
time_steps = 128  # Example value, adjust as per your input shape
num_features = 41  # Example value, adjust according to your feature vector size
num_classes = 23  # Example value, adjust to your number of genres

model = Sequential()

# CNN Layer 1: Extract local features
model.add(Conv1D(64, 3, activation='relu', input_shape=(time_steps, num_features)))
model.add(MaxPooling1D(pool_size=2))

# CNN Layer 2: Further feature extraction
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# LSTM Layer: Capture temporal dependencies
model.add(LSTM(128, return_sequences=True))  # return_sequences=True for stacking more LSTMs
model.add(LSTM(64))  # Reducing the sequence length

# Flatten layer: Convert 2D output to 1D
model.add(Flatten())

# Dense Layer: For final classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization

# Output Layer: Multi-class classification
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Save the model to a file
model_save_path = MODEL_PATH
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
