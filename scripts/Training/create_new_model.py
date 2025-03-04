### create_new_model.py

# Change version number with each new iteration
filename = r"genre_classifier_model_vX.Y.keras"

# Where you save your model to
model_directory = r"path_to_your_model_directory"


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
import os


# Define a function to create a new model
def create_model():
    model = Sequential([
        Input(shape=(32,)),  # Adjust this based on your features
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')  # 10 output neurons for 10 genres
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("New model created successfully!")
    return model


# Create a new model
model = create_model()

# Save the new model
models_dir = model_directory
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, filename)
model.save(model_path)
print(f"Model saved successfully at {model_path}")
