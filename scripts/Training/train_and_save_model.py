### train_and_save_model.py

# Change before running program
model_path = "C:/Users/aidan\PycharmProjects\genreModel\models\genre_classifier_model_v3.0.keras"

import numpy as np
from tensorflow.keras.models import load_model

# Load preprocessed data
data_path = "C:/Users/aidan\PycharmProjects\genreModel\currentData\processed\processed.npz"
data = np.load(data_path)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# Load the pre-trained model
model = load_model(model_path)
print("Model loaded successfully!")

# Evaluate the current model
current_loss, current_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Current model accuracy: {current_acc}")

# Train the loaded model
model.fit(X_train, y_train, epochs=15, batch_size=16)

# Evaluate the updated model
updated_loss, updated_acc = model.evaluate(X_test, y_test)
print(f"Updated model accuracy: {updated_acc}")

# Save the updated model only if it's better
if updated_acc > current_acc:
    model.save(model_path)
    print(f"Model saved successfully at {model_path} with improved accuracy: {updated_acc}")
else:
    print("Model not saved as the updated accuracy did not improve.")
