from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Input(shape=(100,)),  # Input layer specifying shape
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




