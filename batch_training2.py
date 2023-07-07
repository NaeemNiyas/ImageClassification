import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os

# Load the training data
with open("Dataset.pkl", "rb") as file:
    x_train = pickle.load(file)

# Load the labels
with open("Dataset_Label.pkl", "rb") as file:
    y_train = pickle.load(file)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

input_shape = x_train[0].shape

# Define batch size and calculate the number of batches
batch_size = 32
num_batches = len(x_train) // batch_size

# Create an instance of Sequential model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu',input_shape=input_shape))
# model.add(MaxPooling2D(2, 2))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(2, 2))

# model.add(Flatten())

# model.add(Dense(128,activation='relu'))
# model.add(Dense(16, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the callbacks
NAME = "Dataset"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}/")
early_stopping = EarlyStopping(monitor="val_accuracy", patience=1, mode="max", verbose=1)
model_checkpoint = ModelCheckpoint(
    "Classifier_Best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

try:
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard, early_stopping, model_checkpoint]
    )
except KeyboardInterrupt:
    print("Training interrupted manually.")

# Evaluate the model on the whole training set
loss, accuracy = model.evaluate(x_train, y_train)

# Print the evaluation results
print("Training Loss:", loss)
print("Training Accuracy:", accuracy)

# Specify the directory to save the trained model
save_directory = r'C:\Users\Dell\Desktop\Image Classification'

# Set the model save path
model_path = os.path.join(save_directory, 'Classifier.h5')

# Save the trained model
model.save(model_path)
print("Model saved successfully.")
