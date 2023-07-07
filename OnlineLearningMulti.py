from keras.models import load_model
import numpy as np
from PIL import Image

# Define the paths to the saved models
model_paths = [
      r'C:\Users\Dell\Desktop\Image Classification\Mobile_Model.h5',
    r'C:\Users\Dell\Desktop\Image Classification\Camera_Model.h5',
    r'C:\Users\Dell\Desktop\Image Classification\LargeAppliance_Model.h5',
     r'C:\Users\Dell\Desktop\Image Classification\TV_Model.h5'
]

# Define the class labels for each model
class_labels = [
    'Camera',
    'Mobile',
    'Large Appliance',
    'TV'
]

# Load the saved models
models = []
for model_path in model_paths:
    model = load_model(model_path)
    models.append(model)

# Load and preprocess the image
image_path = r'C:\Users\Dell\Desktop\1.jpg'
image = Image.open(image_path)
image = image.resize((120, 120))  # Resize the image to match the input shape of the models
image_array = np.array(image)  # Convert the image to a numpy array
image_array = image_array.astype('float32') / 255.0  # Normalize the pixel values
image_array = np.expand_dims(image_array, axis=0)  # Add an extra dimension as the models expect a batch of images

# Initialize match_found variable
match_found = False

# Iterate through each model until one identifies the image
for i, model in enumerate(models):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    class_label = class_labels[i]
    
    if predicted_class == 0:  # If the model identifies the image, stop iterating
        print("Predicted class:", class_label)
        match_found = True
        break

if not match_found:
    print("No Match")

# Ask for user feedback
correct_label = input("Enter the correct label for the image (or 'no' for no match): ")

# Update the model based on user feedback
if correct_label.lower() == "no":
    print("No match found.")
else:
    for i, model in enumerate(models):
        if class_labels[i].lower() == correct_label.lower():
            # Retrieve the model corresponding to the correct label
            corrected_model = model

            # Prepare the corrected label
            corrected_label = np.array([[0]])  # Assuming the corrected label index is 0

            # Retrain the corrected model using the corrected label
            corrected_model.fit(image_array, corrected_label, epochs=1)  # Retrain the model with the corrected label

            # Save the updated model
            corrected_model.save(model_paths)
            print("Model", class_labels[i], "updated with the corrected label.")
            break
    else:
        print("Invalid label. No model updated.")
