import cv2
import easyocr
import re
from collections import defaultdict
from keras.models import load_model
import numpy as np
from PIL import Image

# Initialize EasyOCR reader with English as the language
reader = easyocr.Reader(['en'])

# Path to the image file
image_file = r'C:\Users\Dell\Desktop\2.jpg'  # Update with your image file path
image_path=image_file   #image prediction path

# Define the mapping of categories to words
category_words = {
    'Vegetable':    ['POTATO', 'TOMATO','CARROT','LEMON'],
    'Fruits':   ['MANGO','BANANA','GRAPES','ORANGE'],
    'Meat': ['BEEF','MUTTON','BONELESS'],
    'Oil & Ghee':   ['OIL','OLIVE','VIRGIN','EXTRA','SUNFLOWER','GHEE'],
    'Camera':   ['CANON','EOS','CMOS','DSLR','NIKON','LENS'],
    'Mobile':   ['IPHONE','OPPO','HONOR','RAVOZ'],
    'TV':   ['TV','UHD','HDMI','TCL','QLED','TELEVISION','ULTRAHD'],
    'Large Appliance':  ['WASHING','REFRIGERATOR','MACHINE','WASHER'],
    'Drinks':   ['WATER','JUICE','DRINK','PEPSI','MIRINDA'],
    'Fish': ['SEABREAM','BARRACUDA','FISH','TUNA','SALMON','TILAPIA','ROHU'],
    'Bakes & Nuts': ['NUTS','BAKES','CHOCOLATE','ALMOND','COOKIES','CAKE','BISCUIT'],
    'Furniture':    ['FURNITURE','KITCHEN','DINING'],
    'Baby Needs': ['PAMPERS','BABY','DIAPERS','HUGGIES','SWADDLERS','WIPES'],
    'Chicken':  ['CHICKEN','SADIA','BREAST','GRILLER','NUGGETS','TENDER'],
    'Tablet':   ['TAB','TABLET','IPAD',]

   
}

# Create a defaultdict with default value 'Unknown'
word_categories = defaultdict(lambda: 'Unknown')

# Update the word_categories mapping with the category-word associations
for category, words in category_words.items():
    for word in words:
        word_categories[word] = category

# Regular expression pattern to match alphanumeric words
pattern = re.compile(r'^[a-zA-Z]+$')

# Load the image
image = cv2.imread(image_file)

# Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform text recognition on the image using EasyOCR
result = reader.readtext(image, detail=0, paragraph=False)

# Extract the words from the result and check their categories
detected_categories = []
for line in result:
    for word in line.split():
        if pattern.match(word):
            category = word_categories[word.upper()]
            if category != 'Unknown':
                detected_categories.append(category)
                

# Print the detected categories
if detected_categories:
    print("Detected categories:")
    for category in set(detected_categories):
        print(category)
else:
     # Load the saved models
    model_path_1 = r'C:\Users\Dell\Desktop\Image Classification\Classifier_Best_2.h5'
    model_path_2 = r'C:\Users\Dell\Desktop\Image Classification\Classifier.h5'

    model_1 = load_model(model_path_1)
    model_2 = load_model(model_path_2)

    # Define the class mapping dictionary
    class_mapping = [
    'TV',
    'Mobiles',
    'Large Appliances',
    'Camera',
    'BabyDiapers',
    'Baby Needs',
    'Bakes&Nuts', 
    'Chicken',
    'Drinks',
    'Fish', 
    'Fruits',
    'Furniture',
    'Meat',
    'Oil&Ghee',
    'Vegs',
    'Control'
    ]

    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((120, 120))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make predictions with model 1
    prediction_1 = model_1.predict(image_array)
    predicted_class_1 = np.argmax(prediction_1)

    # Make predictions with model 2
    prediction_2 = model_2.predict(image_array)
    predicted_class_2 = np.argmax(prediction_2)

    # Check if both models have the same predictions
    if predicted_class_1 == predicted_class_2:
        predicted_class = predicted_class_1
    else:
        predicted_class = -1  # Indicate no match

    # Get the predicted class label
    predicted_label = class_mapping[predicted_class] if predicted_class != -1 else "No match"

    print("Predicted Category:", predicted_label)

    # # Ask for user feedback
    # correct_label = input("Enter the correct label for the image (or 'no' for no match): ")

    # # Update the model based on user feedback
    # if correct_label.lower() == "no":
    #     print("No match found.")
    # else:
    #     # Convert the correct label to class index
    #     corrected_class = class_mapping.index(correct_label)

    #     # Update the correct model with the corrected label
    #     if predicted_class_1 != -1:
    #         model_1.fit(image_array, np.array([corrected_class]), epochs=1)
    #         model_1.save(model_path_1)
    #         print("Model 1 updated with the corrected label.")
    #     if predicted_class_2 != -1:
    #         model_2.fit(image_array, np.array([corrected_class]), epochs=1)
    #         model_2.save(model_path_2)
    #         print("Model 2 updated with the corrected label.")
