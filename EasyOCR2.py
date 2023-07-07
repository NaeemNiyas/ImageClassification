import cv2
import easyocr
import glob
import re


# Initialize EasyOCR reader with English as the language
reader = easyocr.Reader(['en'])

# File path containing the image
image_path = r'C:\Users\Dell\Desktop\4.jpg'

# Define regular expression pattern to match alphanumeric strings
pattern = re.compile(r'^[a-zA-Z]+$')

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform text recognition on the image using EasyOCR
# result = reader.readtext(image)
result = reader.readtext(image)

# Extract individual words from the result and print them with an index
if result:
    print("Detected words:")
    for i, (_, text, _) in enumerate(result):
        words = text.split()
        # Filter out alphanumeric strings (numbers)
        words = [word for word in words if pattern.match(word)]
        for j, word in enumerate(words):
            print(f"{i+1}.{j+1}:{word}")
else:
    print("No words detected.")
