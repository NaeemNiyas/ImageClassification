import cv2
import easyocr
import re
from collections import defaultdict

# Initialize EasyOCR reader with English as the language
reader = easyocr.Reader(['en'])

# Path to the image file
image_file = r'C:\Users\Dell\Desktop\9.jpg'  # Update with your image file path

# Define the mapping of categories to words
category_words = {
    'Vegetable':    ['POTATO', 'EGYPT'],
    'Fruits':   ['MANGO','BANANA'],
    'Meat': ['BEEF','MUTTON'],
    'Oil & Ghee':   ['OIL','OLIVE','VIRGIN','EXTRA','SUNFLOWER'],
    'Camera':   ['CANON','CAMERA','EOS','CMOS','DSLR','NIKON','LENS'],
    'Mobile':   ['IPHONE','OPPO','HONOR','RAVOZ'],
    'TV':   ['TV','UHD','HDMI','TCL','QLED','HISENSE','TELEVISION','ULTRAHD'],
    'Large Appliance':  ['WASHING','REFRIGERATOR','MACHINE','WASHER'],
    'Drinks':   ['WATER','JUICE','DRINK','PEPSI','MIRINDA'],
    'Fish': ['SEABREAM','BARRACUDA','FISH'],
    'Bakes & Nuts': ['NUTS','BAKES','CHOCOLATE','ALMOND','COOKIES'],
    'Furniture':    ['FURNITURE','KITCHEN','DINING'],
    'Baby Diapers': ['PAMPERS','BABY','DIAPERS','HUGGIES','SWADDLERS'],
    'Chicken':  ['CHICKEN','SADIA','BREAST','GRILLER'],
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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform text recognition on the image using EasyOCR
result = reader.readtext(gray, detail=0, paragraph=False)

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
    print("No match found.")
