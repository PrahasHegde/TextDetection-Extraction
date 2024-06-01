# Import required packages
import cv2
import pytesseract
import matplotlib.pyplot as plt

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image from which text needs to be extracted
img = cv2.imread("C:\\Users\\hegde\\OneDrive\\Desktop\\Text Recognition-Extraction\\textimg2.jpg")

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Dilation parameter, bigger means less rect
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Creating a copy of the image
im2 = img.copy()

# List to hold contour details
cnt_list = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Cropping the text block for giving input to OCR
    cropped = gray[y:y + h, x:x + w]
    
    # Apply OCR on the cropped image
    config = "--psm 6"  # Set Page Segmentation Mode to single block
    text = pytesseract.image_to_string(cropped, config=config)
    
    if text.strip():  # Avoid adding empty text
        cnt_list.append([x, y, text.strip()])

# Merge adjacent text blocks within a threshold distance
merged_text_blocks = []
threshold_distance = 20

for i in range(len(cnt_list) - 1):
    x1, y1, text1 = cnt_list[i]
    x2, y2, text2 = cnt_list[i + 1]

    if abs(y2 - y1) < threshold_distance:
        # Merge text blocks
        cnt_list[i + 1] = [min(x1, x2), min(y1, y2), text1 + ' ' + text2]
    else:
        # Add current merged block to list
        merged_text_blocks.append(cnt_list[i])

# Add the last block
merged_text_blocks.append(cnt_list[-1])

# Open the file in write mode with UTF-8 encoding
with open("recognized.txt", "w", encoding="utf-8") as file:
    paragraph = ' '.join([text for _, _, text in merged_text_blocks])
    file.write(paragraph)

# Read the text from the file
with open("recognized.txt", "r", encoding="utf-8") as file:
    text = file.read()
    cleaned_text = ''.join(char if ord(char) < 128 else ' ' for char in text)

    # Print the cleaned text
    print(cleaned_text)

# Read and resize images for display
rgb_image = cv2.resize(im2, (0, 0), fx=0.4, fy=0.4)
dilation = cv2.resize(dilation, (0, 0), fx=0.4, fy=0.4)

# Show the images, provide window name first
cv2.imshow('dilation', dilation)
cv2.imshow('gray', gray)

# Add wait key, window waits until user presses a key
cv2.waitKey(0)

# Destroy/close all open windows
cv2.destroyAllWindows()
