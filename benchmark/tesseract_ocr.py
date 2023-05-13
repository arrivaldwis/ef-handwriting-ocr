import os
import pytesseract
import cv2
from pytesseract import Output
import numpy as np

__author__ = 'Arrival Dwi Sentosa <arrivaldwisentosa@gmail.com>'
__source__ = ''
output_path = 'output/images/'

def pre_process_image(image):
    img = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply thresholding to preprocess the image
    img = gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # apply median blurring to remove any blurring
    img = gray = cv2.medianBlur(gray, 3)

    auto_edge = auto_canny_edge_detection(img)
    cv2.imwrite(output_path+"edge.jpg", auto_edge)
    return img

def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0-sigma) * md))
    upper_value = int(min(255, (1.0+sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)

# input file
image_path = "images/0EAC26CF-CAA4-4B5B-B521-EA0E42EF650A.JPG"
image = cv2.imread(image_path)
processed_img = pre_process_image(image)
# save the processed image in the /static/uploads directory
ofilename = os.path.join("preprocessed.png".format(os.getpid()))
cv2.imwrite(output_path+ofilename, processed_img)
# perform OCR on the processed image
text = pytesseract.image_to_string(processed_img, config='--psm 11 --oem 3')
d = pytesseract.image_to_data(image, output_type=Output.DICT)
print(d['text'])
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite(output_path+"box.jpg", image)