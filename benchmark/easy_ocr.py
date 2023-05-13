import cv2
import easyocr
import matplotlib.pyplot as plt

__author__ = 'Arrival Dwi Sentosa <arrivaldwisentosa@gmail.com>'
__source__ = ''
output_path = 'output/images/'

def pre_process(image):
    img = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply thresholding to preprocess the image
    img = gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # apply median blurring to remove any blurring
    img = gray = cv2.medianBlur(gray, 3)
    return img

#read image 
image_path = "images/0EAC26CF-CAA4-4B5B-B521-EA0E42EF650A.JPG"
img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(["en"],["id"])
text_ = reader.readtext(pre_process(img), slope_ths=0.01, width_ths=0.01)
threshold = 0.25

# draw bbox and text
for t_,t in enumerate(text_):
    bbox , text, score = t
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    
    if score > threshold:
        cv2.rectangle(img, tl, br,(0,255,0),5)
        cv2.putText(img, text, (tl[0], tl[1] - 10),cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
        print(text)

cv2.imwrite(output_path+"result_thresh_blur_slope_min.jpg", cv2.cvtColor(img,cv2.COLOR_BGR2RGB))