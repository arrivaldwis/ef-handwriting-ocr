import cv2
import easyocr
import streamlit as st
import numpy as np
from PIL import Image
from clearml import Task

# Handwriting OCR for eFishery Test Case, Problem 1
# - OCR Exploration: Tesseract, Keras+Tensorflow, EasyOCR
# -- EasyOCR comes with best detection
# - Preprocessing Function
# - Fine tuning EasyOCR best parameter (minimum slope and width ths to detect small box)
# - Define confidence threshold parameter
# - Draw bounding boxes
# - Implementation of Streamlit and ClearML

__author__ = 'Arrival Dwi Sentosa <arrivaldwisentosa@gmail.com>'
__source__ = ''

# streamlit
st.title('Problem 1: Handwriting OCR')
st.write('Harap menunggu jika input file belum muncul, sedang initialization ClearML..')

# apply threshold filter
threshold = st.slider('Threshold confidence level?', 0, 100, 25)
threshold = threshold / 100

# clearml init
task = Task.init(project_name="ef-handwriting-ocr", task_name="ocr")

def pre_process(image):
    img = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply thresholding to preprocess the image
    img = gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # apply median blurring to remove any blurring
    img = gray = cv2.medianBlur(gray, 3)
    return img

#read image 
uploaded_file = st.file_uploader("Upload the image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # instance text detector
    reader = easyocr.Reader(["en"],["id"])
    text_ = reader.readtext(pre_process(img_array), slope_ths=0.01, width_ths=0.01)

    text_list = []
    # draw bbox and text
    for t_,t in enumerate(text_):
        bbox , text, score = t
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        # filter score/confidence threshold to draw
        if score > threshold:
            cv2.rectangle(img_array, tl, br,(0,255,0),5)
            cv2.putText(img_array, text, (tl[0], tl[1] - 10),cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
            desc = '{text:'+text+', confidence:'+str(score)+'}'
            text_list.append(desc)

    # write the result
    st.image(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
    st.write("Confidence threshold: "+str(threshold))
    st.write(text_list)