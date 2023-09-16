import numpy as np
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
#import pytesseract
import easyocr
reader = easyocr.Reader(['en'])

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
img_file_buffer = st.camera_input("Take a picture")


 ##PROCESSING STEPS
if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)
    img_size = np.shape(img_array)

    #crop image to only right side
    img_array = img_array[:, int(np.floor(img_size[1]/2)):img_size[1], :]
    cv2.imwrite('im1.png', img_array)

    # load the image as grayscale
    imgc = cv2.imread('im1.png',cv2.IMREAD_GRAYSCALE)

    # Change all pixels to black or white
    imgc[imgc > 128] = 255
    imgc[imgc <= 128] = 0

    # Scale it 10x
    scaled = cv2.resize(imgc, (0,0), fx=20, fy=20, interpolation = cv2.INTER_CUBIC)
    #io.imshow(scaled)

    # Retained your bilateral filter
    filtered = cv2.bilateralFilter(scaled, 11, 17, 17)

    # Thresholded OTSU method
    thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Erode the image to bulk it up for tesseract
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations = 2)
    cv2.imwrite('im1.png', eroded)
    
    result = reader.readtext("im1.png",paragraph="False", detail=0, allowlist="0,1,2,3,4,5,6,7,8,9,/,E,C,G,")


    #result= pytesseract.image_to_string(img) #, config=custom_config)

    st.write(str(result));



with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
