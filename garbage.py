import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('Garbage_MOdel.h5')

labels = ["battery", "biological", "brown_glass", "cardboard", "clothes", "green_glass", "metal", "paper", "plastic", "shoes", "trash", "white_glass"]

st.title('Garbage Classification')
st.markdown('Upload Image')

img = st.file_uploader("Upload image")
submit = st.button('Predict')
if submit:
    if img is not None:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (224, 224))
        opencv_image = np.expand_dims(opencv_image, axis=0)  # Add batch dimension
        Y_pred = model.predict(opencv_image)
        predicted_label_index = np.argmax(Y_pred)  # Get the index with the highest prediction value

        if 0 <= predicted_label_index < len(labels):
            predict = labels[predicted_label_index]
        else:
            predict = "Unknown"

        st.title(predict)
