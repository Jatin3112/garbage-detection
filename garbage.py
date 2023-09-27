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
        opencv_image.shape = (1, 224, 224, 3)
        Y_pred = model.predict(opencv_image)
        ypred1 = np.round(Y_pred)
        ypred1 = np.asarray(ypred1, dtype='int')
        
        # Get the predicted label based on the index
        predicted_label_index = ypred1[0]
        
        if 0 <= predicted_label_index < len(labels):
            predict = labels[predicted_label_index]
        else:
            predict = "Unknown"

        st.title(predict)
