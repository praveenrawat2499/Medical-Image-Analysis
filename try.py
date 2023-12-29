from tensorflow.keras.preprocessing import image

import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('C:\\Users\\sahil\\Downloads\\bestmodel.h5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Brain Tumor Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    predictions = import_and_predict(image, model)
    score = tf.nn.sigmoid(predictions[0])

    if score == 0 :
        st.write("The MRI image is NOT HAVING a tumor.")
    else:
        st.write("The MRI image is HAVING a tumor.")
