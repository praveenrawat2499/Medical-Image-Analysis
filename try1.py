from tensorflow.keras.preprocessing import image
import tensorflow as tf
import streamlit as st
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
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = (predictions > 0.5) * 1

    if score == 0:
        st.write("The MRI image is NOT HAVING a tumor")
    else:
        st.write("The MRI image is HAVING a tumor")
    st.write("Confidence: ", predictions[0][0])
