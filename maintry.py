
from streamlit_option_menu import option_menu
import tensorflow as tf
import streamlit as st
import numpy as np
from keras.models import load_model
import base64
import streamlit as st
from PIL import Image
# import hydralit_components as hc


# Adding CSS and Background image
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def add_bg(bg_image_file):
    with open(bg_image_file, "rb") as bg_file:
        bg_encoded_string = base64.b64encode(bg_file.read())

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{bg_encoded_string.decode()});
            background-size: cover;
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
                            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg('C:\\Users\\hp\\PycharmProjects\\pythonProject\\bg4.jpg')


#sidebar

img = get_img_as_base64("C:\\Users\\hp\\PycharmProjects\\pythonProject\\bg5.jpg")


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("../pythonProject/bg.jpg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: repeat;
background-size: cover;
position: relative;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

[data-testid="stSidebar"] > 

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Importing Models
model = load_model("C:\\Users\\hp\\Downloads\\BrainTumor_model.h5")
model1 = tf.keras.models.load_model("C:\\Users\\hp\\Downloads\\CTmodel.h5")

# sidebar for navigation
with st.sidebar:

    selected = option_menu('Medical Image Analysis System',

                           ['Home','Brain Tumor Prediction',
                            'Chest Cancer Prediction',
                            ],
                           icons=['house', '🌺', 'skeleton-rib'],
                           default_index=0)


#Home
if (selected == 'Home'):
    # healthcare-image-analysis.svg
    st.markdown(
                "<h1 style='text-align: center; color: #6667AB; font-size: 50px; font-style: oblique; margin-bottom: 10px; margin: 5px;'>"
                "Medical Image Analysis System</h1>",
                unsafe_allow_html=True)

    # Load the image
    image = Image.open("C:\\Users\\hp\\PycharmProjects\\pythonProject\\healthcare-image-analysis.jpg")

   # Get the current size of the image
    current_width, current_height = image.size

    # Calculate the desired size based on your desired width and height
    desired_width = 2800
    aspect_ratio = current_height / current_width
    desired_height = int(desired_width * aspect_ratio)

    # Resize the image to your desired size
    resized_image = image.resize((desired_width, desired_height))

    # Display the resized image
    st.image(resized_image)

    st.write(
        "<h3 style='text-align: center; color: #FA8072; font-size: 25px; font-style: oblique; font-family: Sofia; margin-top: 10px;'>"
        "Here You can you can detect the Brain tumor by Brain MRI image and Chest cancer by Chest CT-Scan Image by uploding the Image in our Model.</h3>",
        unsafe_allow_html=True)

# Brain Tumor Page


if (selected == 'Brain Tumor Prediction'):

    def predict(image):
        # Resize the image to 224x224 pixels and convert to RGB
        size = (224, 224)
        image = image.convert('RGB')
        image = image.resize(size)
        # Convert the image to a numpy array
        image = np.asarray(image)
        # Normalize the pixel values
        image = image / 255
        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
        # Make the prediction
        prediction = (model.predict(image) > 0.5) * 1

        return prediction


    # Define the Streamlit app
    def main():
        st.title('Brain Tumor Detection')
        # Display a file uploader widget
        file = st.file_uploader('Please upload an MRI image', type=['jpg', 'jpeg', 'png'])
        # If a file was uploaded
        if file is not None:
            # Display the uploaded image
            image = Image.open(file)
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
            # Make predictions on the uploaded image
            prediction = predict(image)

            if prediction == 1:

                # text = "The MRI image is HAVING a tumor"
                st.write("<h3 style='text-align: center; color: #1f64d0; font-size: 25px; font-style: oblique; font-family: Sofia; margin-top: 10px;'>"
                            "The MRI IMAGE IS HAVING A TUMOR</h3>", unsafe_allow_html=True)
            else:

                # text1 = "The MRI image is NOT HAVING a tumor "
                st.write(
                    "<h3 style='text-align: center; color: #1f64d0; font-size: 25px; font-style: oblique; font-family: Sofia; margin-top: 10px;'>"
                    "The MRI IMAGE IS NOT HAVING ANY TUMOR</h3>", unsafe_allow_html=True)

    if __name__ == '__main__':
        main()



if (selected == 'Chest Cancer Prediction'):

    def predict(image):
        # Resize the image to 224x224 pixels and convert to RGB
        size = (224, 224)
        image = image.convert('RGB')
        image = image.resize(size)
        # Convert the image to a numpy array
        image = np.asarray(image)
        # Normalize the pixel values
        image = image / 255
        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
        # Make the prediction
        prediction = (model1.predict(image) > 0.5) * 1

        return prediction


    # Define the Streamlit app
    def main():
        st.title('Chest Cancer Detection')
        # Display a file uploader widget
        file = st.file_uploader('Please upload an CT-Scan image', type=['jpg', 'jpeg', 'png'])
        # If a file was uploaded
        if file is not None:
            # Display the uploaded image
            image = Image.open(file)
            st.image(image, caption='Uploaded CT-Scan Image', use_column_width=True)
            # Make predictions on the uploaded image
            prediction = predict(image)
            # Display the predicted class and confidence score
            st.write(f"Prediction: {prediction}")
            # st.write(f"Confidence score: {confidence}")
            if prediction == 0:
                st.write("<h3 style='text-align: center; color: #1f64d0; font-size: 25px; font-style: oblique; font-family: Sofia; margin-top: 10px;'>"
                            "The CT-SCAN IMAGE IS HAVING A CANCER</h3>", unsafe_allow_html=True)
            else:
                st.write(
                    "<h3 style='text-align: center; color: #1f64d0; font-size: 25px; font-style: oblique; font-family: Sofia; margin-top: 10px;'>"
                    "The CT-SCAN IMAGE IS NOT HAVING A CANCER</h3>", unsafe_allow_html=True)
                # st.markdown("The CT-Scan image is NOT HAVING a Cancer ")


    if __name__ == '__main__':
        main()
