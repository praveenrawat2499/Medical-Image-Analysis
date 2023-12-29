import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing import image

# from keras import preprocess_input
# from keras import decode_predictions


# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('C:\\Users\\hp\\Downloads\\BrainTumor_model.h5')
    return model
model = load_model()

# Define a function to make predictions on uploaded image
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
    prediction = (model.predict(image) > 0.5)*1
    # Get the predicted class label
    # pred_label = np.argmax(prediction[0])
    # Get the corresponding class name
    # class_names = ['No Tumor', 'Tumor']
    # pred_class = class_names[pred_label]
    # Get the confidence score
    # confidence = prediction[0][pred_label]
    # Return the predicted class and confidence score
    # return pred_class, confidence
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
        # Display the predicted class and confidence score
        st.write(f"Prediction: {prediction}")
        # st.write(f"Confidence score: {confidence}")
        if prediction == 1:
            st.write("The MRI image is HAVING a tumor")
        else:
            st.write("The MRI image is NOT HAVING a tumor ")

if __name__ == '__main__':
    main()
