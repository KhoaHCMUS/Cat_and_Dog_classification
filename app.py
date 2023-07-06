import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("Model/model.h5", compile=False)

# Define the Streamlit app
def main():
    st.title("Exercise 1 – Image Classification")
    st.markdown("---")

    # Add file uploader to select multiple image files
    uploaded_files = st.file_uploader("Choose multiple image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown("---")

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            st.markdown("---")
            # Predict
            class_name, confidence_score = predict_picture(uploaded_file)

            # Center-align the image
            col_image, col_prediction = st.columns(2)
            with col_image:
                st.image(uploaded_file, caption='Uploaded Image', width=300, use_column_width=True)

            # Show prediction and confidence score
            with col_prediction:
                st.subheader("Prediction Result")
                st.write(f"Image Category: **{class_name}**")
                st.write(f"Confidence Score: **{confidence_score:.4f}**")


def predict_picture(uploaded_file):
    class_names = {0: 'Cat', 1: 'Dog'}

    # Load the image using PIL's Image module
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))  # Resize the image if needed

    # Convert the image to an array
    img_array = image.img_to_array(img)

    # Preprocess the image using VGG19's preprocess_input function
    preprocessed_img = preprocess_input(img_array)

    # Reshape the image to match the expected input shape of the model
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    # Predict the image
    prediction = model.predict(preprocessed_img)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

if __name__ == "__main__":
    main()
