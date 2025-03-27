import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# -------------------------------
# Custom CSS for Aesthetic Styling
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        color: #333333;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Load the Model (cached for performance)
# -------------------------------
@st.cache(allow_output_mutation=True)
def load_mnist_model():
    model = load_model('best_mnist_model.h5')
    return model

model = load_mnist_model()

# -------------------------------
# App Title and Description
# -------------------------------
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (JPEG/PNG) and let the model predict the digit.")

# -------------------------------
# File Uploader
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image: convert to grayscale, resize, normalize
    image_gray = ImageOps.grayscale(image)
    image_resized = image_gray.resize((28, 28))
    
    # Optionally, allow the user to invert the image colors if necessary
    invert = st.checkbox("Invert image colors (if digit is dark on light background)")
    img_array = np.array(image_resized).astype('float32')
    if invert:
        img_array = 255 - img_array
    img_array /= 255.0  # Normalize pixel values to [0,1]
    
    # Reshape image to match model's input shape: (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=-1)  # add channel dimension
    img_array = np.expand_dims(img_array, axis=0)    # add batch dimension

    # -------------------------------
    # Make Prediction on Button Click
    # -------------------------------
    if st.button("Predict Digit"):
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        st.success(f"Predicted Digit: {predicted_digit}")