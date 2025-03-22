import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Define the categories
CATEGORIES = ['brain_glioma', 'brain_menin']

# Streamlit UI
st.title("Cancer Classification using CNN")
st.write("Upload an image to classify it as Brain Glioma or Brain Meningioma.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Display uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    
    # Display result
    st.write(f"### Prediction: {CATEGORIES[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
