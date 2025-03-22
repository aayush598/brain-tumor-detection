import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Define the categories
CATEGORIES = ['brain_glioma', 'brain_menin']

# Streamlit UI
st.title("Cancer Classification using CNN")
st.write("Upload multiple images or select from the media folder for classification.")

# Option to select multiple images from the media folder
media_folder = "media"
media_files = [f for f in os.listdir(media_folder) if f.endswith(('jpg', 'png', 'jpeg'))] if os.path.exists(media_folder) else []

selected_files = st.multiselect("Select images from media folder:", media_files)

# Option to upload multiple images
uploaded_files = st.file_uploader("Upload images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Collect all selected images for processing
images_to_process = []

# Load images from uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        images_to_process.append(load_img(uploaded_file, target_size=(224, 224)))

# Load selected images from media folder
if selected_files:
    for selected_file in selected_files:
        image_path = os.path.join(media_folder, selected_file)
        images_to_process.append(load_img(image_path, target_size=(224, 224)))

# Process images when selected or uploaded
if images_to_process:
    for image in images_to_process:
        with st.spinner("Processing... Please wait."):
            # Preprocess image
            image_array = img_to_array(image)
            image_array = preprocess_input(image_array)
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Display the image
            st.image(image, caption="Processed Image", use_column_width=True)
            st.write("Classifying...")

            # Make prediction
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100

            # Display result
            st.write(f"### Prediction: {CATEGORIES[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}%")
