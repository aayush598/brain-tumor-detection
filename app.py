import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

model = load_model()

# Define the categories
CATEGORIES = ['brain_glioma', 'brain_menin']

# Define Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Dataset Overview", "EDA", "Model Training", "Model Evaluation"])

# Home Page
if page == "Home":
    st.title("🧠 Brain Tumor Classification using CNN")
    st.write(
        """
        This project aims to classify brain tumor images using a Convolutional Neural Network (CNN).  
        You can navigate through different sections using the sidebar.  
        """
    )
    st.image("media/brain_glioma_0001.jpg", caption="Brain Tumor MRI Sample", use_column_width=True)

# Prediction Page
elif page == "Prediction":
    st.title("🩺 Brain Tumor Prediction")

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
                st.write(f"### 🧬 Prediction: {CATEGORIES[predicted_class]}")
                st.write(f"✅ Confidence: {confidence:.2f}%")

elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write(
        """
        **Brain Cancer Dataset**  
        
        - **Source:** Compiled using images provided in a [Figshare dataset](https://doi.org/10.6084/m9.figshare.1512427.v8).  
        - **Total Images:** 15,000 covering three main types of brain cancer.
        - **Path:** `/Brain Cancer`
        
        ### **Dataset Categories:**
        - **Glioma (`/brain_glioma`)** - The most common type of brain tumor.
        - **Meningioma (`/brain_menin`)** - Tumors affecting brain membranes.
        - **Pituitary Tumor (`/brain_tumor`)** - Tumors affecting the pituitary gland.
        
        The dataset contains **3,064 T1-weighted contrast-enhanced MRI images** with three types of brain tumors.  
        
        **Dataset Information:**
        - First Online Date: April 3, 2017
        - Latest Update: December 21, 2024
        - Includes MATLAB code for `.mat` to `.jpg` conversion.
        """
    )

    st.image("media/brain_glioma_0001.jpg", caption="Sample Brain MRI Dataset", use_column_width=True)

# Exploratory Data Analysis (EDA)
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    st.write(
        """
        Exploratory Data Analysis (EDA) helps in understanding the distribution of the dataset,  
        the number of images per category, and visual patterns in the MRI scans.  
        """
    )
    st.image("eda_chart.jpg", caption="EDA Analysis", use_column_width=True)

# Model Training Information
elif page == "Model Training":
    st.title("🚀 Model Training Information")
    st.write(
        """
        The model is trained using a **Convolutional Neural Network (CNN)** architecture based on **MobileNetV2**.  
        - Optimizer: **Adam**  
        - Loss Function: **Categorical Crossentropy**  
        - Learning Rate: **0.0001**  
        - Training Epochs: **50**  
        """
    )
    st.image("cnn_architecture.jpg", caption="CNN Model Architecture", use_column_width=True)

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")
    st.write(
        """
        The trained model is evaluated based on accuracy, precision, recall, and F1-score.  
        Below are the evaluation metrics from the validation dataset:  
        - **Accuracy:** 92.5%  
        - **Precision:** 90.8%  
        - **Recall:** 91.2%  
        - **F1-Score:** 91.0%  
        """
    )
    st.image("confusion_matrix.jpg", caption="Confusion Matrix", use_column_width=True)
