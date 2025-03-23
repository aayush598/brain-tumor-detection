import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

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

elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    # Class Distribution
    st.subheader("Class Distribution")
    st.image("eda/ClassDistribution.png", caption="Distribution of Brain Tumor Classes")
    st.write("- **Number of brain_glioma images:** 5000")
    st.write("- **Number of brain_menin images:** 5000")

    # Sample Images
    st.subheader("Sample Images from Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.image("eda/sample_brain_glioma.png", caption="Sample: Brain Glioma", use_column_width=True)
    with col2:
        st.image("eda/sample_brain_menin.png", caption="Sample: Brain Meningioma", use_column_width=True)

    # Class Balance
    st.subheader("Class Balance Percentage")
    st.image("eda/ClassBalancePercentage.png", caption="Class Balance Analysis")

    # Aspect Ratio Distribution
    st.subheader("Aspect Ratio Distribution")
    st.image("eda/AspectRatioDistribution.png", caption="Aspect Ratio of Images")

    # Data Augmentation Visualization
    st.subheader("Data Augmentation Analysis")
    st.image("eda/DataAugmentation.png", caption="Effect of Data Augmentation")

    # Image Quality Analysis
    st.subheader("Image Quality Analysis")
    st.write(f"- **Sharpness (With Mask):** 33.32")
    st.write(f"- **Brightness (With Mask):** 22.59")
    st.write(f"- **Sharpness (Without Mask):** 79.43")
    st.write(f"- **Brightness (Without Mask):** 32.47")

    # Corrupted & Duplicate Images
    st.subheader("Dataset Integrity Check")
    st.write(f"- **Number of corrupted images:** 0")
    st.write(f"- **Number of duplicate images:** 0")


# Model Training Information
elif page == "Model Training":
    st.title("Model Training Details")
    
    # Dataset Information
    st.subheader("Dataset Information")
    st.write("**Dataset Location:** `/kaggle/input/multi-cancer/Multi Cancer/Multi Cancer/Brain Cancer`")
    st.write("**Categories:** `brain_glioma`, `brain_menin`")
    
    # Dynamic Class Distribution
    num_glioma = 5000
    num_menin = 5000

    st.write(f"**Number of Images:**")
    st.write(f"- Brain Glioma: {num_glioma}")
    st.write(f"- Brain Menin: {num_menin}")
    
    fig, ax = plt.subplots()
    ax.bar(["Brain Glioma", "Brain Menin"], [num_glioma, num_menin], color=['blue', 'red'])
    ax.set_ylabel("Number of Images")
    st.pyplot(fig)
    
    # Preprocessing & Augmentation
    st.subheader("Preprocessing & Augmentation")
    st.write("**Image Preprocessing:**")
    st.write("- Resized images to `150x150`")
    st.write("- Applied `preprocess_input` from MobileNetV2")
    
    st.write("**Augmentation using `ImageDataGenerator`**")
    st.write("- Rotation: `20°`")
    st.write("- Width & Height Shift: `0.2`")
    st.write("- Shear: `0.15`")
    st.write("- Zoom: `0.15`")
    st.write("- Horizontal Flip: `Enabled`")
    st.write("- Fill Mode: `Nearest`")
    
    # Model Architecture
    st.subheader("Model Architecture")
    st.write("**Base Model:** `MobileNetV2 (Pretrained)`")
    st.write("**Additional Layers:**")
    st.write("- `AveragePooling2D`\n- `Flatten`\n- `Dense` (Fully Connected Layer)\n- `Dropout` (Regularization)\n- `Softmax Output Layer`")
    
    # Training Configuration
    st.subheader("Training Configuration")
    st.write("- **Epochs:** `2`")
    st.write("- **Batch Size:** `32`")
    st.write("- **Optimizer:** `Adam` with Learning Rate Scheduling")
    st.write("- **Loss Function:** `Categorical Crossentropy`")
    st.write("- **Callbacks Used:**\n  - `EarlyStopping`\n  - `TensorBoard`")
    
    # Hardware Acceleration
    st.subheader("Hardware Acceleration")
    st.write("**GPU Support:** Automatically detects CUDA availability")
    st.write("**Computation Mode:**")
    st.write("- If GPU available → Uses `/GPU:0`")
    st.write("- Otherwise → Uses `/CPU:0`")


# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.image("modelEvaluation/ConfusionMatrix.png", caption="Confusion Matrix", use_column_width=True)
    
    # Accuracy Graph
    st.subheader("Accuracy Over Epochs")
    st.image("modelEvaluation/Accuracy.png", caption="Accuracy Graph", use_column_width=True)
    
    # Loss Graph
    st.subheader("Loss Over Epochs")
    st.image("modelEvaluation/Loss.png", caption="Loss Graph", use_column_width=True)
    
    # Dataset Loading Information
    st.subheader("Dataset Loading Information")
    st.write("Loading brain_glioma images: 100%|██████████| 5000/5000 [00:23<00:00, 213.81image/s]")
    st.write("Loading brain_menin images: 100%|██████████| 5000/5000 [00:24<00:00, 202.71image/s]")
    
    # Training Performance
    st.subheader("Training Performance")
    training_logs = """
    Epoch 1/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 139s 1s/step - accuracy: 0.7628 - loss: 1.0700 - val_accuracy: 0.9415 - val_loss: 0.1463
    Epoch 2/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 121s 1s/step - accuracy: 0.9186 - loss: 0.2206 - val_accuracy: 0.9520 - val_loss: 0.1191
    Epoch 3/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 81s 810ms/step - accuracy: 0.9212 - loss: 0.1977 - val_accuracy: 0.9540 - val_loss: 0.1072
    Epoch 4/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 123s 1s/step - accuracy: 0.9131 - loss: 0.2130 - val_accuracy: 0.9635 - val_loss: 0.0991
    Epoch 5/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 120s 1s/step - accuracy: 0.9230 - loss: 0.1967 - val_accuracy: 0.9600 - val_loss: 0.0980
    Epoch 6/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 80s 796ms/step - accuracy: 0.9379 - loss: 0.1508 - val_accuracy: 0.9595 - val_loss: 0.0949
    Epoch 7/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 123s 1s/step - accuracy: 0.9269 - loss: 0.1857 - val_accuracy: 0.9670 - val_loss: 0.0811
    Epoch 8/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 121s 1s/step - accuracy: 0.9278 - loss: 0.1597 - val_accuracy: 0.9690 - val_loss: 0.0866
    Epoch 9/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 81s 813ms/step - accuracy: 0.9273 - loss: 0.1715 - val_accuracy: 0.9660 - val_loss: 0.0988
    Epoch 10/10
    100/100 ━━━━━━━━━━━━━━━━━━━━ 124s 1s/step - accuracy: 0.9372 - loss: 0.1374 - val_accuracy: 0.9695 - val_loss: 0.0770
    """
    st.text(training_logs)
