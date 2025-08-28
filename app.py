# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_cnn.h5")

model = load_model()

# Class labels
class_names = ['Cancer', 'Non-Cancer']

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha if present
    return np.expand_dims(img_array, axis=0)

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📘 About", "🔬 Predict"])

# --------------------------
# 🏠 Home Page
# --------------------------
if page == "🏠 Home":
    st.title("🧠 Early Detection of Oral Cancer")
    st.markdown("""
    Welcome to the **Oral Cancer Detection System** powered by Deep Learning.

    Upload an oral image to predict whether it is **Cancerous** or **Non-Cancerous** using a trained Convolutional Neural Network (CNN).

    🔍 Navigate using the sidebar to learn more or make predictions.
    """)

    st.image("https://img.freepik.com/free-photo/medical-oral-examination-dentist_23-2149249824.jpg", use_column_width=True)

# --------------------------
# 📘 About Page
# --------------------------
elif page == "📘 About":
    st.title("📘 About This Project")
    st.markdown("""
    ### 🎯 Objective:
    Early and accurate detection of oral cancer using image classification.

    ### 🛠️ Technologies Used:
    - **TensorFlow / Keras** for deep learning
    - **Streamlit** for the interactive web app
    - **Convolutional Neural Networks (CNN)** for image classification

    ### 📁 Dataset:
    - Two classes: `Cancer` and `Non-Cancer`
    - Images resized to 128x128
    - Preprocessing includes normalization and resizing

    ### 👨‍⚕️ Why It Matters:
    Oral cancer is deadly if not caught early. This tool helps provide quick preliminary results to assist medical professionals.
    """)

# --------------------------
# 🔬 Prediction Page
# --------------------------
elif page == "🔬 Predict":
    st.title("🔬 Oral Cancer Prediction")
    st.markdown("Upload an image of an oral cavity to check if it's cancerous.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]
            pred_label_index = int(prediction > 0.5)
            label = class_names[pred_label_index]
            confidence = prediction if prediction > 0.5 else 1 - prediction

        st.success(f"🩺 **Prediction:** {label}")
        st.info(f"🔬 **Confidence:** {confidence*100:.2f}%")

        if label == "Cancer":
            st.warning("⚠️ This image may indicate signs of oral cancer. Please consult a medical professional.")
        else:
            st.success("✅ No signs of oral cancer detected.")

        st.markdown("---")
        st.markdown("ℹ️ **Note**: This tool is for educational/demo purposes and not a substitute for professional diagnosis.")
