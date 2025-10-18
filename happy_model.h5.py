import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import time
import os

# ================= CONFIGURATION =================
MODEL_PATH = "happy_model.h5"  # Trained model file
IMG_SIZE = (48, 48)
CLASS_NAMES = ["Not Happy", "Happy"]

# ================= LOAD MODEL ===================
@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        return model
    else:
        return None  # No model, predictions disabled

# ================= IMAGE PREPROCESS =================
def preprocess_image(image):
    img = image.convert("L")              # Convert to grayscale
    img = img.resize(IMG_SIZE)            # Resize to 48x48
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ================= PREDICTION =================
def predict_image(model, processed_img):
    start = time.time()
    pred_prob = model.predict(processed_img, verbose=0)[0][0]  # Sigmoid output
    end = time.time()
    exec_time = end - start

    if pred_prob >= 0.5:
        prediction = "Happy 😄"
        confidence = pred_prob
    else:
        prediction = "Not Happy 😔"
        confidence = 1 - pred_prob

    return prediction, confidence*100, exec_time

# ================= STREAMLIT APP =================
def main():
    st.set_page_config(page_title="😀 Happy vs Not Happy Classifier", layout="centered")
    st.title("🌟 CNN Image Classifier: Happy vs Not Happy")
    st.markdown("Upload any face image and predict whether the person is **Happy** or **Not Happy**.")

    # Load model
    model = load_cnn_model()
    if model is None:
        st.sidebar.warning("Model file `happy_model.h5` not found. Add it to enable predictions.")
    else:
        st.sidebar.success("✅ Model Loaded Successfully")

    # Upload image
    uploaded_file = st.file_uploader("📂 Upload a face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if model is None:
                st.warning("Cannot make predictions. Model file not found.")
            else:
                processed_img = preprocess_image(image)
                prediction, confidence, exec_time = predict_image(model, processed_img)

                st.subheader("✅ Prediction Result")
                st.markdown(f"**Prediction:** <span style='font-size:36px; color:green'>{prediction}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.info(f"Prediction Time: {exec_time:.4f} seconds")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.info("Please upload a valid image.")

if __name__ == "__main__":
    main()
