import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import cv2
from PIL import Image
from skimage.feature import hog

# Load Models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
cnn_model = tf.keras.models.load_model("cnn_model.h5")

# Load Label Encoder (for Random Forest)
label_encoder = joblib.load("label_encoder.pkl")

cnn_input_shape = cnn_model.input_shape
CNN_HEIGHT = cnn_input_shape[1]
CNN_WIDTH = cnn_input_shape[2]
CNN_CHANNELS = cnn_input_shape[3]

st.set_page_config(page_title="Breast Cancer Detection", layout="centered")
st.title("Breast Cancer Classification Using Machine Learning")
st.write("Upload a histopathology image to classify as Normal, Benign, or Malignant.")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ("SVM", "Random Forest", "CNN")
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Uploaded Image", width=400)

    image_array = np.array(original_image)

    if st.button("Predict"):

        # SVM & RANDOM FOREST
        if model_choice in ["SVM", "Random Forest"]:

            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            # Resize to 128x128
            gray = cv2.resize(gray, (128, 128))

            # Extract HOG features
            fd, _ = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=True
            )

            img_hog = fd.reshape(1, -1)

            # SVM
            if model_choice == "SVM":
                prediction = svm_model.predict(img_hog)[0]
                probabilities = svm_model.predict_proba(img_hog)[0]

                class_names = ["normal", "benign", "malignant"]

            # RANDOM FOREST
            else:
                pred_encoded = rf_model.predict(img_hog)
                probabilities = rf_model.predict_proba(img_hog)[0]

                predicted_label = label_encoder.inverse_transform(pred_encoded)[0]
                class_names = label_encoder.classes_

        # CNN
        else:

            rgb = cv2.resize(image_array, (CNN_WIDTH, CNN_HEIGHT))

            if CNN_CHANNELS == 1:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                rgb = rgb.reshape(CNN_HEIGHT, CNN_WIDTH, 1)

            rgb = rgb / 255.0
            img_cnn = rgb.reshape(1, CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS)

            probabilities = cnn_model.predict(img_cnn)[0]
            prediction = np.argmax(probabilities)

            class_names = ["normal", "benign", "malignant"]
            predicted_label = class_names[prediction]

        # DISPLAY RESULT
        if model_choice == "Random Forest":
            final_label = predicted_label
        else:
            final_label = class_names[prediction]

        if final_label.lower() == "malignant":
            st.error("Prediction: MALIGNANT")
        elif final_label.lower() == "benign":
            st.warning("Prediction: BENIGN")
        else:
            st.success("Prediction: NORMAL")

        # DISPLAY PROBABILITIES
        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(class_names):
            confidence = probabilities[i] * 100
            st.write(f"{class_name.capitalize()}: {confidence:.2f}%")
            st.progress(float(probabilities[i]))