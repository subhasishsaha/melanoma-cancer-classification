import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = "Model/classifier.tflite"
INPUT_SIZE = (64, 64)

# --- Load TFLite Model (cached) ---
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# --- Preprocessing ---
def preprocess_image(img_pil, target_size=INPUT_SIZE):
    img_pil = img_pil.convert("RGB")
    img_resized = img_pil.resize(target_size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Predict ---
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    if len(output) == 1:
        malignant_prob = float(output[0])
        benign_prob = 1 - malignant_prob
    else:
        benign_prob = float(output[0])
        malignant_prob = float(output[1])

    return benign_prob, malignant_prob

# --- UI ---
st.set_page_config(page_title="Melanoma Detection")

st.title("🔬 Melanoma Detection - TFLite Model")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "png", "jpeg"])
    threshold = st.slider("Malignant Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
    predict_btn = st.button("Predict")

with col2:
    output_label = st.empty()
    output_text = st.empty()
    output_plot = st.empty()

# --- Prediction Logic ---
if predict_btn:
    if uploaded_file is None:
        st.warning("Please upload an image.")
    else:
        try:
            img = Image.open(uploaded_file)

            interpreter = load_tflite_model()
            img_array = preprocess_image(img)
            benign_prob, malignant_prob = predict_tflite(interpreter, img_array)

            is_malignant = malignant_prob > threshold
            prediction_label = "Malignant" if is_malignant else "Benign"
            confidence = malignant_prob if is_malignant else benign_prob

            result_text = f"Prediction: {prediction_label}"
            confidence_text = f"Confidence: {confidence:.2%}"

            if is_malignant:
                message = "⚠️ High risk detected. Please consult a dermatologist."
            else:
                message = "✅ Low risk. Continue regular monitoring."

            full_text = f"{result_text}\n{confidence_text}\n\n{message}"

            # Display outputs
            output_label.metric("Prediction", prediction_label)
            output_text.text(full_text)

            # Plot
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["Benign", "Malignant"], [benign_prob, malignant_prob])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Confidence")
            plt.tight_layout()

            output_plot.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")