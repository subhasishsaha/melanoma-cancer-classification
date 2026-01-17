import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = "Model/classifier.tflite"
INPUT_SIZE = (64, 64)

# --- Load TFLite Model ---
# Gradio doesn't have a direct equivalent to st.cache_resource, 
# but we can load the model globally or use a singleton pattern.
# For simplicity, we'll load it globally or lazily.
interpreter = None

def load_tflite_model():
    global interpreter
    if interpreter is None:
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

# --- Gradio Logic ---
def predict_fn(image, threshold):
    if image is None:
        return "Please upload an image.", "", None

    try:
        # Image comes as a numpy array from Gradio by default, convert to PIL
        img = Image.fromarray(image)
        
        interp = load_tflite_model()
        img_array = preprocess_image(img)
        benign_prob, malignant_prob = predict_tflite(interp, img_array)

        is_malignant = malignant_prob > threshold
        prediction_label = "Malignant" if is_malignant else "Benign"
        confidence = malignant_prob if is_malignant else benign_prob
        
        # Result Text
        result_text = f"Prediction: {prediction_label}"
        confidence_text = f"Confidence: {confidence:.2%}"
        
        if is_malignant:
            message = "⚠️ High risk detected. Please consult a dermatologist."
        else:
            message = "✅ Low risk. Continue regular monitoring."
            
        full_text = f"{result_text}\n{confidence_text}\n\n{message}"

        # Plot
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Benign", "Malignant"], [benign_prob, malignant_prob], color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        plt.tight_layout()
        
        return prediction_label, full_text, fig
        
    except Exception as e:
        return "Error", f"Error processing image: {str(e)}", None

# --- Gradio Interface ---
with gr.Blocks(title="Melanoma Detection") as demo:
    gr.Markdown("# 🔬 Melanoma Detection - TFLite Model")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload a dermoscopic image", type="numpy")
            threshold_slider = gr.Slider(minimum=0.5, maximum=0.95, value=0.7, step=0.05, label="Malignant Confidence Threshold")
            submit_btn = gr.Button("Predict", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(label="Prediction")
            output_text = gr.Textbox(label="Details", lines=4)
            output_plot = gr.Plot(label="Confidence Plot")

    submit_btn.click(
        fn=predict_fn,
        inputs=[input_image, threshold_slider],
        outputs=[output_label, output_text, output_plot]
    )

if __name__ == "__main__":
    demo.launch()
