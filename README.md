# Melanoma Cancer Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

An end-to-end deep learning project for melanoma detection from dermoscopic skin lesion images. The project includes model training, inference, TensorFlow Lite conversion, and an interactive Streamlit application for real-time predictions.

---

## Overview

Melanoma is one of the most aggressive forms of skin cancer. Early detection can significantly improve treatment outcomes.

This project uses a Convolutional Neural Network (CNN) trained from scratch to classify skin lesion images and determine whether a lesion is likely to be melanoma.

The repository contains:

- Model training pipeline
- Trained TensorFlow model
- TensorFlow Lite conversion workflow
- Streamlit deployment interface
- Dataset and experimentation notebooks

---

## Features

### Deep Learning Classification

- Binary skin lesion classification
- Melanoma detection from images
- CNN trained from scratch
- No transfer learning or pretrained backbone

### Interactive Web Application

- Upload lesion images
- Real-time predictions
- User-friendly Streamlit interface

### Edge Deployment Support

- TensorFlow Lite conversion
- Lightweight deployment pipeline
- Mobile and edge-device compatibility

### Reproducible Training Pipeline

- Dataset preprocessing
- Model training notebook
- Saved Keras model artifacts

---

## System Architecture

```text
Skin Lesion Image
        |
        v
 Image Preprocessing
        |
        v
 Convolutional Neural Network
        |
        v
 Probability Prediction
        |
        v
 Melanoma / Non-Melanoma
```

---

## Repository Structure

```text
melanoma-cancer-classification/
│
├── Dataset/
│   └── Training and evaluation images
│
├── Model/
│   ├── classifier.keras
│   └── TensorFlow Lite models
│
├── app.py
├── tf_lite_conversion.py
├── melanoma-cancer-detection-no-pretrained-model.ipynb
├── requirements.txt
└── README.md
```

---

## Technology Stack

### Machine Learning

- TensorFlow
- Keras

### Deployment

- Streamlit
- TensorFlow Lite

### Data Processing

- NumPy
- OpenCV
- Pillow

---

## Installation

### Clone Repository

```bash
git clone https://github.com/subhasishsaha/melanoma-cancer-classification.git
cd melanoma-cancer-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open the generated local URL in your browser and upload a skin lesion image for prediction.

---

## Training the Model

To retrain the model from scratch:

```text
melanoma-cancer-detection-no-pretrained-model.ipynb
```

The notebook contains:

- Data preprocessing
- Model architecture
- Training workflow
- Evaluation
- Model export

---

## TensorFlow Lite Conversion

Convert the trained model for edge deployment:

```bash
python tf_lite_conversion.py
```

This generates a TensorFlow Lite model suitable for mobile and embedded environments.

---

## Applications

- Medical AI research
- Skin lesion analysis
- Computer vision experimentation
- Edge AI deployment
- Healthcare-focused machine learning projects

---

## Future Improvements

- Transfer learning using EfficientNet
- Vision Transformer comparison
- Explainable AI (Grad-CAM)
- Confidence estimation
- Test-time augmentation
- Multi-class skin disease classification
- Mobile application deployment

---

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## License

See the repository LICENSE file for licensing information.
