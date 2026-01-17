# Melanoma Cancer Classification

This repository provides an end-to-end pipeline for melanoma cancer classification using deep learning. It leverages **TensorFlow** and **Streamlit** to build and deploy a convolutional neural network (CNN) for image-based melanoma classification.

## Features

- **Image Classification**: Detects melanoma from skin lesion images using a CNN.
- **Streamlit Web App**: Simple web interface for users to upload images and view predictions.
- **No Pretrained Models**: Models are trained from scratch for this specific use-case.
- **TensorFlow Lite Conversion**: Includes scripts for model conversion to TensorFlow Lite, useful for deployment on edge devices.
- **Notebook & Scripts**: Jupyter Notebook for exploration and model-building, Python scripts for model operations and app serving.

## How to Run

1. **Clone the Repository**
    ```
    git clone https://github.com/subhasishsaha/melanoma-cancer-classification.git
    cd melanoma-cancer-classification
    ```

2. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**
    ```
    streamlit run app.py
    ```
    - The app allows users to upload a skin lesion image and get melanoma classification results.

4. **Train the Model (Optional)**
    - Use the provided Jupyter Notebook (`melanoma-cancer-detection-no-pretrained-model.ipynb`) for model development and experimentation.
    - Note: The exact notebook has been used to train the model "classifier.keras". You may run the notebook as save your model instead of downloading.
   

## Dependencies

Key Python packages (see `requirements.txt`):

- TensorFlow
- Streamlit

## Files in the Repository

| File/Folder                               | Purpose                                    |
|-------------------------------------------|--------------------------------------------|
| `Dataset/`                                | Contains training and test images          |
| `Model/`                                  | Stores trained and converted models        |
| `app.py`                                  | Streamlit web interface                    |
| `tf_lite_conversion.py`                   | Model conversion script                    |
| `melanoma-cancer-detection-no-pretrained-model.ipynb` | Model building and experiments  |
| `requirements.txt`                        | Dependency list                            |
