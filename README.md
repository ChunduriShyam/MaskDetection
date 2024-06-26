# Face Detection Project

This project implements a face detection model using convolutional neural networks (CNN). The repository contains Jupyter notebooks for training models, scripts for real-time face detection via camera, and evaluation notebooks to assess model performance of models .

## Project Structure

- `notebooks/` - Jupyter notebooks for training and evaluation
  - `CNN_pretrained.ipynb` - Notebook to train the model using a pre-trained CNN.
  - `CNN_train.ipynb` - Notebook to train the model from scratch.
  - `model_evaluations.ipynb` - Notebook for evaluating and comparing the performance of the models.
- `scripts/` - Python scripts for testing camera functionality and running the detection model.
  - `cameratest.py` - Script to test if the camera works correctly.
  - `model.py` - Script that runs the best-trained model to detect faces in video feeds.
- `test/` - Contains the test dataset used in model evaluation.
- `models/` - Saved model files and weights.
- `requirements.txt` - Required Python libraries for the project.
- `app.py` - A python main script to run the model.
- `model_download.py` - A python script to download trained CNN models from google drive to models/ folder.

## Setup and Installation

Before running the project, make sure you have Python 3.10 installed on your system. 

1. **Install Required Dependencies**:
   - Open your terminal or command prompt.
   - Navigate to the project's root directory.
   - Run the following command to install the required Python packages:

     "pip install -r requirements.txt"

2. **Download the trained models**:
   - Run the `model_download.py` script to download the CNN basic and pretrained models to the local directory,
     after the execution of this script the models will be downloaded to the /models directory.

     If you want to skip this please use below link to download the models manually to the local directory, due to the size of
     these models we have used google drive to store them.

     Link : "https://drive.google.com/drive/folders/1awzQfYpkqx8KgRiuHJSYgHgNYAZuo76y?usp=sharing"

3. **Running the Application**:
   - Run the `app.py` script to start the face detection application.