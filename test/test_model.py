from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


# Specify the path to your image
image_path = r"C:\Users\shyam\Pictures\Camera Roll\WIN_20240409_19_28_26_Pro.jpg"


# Load the pre-trained model
model_path = r'S:\ShyamKrishna_Chunduri\UofM\UofMSem4\IntroToAI\project\mask_detection_app\models\best_cnnbasic_model'
model = load_model(model_path)

# Open and resize the image
image = Image.open(image_path)
image_resized = image.resize((320, 320))

# Convert the PIL image to a numpy array and normalize it
image_array = np.array(image_resized) / 255.0
face_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(face_array)
# Interpret the prediction
label = 'No Mask' if prediction < 0.5 else 'With Mask'

print(label)