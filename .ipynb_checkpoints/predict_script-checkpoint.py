
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load and prepare the image in the right format
def load_and_prepare_image(image_path, image_size=(320, 320)):
    # Load the image
    image = load_img(image_path, target_size=image_size)
    # Convert the image to a numpy array
    image_array = img_to_array(image)
    # Scale the image
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to load the model and make a prediction
def predict_with_model(image_path, model_path):
    # Load the saved model
    model = load_model(model_path)
    # Prepare the image
    prepared_image = load_and_prepare_image(image_path)
    # Make a prediction
    predictions = model.predict(prepared_image)
    # Convert probabilities to class index
    predicted_class_index = np.argmax(predictions, axis=1)
    # Define class names based on your dataset
    class_names = ['No Mask', 'Mask Incorrect', 'Mask Correct']
    # Return the predicted class name
    return class_names[predicted_class_index[0]]

# Replace 'your_image_path.jpg' with the path to the image you want to use for prediction
# Replace 'your_model_path.h5' with the path to your saved model
image_path = r'S:\ShyamKrishna_Chunduri\UofM\UofMSem4\IntroToAI\project\mask_detection_app\Test\Shyam_correct_mask.jpg'
model_path = r'S:\ShyamKrishna_Chunduri\UofM\UofMSem4\IntroToAI\project\mask_detection_app\best_cnnbasic_model.keras'


# Predict the class of the image
predicted_class = predict_with_model(image_path, model_path)

# Print out the result
print("Predicted class:", predicted_class)

# Display the preprocessed image
plt.imshow(image_array_scaled)
plt.title(f"Preprocessed Image - Predicted as {predicted_class}")
plt.axis('off') # Hide the axis
plt.show()