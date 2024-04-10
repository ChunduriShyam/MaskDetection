import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model_path = r'models/best_cnnbasic_model'
model = load_model(model_path)

# Initialize video capture (0 for the default webcam, or 'path_to_video' for using a file)
cap = cv2.VideoCapture(0)

# Load a pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face found
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        # Resize the face ROI to the target size required by the model
        resized_face = cv2.resize(face_roi, (320, 320))
        # Preprocess the face ROI
        face_array = img_to_array(resized_face) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Make prediction
        prediction = model.predict(face_array)
        # Interpret the prediction
        label = 'No Mask' if prediction < 0.5 else 'With Mask'

        # Draw the face bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
