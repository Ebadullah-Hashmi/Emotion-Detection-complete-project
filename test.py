# Import necessary libraries
import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import os
import gdown

st.title("Emotion Detection App")
st.write("Turn on your camera to detect emotions in real-time.")

# --- 1. Load the Haar Cascade Classifier ---
# Using OpenCV's built-in path so it works perfectly on Streamlit Cloud
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(cascade_path)

# --- 2. Load the Emotion Model from Google Drive ---
model_path = 'Custom_CNN_model.keras'

# @st.cache_resource is used so the model loads only once and doesn't slow down the app
@st.cache_resource 
def load_emotion_model():
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive... Please wait.")

        file_id = '1ltv5mD7xsRYNBnPMm78jbGtJtoBsKHPq' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully!")
    
    return load_model(model_path)

classifier = load_emotion_model()

# Define the list of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- 3. Streamlit Camera Input ---
# This creates a web-based camera view instead of cv2.VideoCapture(0)
img_file_buffer = st.camera_input("Take a picture to detect emotion")

if img_file_buffer is not None:
    # Read image from Streamlit camera buffer
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)

    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract the region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        # Resize the ROI to 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Proceed if the ROI is not empty
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # Normalize pixel values
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Predict the emotion
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10) # Adjusted position slightly above the box

            # Display the predicted emotion label on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the final image with boxes and labels
    # We must convert BGR to RGB because Streamlit expects RGB colors
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
