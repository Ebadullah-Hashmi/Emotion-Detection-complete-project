import streamlit as st
import cv2
import numpy as np
import os
import gdown
import av
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer

st.title("Live Emotion Detection")
st.write("Camera permission allow karein aur START par click karein.")

# --- 1. Model Loading (with Google Drive Download) ---
# st.cache_resource is liye use kiya taake model baar baar load/download na ho
@st.cache_resource
def load_emotion_model():
    model_path = 'Custom_CNN_model.keras'
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive, please wait...")
        # APNI GOOGLE DRIVE FILE ID YAHAN PASTE KAREIN:
        file_id = '1ltv5mD7xsRYNBnPMm78jbGtJtoBsKHPq' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

classifier = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- 2. Face Classifier Setup ---
# cv2.data.haarcascades use kiya hai taake XML file ka error na aaye
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 3. Live Video Frame Processor ---
def video_frame_callback(frame):
    # WebRTC se aane wale frame ko OpenCV format (numpy array) mein convert karein
    img = frame.to_ndarray(format="bgr24")

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process each face detected
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)

            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Processed frame wapis return karein live screen ke liye
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. Streamlit WebRTC Component ---
webrtc_streamer(
    key="emotion-detection", 
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False} # Sirf video chahiye, audio nahi
)
