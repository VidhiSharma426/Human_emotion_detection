import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# About Section
def about_section():
    st.subheader("About Emotion Detector App")
    st.info(
        """
        **Emotion Detector App**  
        This application uses a pre-trained deep learning model to detect and classify emotions from facial expressions in real-time or uploaded images.  

        ### Key Features:
        - **Webcam Mode**: Detect emotions in real-time using your webcam.
        - **Image Upload Mode**: Analyze emotions from uploaded images.
        - **Advanced AI**: Powered by a convolutional neural network (CNN) trained on facial emotion datasets.
        - **Easy-to-Use**: Designed with a simple and intuitive interface.

        ### Supported Emotions:
        - Angry  
        - Disgust  
        - Fear  
        - Happy  
        - Neutral  
        - Sad  
        - Surprise  


        Developed by Vidhi 
        For questions or feedback, please contact vidhi2821426@gmail.com
        """
    )
# Load the model
@st.cache_resource
def load_model():
    with open("model_a.json", "r") as file:
        loaded_model_json = file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_weights.weights.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Detect emotion from a given frame
def detect_emotion(frame, face_cascade, model, emotion_list):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    emotions_detected = []
    for (x, y, w, h) in faces:
        face = gray_image[y:y+h, x:x+w]
        roi = cv2.resize(face, (48, 48))
        pred = emotion_list[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        emotions_detected.append(pred)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, emotions_detected
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.model = load_model()
        self.emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(face, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis]

            pred = self.emotion_list[np.argmax(self.model.predict(roi))]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, pred, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)

        return img
# Main Streamlit App
def main():
    st.title("Emotion Detector")
    st.sidebar.title("Choose an Option")

    # Load resources
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = load_model()
    emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Options: Webcam or Upload Image
    option = st.sidebar.radio("Select mode:", ("Webcam", "Upload Image"))

    if option == "Webcam":
        st.info("Click 'Start' to use your webcam")
    
        webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=EmotionDetector
        )

    elif option == "Upload Image":
        # Image upload logic
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Detect emotion and annotate image
            annotated_image, detected_emotions = detect_emotion(opencv_image, face_cascade, model, emotion_list)

            # Display the result
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
            st.write("Detected Emotions: ", ", ".join(detected_emotions))

if __name__ == "__main__":
    main()
    if st.sidebar.button("About"):
        about_section()
