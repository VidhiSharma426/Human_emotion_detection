import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import streamlit as st


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
        # Webcam logic
        run_video = st.checkbox("Run Webcam")
        st.info("Check the box to start webcam detection.")

        if run_video:
            # Start video capture
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while run_video:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam not detected. Please check your device.")
                    break

                # Detect emotion and annotate frame
                annotated_frame, detected_emotions = detect_emotion(frame, face_cascade, model, emotion_list)

                # Display the video frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

            # Release the video capture when done
            cap.release()

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
