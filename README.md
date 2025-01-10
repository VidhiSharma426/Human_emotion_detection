Emotion Detector App

Overview

The Emotion Detector App is a web application designed to detect and classify emotions from facial expressions using a deep learning model. The app supports both real-time webcam detection and image uploads for emotion analysis.

Key Features:

Webcam Mode: Detects emotions in real-time using your webcam.

Image Upload Mode: Analyze emotions from uploaded images.

AI-Powered: Uses a pre-trained convolutional neural network (CNN) for emotion classification.

User-Friendly Interface: Built with Streamlit for simplicity and ease of use.

Supported Emotions:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

Installation

Prerequisites:

Python 3.8+

System dependencies:

libgl1-mesa-glx (for OpenCV GUI features)

Steps:

Clone the repository:

git clone https://github.com/your-repo/emotion-detector-app.git
cd emotion-detector-app

Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate

Install the required Python packages:

pip install -r requirements.txt

(Optional) Install system dependencies if needed:

sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

Usage

Running the App:

Start the Streamlit app:

streamlit run app1.py

Open the app in your browser. By default, it will be available at http://localhost:8501.

Choose an option from the sidebar:

Webcam Mode: Detect emotions in real-time using your webcam.

Upload Image Mode: Upload an image and detect emotions from the faces in the image.

File Structure

.
├── app1.py                 # Main application script
├── model_a.json            # Pre-trained CNN model architecture
├── model_weights.h5        # Pre-trained model weights
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies for Streamlit Cloud
└── README.md               # Documentation

Deployment

Streamlit Cloud

Ensure your repository includes requirements.txt and packages.txt.

Deploy the repository directly to Streamlit Cloud.

Docker

Build the Docker image:

docker build -t emotion-detector-app .

Run the Docker container:

docker run -p 8501:8501 emotion-detector-app

Troubleshooting

Common Errors:

ImportError: libGL.so.1: cannot open shared object file

Install the missing system dependency:

sudo apt-get install -y libgl1-mesa-glx

Model Loading Issues:

Ensure model_a.json and model_weights.h5 are in the same directory as app1.py.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

Contact

Developed by Vidhi Sharma. For questions or feedback, please contact: vidhi2821426@gmail.com.
