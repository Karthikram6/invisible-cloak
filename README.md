# Invisible Cloak using OpenCV

An Invisible Cloak project built using **Python, OpenCV, NumPy, and Streamlit** that creates the illusion of invisibility by detecting a red-colored cloth and replacing it with a previously captured background in real time.

## 🚀 Live Demo

👉 [Try the Invisible Cloak Live](https://invisible-cloak.streamlit.app)

## 💻 GitHub Repository

👉 [View Source Code](https://github.com/Karthikram6/invisible-cloak)

## ✨ Features

* Real-time webcam processing
* Background capture and replacement
* Red color detection using HSV color space
* Gaussian Blur for noise reduction
* Morphological operations for mask refinement
* Noise removal and mask cleaning
* Smooth real-time invisibility effect
* Adjustable HSV calibration using Streamlit sliders
* Live browser-based webcam support

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* Streamlit
* Streamlit-WebRTC

## 🔍 How It Works

1. Start the webcam through the web application.
2. Capture the background without the user in the frame.
3. Wear a bright red-colored cloth.
4. Detect the red cloth using HSV color segmentation.
5. Refine the detected mask using image processing techniques.
6. Replace the detected red region with the captured background.
7. Display the invisible cloak effect in real time.

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/Karthikram6/invisible-cloak.git
cd invisible-cloak
Create a Virtual Environment
python -m venv venv
Activate the Virtual Environment
Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
🖥️ Run the Desktop Version
python main.py
Desktop Controls
Q — Quit the application
R — Recapture the background
M — Toggle mask preview
HSV Trackbars — Adjust red color detection
🌐 Run the Web Version
streamlit run app.py

After running, open the local URL provided by Streamlit in your browser.

📋 Requirements
Python 3.8+
Webcam
Bright red cloth
Stable lighting
Internet connection for the deployed web version
📁 Project Structure
Invisible-Cloak/
│
├── app.py
├── main.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
🎮 Web App Controls
Start Camera — Start the webcam
Capture Background — Capture the background without the red cloth
Reset Background — Reset and recapture the background
HSV Sliders — Adjust red color detection parameters
⚙️ Image Processing Pipeline
Webcam frame capture
Gaussian Blur
BGR to HSV color conversion
Red color segmentation
Morphological operations
Mask refinement
Background replacement
Real-time output display
🌍 Deployment

The web application is deployed using Streamlit Community Cloud.

Live Application

👉 Open Invisible Cloak

Source Code

👉 GitHub Repository

🚀 Future Enhancements
Support for multiple cloak colors
Improved background stabilization
Enhanced mask refinement
Better performance under changing lighting conditions
Improved real-time video processing
Mobile camera support
Better object segmentation
Automatic color calibration
📌 Notes

For the best results:

Use a bright red cloth.
Keep the camera stationary.
Capture the background before entering the frame.
Use stable and consistent lighting.
Avoid red-colored objects in the background.
📄 License

This project is created for educational and learning purposes.
