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
