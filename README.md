# 🧙 Invisible Cloak using OpenCV

An **Invisible Cloak** project built using **Python, OpenCV, NumPy, and Streamlit** that creates the illusion of invisibility by detecting a red-colored cloth and replacing it with a previously captured background in real time.

## 🚀 Live Demo

👉 [Try the Invisible Cloak Live](https://invisible-cloak.streamlit.app)

## 💻 GitHub Repository

👉 [View Source Code](https://github.com/Karthikram6/invisible-cloak)

---

## ✨ Features

- 🎥 Real-time webcam processing
- 🖼️ Background capture and replacement
- 🔴 Red color detection using HSV color space
- 🌫️ Gaussian Blur for noise reduction
- 🧹 Morphological operations for mask refinement
- ✨ Noise removal and mask cleaning
- 🪄 Smooth real-time invisibility effect
- 🎚️ Adjustable HSV calibration using Streamlit sliders
- 🌐 Live browser-based webcam support

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Core Programming Language |
| 👁️ OpenCV | Computer Vision & Image Processing |
| 🔢 NumPy | Numerical & Array Operations |
| 🎨 Streamlit | Web Application Interface |
| 📡 Streamlit-WebRTC | Real-Time Webcam Streaming |

---

## 🔍 How It Works

### Step 1 — Start the Webcam

Start the webcam through the web application.

### Step 2 — Capture Background

Capture the background **without the user in the frame**.

### Step 3 — Wear the Red Cloth

Wear a bright **red-colored cloth** and stand in front of the camera.

### Step 4 — Detect the Cloth

The system detects the red cloth using **HSV color segmentation**.

### Step 5 — Refine the Mask

Image processing techniques are applied to remove noise and improve the detected mask.

### Step 6 — Replace the Cloth

The detected red region is replaced with the previously captured background.

### Step 7 — Display the Result

The final processed frame creates the **Invisible Cloak effect in real time**.

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Karthikram6/invisible-cloak.git
cd invisible-cloak

## 2️⃣ Create a Virtual Environment
python -m venv venv
## 3️⃣ Activate the Virtual Environment
Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate
## 4️⃣ Install Dependencies
pip install -r requirements.txt
🖥️ Run the Desktop Version

Run the original OpenCV desktop application using:

python main.py
🎮 Desktop Controls
Key	Action
Q	Quit the application
R	Recapture the background
M	Toggle mask preview
HSV Trackbars	Adjust red color detection
🌐 Run the Web Version

Run the Streamlit application using:

streamlit run app.py

After running the command, open the local URL provided by Streamlit in your browser.

🎮 Web App Controls
📷 Start Camera

Start the webcam through the browser.

🖼️ Capture Background

Capture the background before entering the frame.

🔄 Reset Background

Reset the current background and capture a new one.

🎚️ HSV Controls

Adjust the HSV parameters to improve red-color detection.

📋 Requirements
🐍 Python 3.8+
📷 Working webcam
🔴 Bright red cloth
💡 Stable lighting
🌐 Internet connection for the deployed web version
📁 Project Structure
Invisible-Cloak/
│
├── app.py
├── main.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Image Processing Pipeline
Webcam Frame
     ↓
Gaussian Blur
     ↓
BGR → HSV Conversion
     ↓
Red Color Segmentation
     ↓
Mask Creation
     ↓
Morphological Operations
     ↓
Mask Refinement
     ↓
Background Replacement
     ↓
Invisible Cloak Effect
🧠 Computer Vision Techniques
🎨 HSV Color Segmentation

HSV color space is used to identify the red-colored cloth more effectively than directly working with RGB/BGR values.

🌫️ Gaussian Blur

Gaussian Blur helps reduce small amounts of noise in the webcam frame.

🧹 Morphological Operations

Morphological operations improve the mask by removing unwanted noise and filling small gaps.

🖼️ Background Replacement

The detected red-cloth region is replaced with the corresponding region from the captured background.

🌍 Deployment

The web application is deployed using Streamlit Community Cloud.

🚀 Live Application

👉 Open Invisible Cloak

💻 Source Code

👉 GitHub Repository

🚀 Future Enhancements
🎨 Support for multiple cloak colors
🖼️ Improved background stabilization
🧹 Enhanced mask refinement
💡 Better performance under changing lighting conditions
⚡ Improved real-time video processing
📱 Mobile camera support
🧠 Better object segmentation
🎯 Automatic color calibration
📌 Notes

For the best results:

🔴 Use a bright red cloth.
📷 Keep the camera stationary.
🖼️ Capture the background before entering the frame.
💡 Use stable and consistent lighting.
🚫 Avoid red-colored objects in the background.
⚠️ Limitations
Performance depends on lighting conditions.
The camera should remain relatively stationary.
Other red-colored objects may also be detected.
Very dark or poorly lit environments may reduce accuracy.
Webcam processing performance depends on the device and browser.
🎯 Use Cases
🎓 Computer Vision learning projects
🧪 Image Processing experiments
🎨 Interactive computer vision applications
🏫 College mini-projects
💻 Python and OpenCV demonstrations
📄 License

This project is created for educational and learning purposes.
