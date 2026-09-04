# Invisible Cloak using OpenCV

An Invisible Cloak project built using **Python, OpenCV, and NumPy** that creates the illusion of invisibility by replacing a red-colored cloth with a previously captured background in real time.

## Features

* Real-time webcam processing
* Background capture and replacement
* Red color detection using HSV color space
* Gaussian Blur for noise reduction
* Morphological operations for mask refinement
* Noise removal and mask cleaning
* Smooth real-time invisibility effect
* Adjustable HSV calibration using trackbars

## Tech Stack

* Python
* OpenCV
* NumPy

## How It Works

1. Capture the background without the user in the frame.
2. Detect the red-colored cloth using HSV color segmentation.
3. Refine the detected mask using image processing techniques.
4. Replace the detected red region with the captured background.
5. Display the final invisible cloak effect in real time.

## Installation

### Clone the Repository

```bash
git clone https://github.com/Karthikram6/invisible-cloak.git
cd invisible-cloak
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python main.py
```

## Requirements

* Python 3.8+
* Webcam
* Bright red cloth
* Stable lighting

## Project Structure

```text
Invisible-Cloak/
│── main.py
│── utils.py
│── requirements.txt
│── README.md
```

## Controls

* **Q** — Quit
* **R** — Recapture background
* **M** — Toggle mask preview

## Future Enhancements

* Support for multiple cloak colors
* Improved background stabilization
* Enhanced mask refinement
* Better performance under changing lighting conditions

## License

This project is created for educational and learning purposes.
