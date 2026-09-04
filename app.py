import streamlit as st
import cv2
import numpy as np
from utils import create_red_mask, clean_mask, replace_background

st.set_page_config(
    page_title="Invisible Cloak",
    page_icon="🧙",
    layout="wide"
)

st.title("🧙 Invisible Cloak using OpenCV")
st.write("Real-time red cloth detection and background replacement")

uploaded_bg = st.file_uploader(
    "Upload a background image",
    type=["jpg", "jpeg", "png"]
)

camera = st.camera_input("📷 Capture your current frame")

st.sidebar.header("HSV Calibration")

h1_lo = st.sidebar.slider("H1 Low", 0, 30, 0)
h1_hi = st.sidebar.slider("H1 High", 0, 30, 10)
h2_lo = st.sidebar.slider("H2 Low", 150, 180, 162)
h2_hi = st.sidebar.slider("H2 High", 150, 180, 180)
s_lo = st.sidebar.slider("S Minimum", 0, 255, 120)
v_lo = st.sidebar.slider("V Minimum", 0, 255, 80)

if uploaded_bg is not None and camera is not None:

    bg_bytes = uploaded_bg.getvalue()
    bg_array = np.frombuffer(bg_bytes, np.uint8)
    background = cv2.imdecode(bg_array, cv2.IMREAD_COLOR)

    camera_bytes = camera.getvalue()
    camera_array = np.frombuffer(camera_bytes, np.uint8)
    frame = cv2.imdecode(camera_array, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (640, 480))
    background = cv2.resize(background, (640, 480))

    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    params = {
        "h1_lo": h1_lo,
        "h1_hi": h1_hi,
        "h2_lo": h2_lo,
        "h2_hi": h2_hi,
        "s_lo": s_lo,
        "v_lo": v_lo
    }

    raw_mask = create_red_mask(hsv, params)

    clean = clean_mask(raw_mask)

    output = replace_background(
        frame,
        background,
        clean
    )

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.subheader("Invisible Cloak Result")
    st.image(output_rgb, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    with col2:
        st.subheader("Mask")
        st.image(clean, use_container_width=True)

else:
    st.info(
        "First upload a background image and then capture your current frame."
    )
