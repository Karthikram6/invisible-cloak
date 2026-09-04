import streamlit as st
import cv2
import numpy as np
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Invisible Cloak",
    page_icon="🧙",
    layout="wide"
)

st.title("🧙 Invisible Cloak using OpenCV")
st.write("Wear a red cloth and make it disappear in real time!")

class CloakProcessor(VideoProcessorBase):

    def __init__(self):
        self.background = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.latest_frame = img.copy()
            background = None if self.background is None else self.background.copy()

        if background is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img = cv2.resize(img, (640, 480))
        background = cv2.resize(background, (640, 480))

        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 80])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([162, 120, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 + mask2

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        alpha = mask.astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]

        result = (
            img * (1 - alpha) +
            background * alpha
        ).astype(np.uint8)

        return av.VideoFrame.from_ndarray(
            result,
            format="bgr24"
        )


st.sidebar.header("Controls")

ctx = webrtc_streamer(
    key="invisible-cloak",
    video_processor_factory=CloakProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

if ctx.video_processor:

    if st.sidebar.button("📸 Capture Background"):

        with ctx.video_processor.lock:

            if ctx.video_processor.latest_frame is not None:

                ctx.video_processor.background = (
                    ctx.video_processor.latest_frame.copy()
                )

                st.success(
                    "Background captured! Now wear the red cloth."
                )

    if st.sidebar.button("🔄 Reset Background"):

        with ctx.video_processor.lock:
            ctx.video_processor.background = None

        st.info("Background reset.")

st.sidebar.markdown("---")

st.sidebar.write("### How to use")

st.sidebar.write(
    "1. Start the camera"
)

st.sidebar.write(
    "2. Stay out of the frame"
)

st.sidebar.write(
    "3. Click Capture Background"
)

st.sidebar.write(
    "4. Wear the red cloth"
)

st.sidebar.write(
    "5. Come in front of the camera"
)

st.sidebar.write(
    "6. The red cloth becomes invisible!"
)
