import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Invisible Cloak",
    page_icon="🧙",
    layout="wide"
)

st.title("🧙 Invisible Cloak")
st.write("Wear a red or black cloth and make it disappear!")

class CloakProcessor(VideoProcessorBase):

    def __init__(self):
        self.background = None
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.latest_frame = img.copy()

        if self.background is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ==========================
        # RED COLOR DETECTION
        # ==========================

        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 100, 70])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )

        red_mask2 = cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )

        red_mask = cv2.bitwise_or(
            red_mask1,
            red_mask2
        )

        # ==========================
        # BLACK COLOR DETECTION
        # ==========================

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])

        black_mask = cv2.inRange(
            hsv,
            lower_black,
            upper_black
        )

        # ==========================
        # RED + BLACK MASK
        # ==========================

        mask = cv2.bitwise_or(
            red_mask,
            black_mask
        )

        # ==========================
        # CLEAN MASK
        # ==========================

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.GaussianBlur(
            mask,
            (7, 7),
            0
        )

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

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        # Smooth mask
        mask = cv2.GaussianBlur(
            mask,
            (9, 9),
            0
        )

        # ==========================
        # BACKGROUND REPLACEMENT
        # ==========================

        alpha = mask.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        result = (
            img * (1 - alpha)
            + self.background * alpha
        )

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return av.VideoFrame.from_ndarray(
            result,
            format="bgr24"
        )

    def capture_background(self):
        if self.latest_frame is not None:
            self.background = self.latest_frame.copy()

    def reset_background(self):
        self.background = None


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

st.sidebar.header("🎮 Controls")

ctx = webrtc_streamer(
    key="invisible-cloak",
    video_processor_factory=CloakProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

if ctx.video_processor:

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "📸 Capture Background",
        use_container_width=True
    ):
        ctx.video_processor.capture_background()
        st.sidebar.success(
            "Background captured!"
        )

    if st.sidebar.button(
        "🔄 Reset Background",
        use_container_width=True
    ):
        ctx.video_processor.reset_background()
        st.sidebar.info(
            "Background reset."
        )

st.markdown("---")

st.info(
    "💡 First start the camera, then capture the background "
    "without the cloak. After that, wear a RED or BLACK cloth."
)

st.markdown(
    """
    ### 🪄 How to Use

    1. 📷 Start the camera.
    2. 🖼️ Stand away from the camera.
    3. 📸 Click **Capture Background**.
    4. 🔴 Wear a **red cloth** OR ⚫ **black cloth**.
    5. 🧙 Move in front of the camera.
    6. ✨ Watch the cloak disappear!
    """
)

st.markdown("---")

st.caption(
    "🧙 Invisible Cloak using Python, OpenCV, NumPy & Streamlit"
)
