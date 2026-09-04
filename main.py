"""
main.py — Invisible Cloak Application
======================================
Entry point.  Handles camera setup, HSV trackbar window, the main
real-time processing loop, and all keyboard controls.

Controls
--------
  Q  — Quit
  R  — Recapture background (5-second countdown, no restart needed)
  M  — Toggle mask-preview debug window

Dependencies: Python, OpenCV, NumPy only — no deep learning.
"""

import cv2
import numpy as np

from utils import (
    warmup_camera,
    capture_background,
    create_red_mask,
    clean_mask,
    replace_background,
    check_camera_moved,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_WINDOW   = "Invisible Cloak Output"
CALIB_WINDOW    = "HSV Calibration"
MASK_WINDOW     = "Mask Preview  [M to close]"

CAM_INDEX       = 0          # Change to 1 / 2 if you have multiple cameras
FRAME_W, FRAME_H = 640, 480

# How often (in frames) to run the camera-stability check.
# We only check when the mask area is tiny so the person doesn't trigger it.
CAM_CHECK_INTERVAL = 90
CAM_MOVED_AREA_LIMIT = 500  # px² — only check stability when cloak is absent


# ─────────────────────────────────────────────────────────────────────────────
# Trackbar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _noop(_: int) -> None:
    """Empty callback required by cv2.createTrackbar."""
    pass


def setup_trackbars() -> None:
    """
    Create the HSV Calibration window with 6 trackbars.

    The window contains a tiny 1-pixel canvas — it exists only to host
    the trackbars, not to display any image content.

    Trackbar layout:
      H1 Lo / H1 Hi — hue bounds for the lower red band (H ≈ 0-10)
      H2 Lo / H2 Hi — hue bounds for the upper red band (H ≈ 170-180)
      S Min          — minimum saturation (rejects skin, beige, pastel)
      V Min          — minimum value      (rejects very dark / shadowed pixels)
    """
    cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CALIB_WINDOW, 400, 220)

    # Lower red band (hue near 0)
    cv2.createTrackbar("H1 Lo",  CALIB_WINDOW,   0,  30, _noop)
    cv2.createTrackbar("H1 Hi",  CALIB_WINDOW,  10,  30, _noop)

    # Upper red band (hue near 180)
    cv2.createTrackbar("H2 Lo",  CALIB_WINDOW, 162, 180, _noop)
    cv2.createTrackbar("H2 Hi",  CALIB_WINDOW, 180, 180, _noop)

    # Shared saturation / value thresholds
    # S Min = 120: rejects skin (~40-90) but catches bright fabric
    # V Min = 80 : rejects very dark shadows
    cv2.createTrackbar("S Min",  CALIB_WINDOW, 120, 255, _noop)
    cv2.createTrackbar("V Min",  CALIB_WINDOW,  80, 255, _noop)


def read_trackbar_params() -> dict:
    """
    Read current trackbar positions and return them as a parameter dict
    consumed by create_red_mask().

    Returns:
        dict with keys: h1_lo, h1_hi, h2_lo, h2_hi, s_lo, v_lo
    """
    return {
        "h1_lo": cv2.getTrackbarPos("H1 Lo", CALIB_WINDOW),
        "h1_hi": cv2.getTrackbarPos("H1 Hi", CALIB_WINDOW),
        "h2_lo": cv2.getTrackbarPos("H2 Lo", CALIB_WINDOW),
        "h2_hi": cv2.getTrackbarPos("H2 Hi", CALIB_WINDOW),
        "s_lo" : cv2.getTrackbarPos("S Min", CALIB_WINDOW),
        "v_lo" : cv2.getTrackbarPos("V Min", CALIB_WINDOW),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray,
             fps: float,
             mask_preview_on: bool,
             cam_moved_warning: bool,
             mask_px: int = 0) -> np.ndarray:
    """
    Draw a minimal Heads-Up Display on top of the composited output frame.

    Includes:
      - FPS counter (top-left)
      - Mask pixel count — confirms how much of the cloak is being replaced
      - Hotkey reminder strip (bottom)
      - Camera-moved warning (top-right, red text) when detected

    Args:
        frame            : Composited output frame (will NOT be mutated —
                           a copy is made internally).
        fps              : Measured frames per second.
        mask_preview_on  : Whether the mask debug window is currently visible.
        cam_moved_warning: Whether the camera-shift warning should be shown.
        mask_px          : Number of white pixels in the clean mask.

    Returns:
        A copy of the frame with HUD text rendered on it.
    """
    out   = frame.copy()
    h, w  = out.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # ── FPS (top-left) ───────────────────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(out, fps_text, (10, 28), font, 0.75,
                (0, 0, 0), 4, cv2.LINE_AA)          # shadow
    cv2.putText(out, fps_text, (10, 28), font, 0.75,
                (0, 255, 120), 2, cv2.LINE_AA)       # foreground

    # ── Mask pixel count (below FPS) ─────────────────────────────────────
    # Shows how many pixels are being replaced each frame.
    # If this reads 0 when wearing the cloak, the HSV range needs tuning.
    if mask_px > 0:
        px_text = f"Cloak: {mask_px:,} px"
        cv2.putText(out, px_text, (10, 52), font, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, px_text, (10, 52), font, 0.6,
                    (0, 200, 255), 2, cv2.LINE_AA)


    # ── Hotkey strip (bottom bar) ─────────────────────────────────────────
    bar_h = 26
    cv2.rectangle(out, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    mask_label = "M: Hide mask" if mask_preview_on else "M: Show mask"
    hint = f"  Q: Quit    R: Recapture bg    {mask_label}"
    cv2.putText(out, hint, (6, h - 8), font, 0.52,
                (180, 180, 180), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Application entry point.

    Lifecycle
    ---------
    1.  Open webcam and warm it up.
    2.  Create output + calibration windows with trackbars.
    3.  Capture averaged background (5-second countdown).
    4.  Enter the real-time processing loop:
          a. Read frame, flip for mirror view.
          b. Light Gaussian blur to denoise before colour detection.
          c. BGR → HSV conversion.
          d. Red detection via dual-range inRange (trackbar-driven params).
          e. Full morphological cleaning pipeline (open, close, fill holes,
             keep largest blob, final smooth).
          f. Soft alpha blend to replace cloak pixels with stored background.
          g. Draw HUD (FPS, hints, warnings).
          h. Display output; optionally show mask debug window.
          i. Handle keystrokes (Q / R / M).
          j. Every CAM_CHECK_INTERVAL frames (when cloak is absent) check
             for camera movement and set a warning flag.
    5.  Release resources.
    """

    # ── 1. Open webcam ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam (index {CAM_INDEX}).")
        return

    # Request a fixed resolution — many drivers honour this
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    # Keep auto-exposure on (0.75 = auto on most UVC drivers)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # Discard the first 80 frames so AGC / AWB settle before background capture
    warmup_camera(cap, num_frames=80)

    # ── 2. Create windows ─────────────────────────────────────────────────
    cv2.namedWindow(OUTPUT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(OUTPUT_WINDOW, FRAME_W, FRAME_H)

    setup_trackbars()      # creates CALIB_WINDOW with all 6 trackbars

    print("\n" + "="*50)
    print("  Invisible Cloak — Controls")
    print("="*50)
    print("  Q  — Quit")
    print("  R  — Recapture background")
    print("  M  — Toggle mask debug window")
    print("="*50 + "\n")

    # ── 3. Initial background capture ─────────────────────────────────────
    background = capture_background(
        cap,
        window_name    = OUTPUT_WINDOW,
        countdown_secs = 5,
        num_frames     = 60,
    )

    # ── 4. Real-time loop state ───────────────────────────────────────────
    mask_preview_on    = False   # toggled by 'M'
    cam_moved_warning  = False   # set by stability checker
    frame_count        = 0       # incremented every loop iteration

    # FPS measurement — use OpenCV's high-resolution tick counter
    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps       = 0.0

    # ── Main processing loop ──────────────────────────────────────────────
    while True:

        # ── a. Capture frame ──────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Frame read failed — retrying...")
            continue

        # Mirror horizontally for a natural "mirror" user experience
        frame = cv2.flip(frame, 1)

        # ── b. Denoise frame before colour analysis ───────────────────────
        # A light Gaussian blur reduces webcam sensor noise.  We keep the
        # kernel small (5×5) to preserve detail and maintain FPS.
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # ── c. BGR → HSV ──────────────────────────────────────────────────
        # HSV separates colour (Hue) from intensity (Value) which makes
        # colour thresholding robust to lighting changes.
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # ── d. Red detection with live trackbar parameters ────────────────
        params  = read_trackbar_params()
        raw_mask = create_red_mask(hsv, params)

        # ── e. Full mask cleaning pipeline ───────────────────────────────
        # Converts a noisy raw mask into one solid, hole-free blob.
        clean = clean_mask(raw_mask)

        # ── f. Soft alpha background replacement ─────────────────────────
        output = replace_background(frame, background, clean)

        # ── g. FPS measurement ────────────────────────────────────────────
        cur_tick = cv2.getTickCount()
        elapsed  = (cur_tick - prev_tick) / tick_freq
        # Exponential moving average for smooth FPS display
        fps      = 0.9 * fps + 0.1 * (1.0 / elapsed if elapsed > 0 else fps)
        prev_tick = cur_tick

        # ── h. Camera stability check (cheap, runs infrequently) ──────────
        frame_count += 1
        mask_area    = int(np.sum(clean > 0))

        if frame_count % CAM_CHECK_INTERVAL == 0 and mask_area < CAM_MOVED_AREA_LIMIT:
            # Only run when there is no cloak in frame (mask empty),
            # otherwise the person standing there would always trigger it.
            cam_moved_warning = check_camera_moved(frame, background)

        # Clear warning once the user recaptures the background
        # (background is fresh, so diff will be low again)
        if cam_moved_warning and mask_area < CAM_MOVED_AREA_LIMIT:
            if not check_camera_moved(frame, background):
                cam_moved_warning = False

        # ── i. Draw HUD on output ─────────────────────────────────────────
        output_hud = draw_hud(output, fps, mask_preview_on, cam_moved_warning,
                              mask_px=mask_area)

        # ── j. Show output window ─────────────────────────────────────────
        cv2.imshow(OUTPUT_WINDOW, output_hud)

        # ── k. Mask debug window (toggled with M) ─────────────────────────
        if mask_preview_on:
            # Show RAW mask (left) and CLEAN mask (right) side-by-side.
            # RAW  = what the HSV detector found before cleaning
            # CLEAN = the final solid blob used for compositing
            # The clean mask must be one solid white region — if it shows
            # holes or multiple blobs, lower S Min in the calibration window.
            raw_bgr   = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
            clean_bgr = cv2.cvtColor(clean,    cv2.COLOR_GRAY2BGR)

            # Label each panel
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(raw_bgr,   (0, 0), (raw_bgr.shape[1],   20), (30, 30, 30), -1)
            cv2.rectangle(clean_bgr, (0, 0), (clean_bgr.shape[1], 20), (30, 30, 30), -1)
            cv2.putText(raw_bgr,   "RAW MASK",
                        (5, 15), font, 0.55, (100, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(clean_bgr, f"CLEAN MASK  ({mask_area:,} px)",
                        (5, 15), font, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

            # Draw a thin separator line between the two panels
            separator = np.zeros((raw_bgr.shape[0], 3, 3), dtype=np.uint8)
            separator[:] = (80, 80, 80)
            panel = np.hstack([raw_bgr, separator, clean_bgr])
            cv2.imshow(MASK_WINDOW, panel)
        else:
            # Destroy the window if the user toggled it off
            try:
                cv2.destroyWindow(MASK_WINDOW)
            except cv2.error:
                pass

        # ── l. Keyboard input ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            print("[App] Quitting...")
            break

        elif key in (ord('r'), ord('R')):
            # Recapture background without restarting the program
            print("[App] Recapturing background...")
            cam_moved_warning = False
            background = capture_background(
                cap,
                window_name    = OUTPUT_WINDOW,
                countdown_secs = 5,
                num_frames     = 60,
            )

        elif key in (ord('m'), ord('M')):
            # Toggle mask preview window
            mask_preview_on = not mask_preview_on
            state = "ON" if mask_preview_on else "OFF"
            print(f"[App] Mask preview {state}")

    # ── 5. Release resources ──────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[App] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
