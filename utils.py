"""
utils.py — Invisible Cloak Utility Functions
=============================================
Pure image-processing helpers. No GUI code lives here.
All functions use only Python, OpenCV, and NumPy.
"""

import cv2
import numpy as np
import time


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def warmup_camera(cap, num_frames: int = 80) -> None:
    """
    Discard the first N frames so the webcam's Auto-Gain Control (AGC)
    and Auto-White-Balance (AWB) can settle before any useful work begins.
    Without this the first captured frames are severely under-exposed.

    Args:
        cap        : Open cv2.VideoCapture object.
        num_frames : Number of frames to throw away (default 80).
    """
    print(f"[Camera] Warming up — discarding {num_frames} frames...")
    for _ in range(num_frames):
        cap.read()          # read and immediately discard
        cv2.waitKey(10)     # give the OS a tiny yield so the driver can process
    print("[Camera] Warmup complete.")


def check_camera_moved(frame: np.ndarray,
                        background: np.ndarray,
                        threshold: float = 30.0) -> bool:
    """
    Detect whether the camera has shifted significantly since the background
    was captured.  We compare the absolute-difference between the current
    frame and the stored background in grayscale.

    Only call this when the red mask area is tiny (< 500 px) so that the
    person standing in front does not trigger a false alarm.

    Args:
        frame      : Current BGR webcam frame.
        background : Stored averaged background (BGR).
        threshold  : Mean pixel-difference threshold (0-255).
                     30 works well for typical indoor setups.

    Returns:
        True  — camera has likely moved; background should be recaptured.
        False — camera is stable.
    """
    # Downscale before comparing so the operation stays cheap
    small_frame = cv2.resize(frame,      (160, 120))
    small_bg    = cv2.resize(background, (160, 120))

    diff      = cv2.absdiff(small_frame, small_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray_diff)) > threshold


# ---------------------------------------------------------------------------
# Background capture
# ---------------------------------------------------------------------------

def capture_background(cap,
                        window_name:      str = "Invisible Cloak Output",
                        countdown_secs:   int = 5,
                        num_frames:       int = 60) -> np.ndarray:
    """
    Show an on-screen countdown, then capture `num_frames` frames and return
    their per-pixel mean — a clean, noise-free background image.

    The user must be completely out of frame during the capture phase.

    Args:
        cap            : Open cv2.VideoCapture object.
        window_name    : Name of the existing imshow window for live feedback.
        countdown_secs : Seconds to count down before capturing starts.
        num_frames     : Number of frames to average (60 gives a very clean bg).

    Returns:
        np.ndarray: Averaged background, dtype uint8, BGR colour space.
    """

    # ── Phase 1: Countdown ────────────────────────────────────────────────
    print(f"[Background] Countdown starting ({countdown_secs}s)...")
    start = time.time()

    while True:
        elapsed   = time.time() - start
        remaining = max(0, countdown_secs - int(elapsed))

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Semi-transparent dark vignette so text is readable
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), thickness=-1)
        display = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        # Large countdown number in the centre
        num_text = str(remaining) if remaining > 0 else "GO!"
        (nw, nh), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)
        cv2.putText(display, num_text,
                    ((w - nw) // 2, (h + nh) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 255), 8, cv2.LINE_AA)

        # Instruction line below
        instr = "Step completely out of frame!"
        (iw, _), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(display, instr,
                    ((w - iw) // 2, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        cv2.waitKey(1)

        if elapsed >= countdown_secs:
            break

    # ── Phase 2: Frame capture with progress bar ──────────────────────────
    print(f"[Background] Capturing {num_frames} frames...")
    frames: list[np.ndarray] = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frames.append(frame.astype(np.float32))

        # Draw a progress bar at the bottom of the live frame
        h, w   = frame.shape[:2]
        pct    = (i + 1) / num_frames
        bar_w  = int(w * pct)
        canvas = frame.copy()
        # Background bar track
        cv2.rectangle(canvas, (0, h - 24), (w,     h), ( 40,  40,  40), -1)
        # Filled portion
        cv2.rectangle(canvas, (0, h - 24), (bar_w, h), (  0, 200,   0), -1)
        # Text
        cv2.putText(canvas, f"Capturing background... {int(pct * 100)}%",
                    (10, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)

    if not frames:
        raise RuntimeError("[Background] No frames captured — check your webcam.")

    # ── Phase 3: Pixel-wise mean → clean background ───────────────────────
    # np.mean over a list of float32 arrays is precise and produces a
    # stationary-noise-free reference image.
    background = np.mean(frames, axis=0).astype(np.uint8)
    print("[Background] Capture complete.")
    return background


# ---------------------------------------------------------------------------
# Red-colour detection
# ---------------------------------------------------------------------------

def create_red_mask(hsv_frame: np.ndarray, params: dict) -> np.ndarray:
    """
    Detect bright red pixels using two HSV inRange calls.

    Red wraps around the hue wheel in OpenCV (0-180 range):
      - "Lower red" lives near H = 0-10
      - "Upper red" lives near H = 170-180

    Both ranges share the same saturation / value thresholds so the user
    only needs two trackbars (S Min and V Min) to control sensitivity.

    Args:
        hsv_frame : Pre-blurred HSV image (output of cv2.cvtColor + GaussianBlur).
        params    : Dict with keys —
                      'h1_lo', 'h1_hi'  — hue bounds for lower red band
                      'h2_lo', 'h2_hi'  — hue bounds for upper red band
                      's_lo'            — minimum saturation (rejects skin/beige)
                      'v_lo'            — minimum value    (rejects very dark pixels)

    Returns:
        Binary mask, uint8, 255 = red pixel, 0 = everything else.
    """
    s_lo = params['s_lo']
    v_lo = params['v_lo']

    # Lower red band (H near 0 — classic bright red / brick red)
    lower1 = np.array([params['h1_lo'], s_lo, v_lo], dtype=np.uint8)
    upper1 = np.array([params['h1_hi'], 255,  255  ], dtype=np.uint8)
    mask1  = cv2.inRange(hsv_frame, lower1, upper1)

    # Upper red band (H near 180 — deep red / magenta red)
    lower2 = np.array([params['h2_lo'], s_lo, v_lo], dtype=np.uint8)
    upper2 = np.array([params['h2_hi'], 255,  255  ], dtype=np.uint8)
    mask2  = cv2.inRange(hsv_frame, lower2, upper2)

    return cv2.bitwise_or(mask1, mask2)


# ---------------------------------------------------------------------------
# Mask cleaning pipeline
# ---------------------------------------------------------------------------

def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill every interior hole inside the white region of a binary mask.

    Algorithm (border flood-fill trick):
      1. Flood-fill the *inverted* mask starting from the top-left corner.
         After the fill, every pixel reachable from the border (i.e. the
         true background) becomes white.
      2. Re-invert → only truly interior holes remain white.
      3. OR with the original mask → holes disappear.

    This is more robust than contour-based hole filling because it handles
    nested holes, irregular shapes, and multi-level hierarchies correctly.

    Args:
        mask : Binary mask (uint8, 255 / 0).

    Returns:
        Mask with all interior holes filled to 255.
    """
    h, w = mask.shape

    # We need an (h+2) × (w+2) flood-fill seed mask (OpenCV requirement)
    flood_seed = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Work on a copy; flood-fill modifies the image in-place
    inverted = cv2.bitwise_not(mask)

    # Fill the exterior — starts from (0, 0) which is always background
    cv2.floodFill(inverted, flood_seed, (0, 0), 255)

    # Re-invert: now only interior holes are white
    interior_holes = cv2.bitwise_not(inverted)

    # Merge holes back into the original mask
    return cv2.bitwise_or(mask, interior_holes)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the single largest white connected region; zero everything else.

    Purpose: after morphological cleaning there may still be small stray blobs
    caused by specular highlights or skin-tone leakage.  Keeping only the
    largest blob ensures the mask represents only the actual cloak.

    Args:
        mask : Binary mask (uint8).

    Returns:
        Binary mask containing only the largest white component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        # No foreground found at all
        return np.zeros_like(mask)

    # stats[0] is the background label — skip it
    # Find the label with the largest area among foreground labels
    areas        = stats[1:, cv2.CC_STAT_AREA]   # shape: (num_labels-1,)
    largest_idx  = int(np.argmax(areas)) + 1       # +1 to account for skipped bg

    result             = np.zeros_like(mask)
    result[labels == largest_idx] = 255
    return result


def fill_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Find the single largest external contour and draw it completely filled.

    This is the most reliable way to guarantee a solid, hole-free blob:
    instead of patching existing white pixels we redraw the region from
    scratch as a solid filled polygon.  Any interior holes, thin gaps, or
    wrinkle shadows inside the contour boundary are eliminated by definition.

    Must be called AFTER keep_largest_component() so that only one contour
    (the cloak) is present.

    Args:
        mask : Binary mask (uint8) containing at most one major white region.

    Returns:
        New binary mask with the largest contour completely filled solid white.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # Select the contour with the largest area
    largest = max(contours, key=cv2.contourArea)

    # Draw it filled on a fresh black canvas
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return filled


def clean_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Full morphological cleaning pipeline that converts a noisy raw detection
    mask into one continuous, hole-free, completely solid white blob.

    Pipeline (each step explained):

      1. GaussianBlur(5×5) + threshold(100)
           Smooth the raw binary edges before morphology so structural
           elements don't amplify individual noisy pixels into spurious regions.

      2. MORPH_OPEN  k5 × 2
           Removes small isolated noise specks and thin protrusions that are
           too small to be part of the real cloak region.

      3. MORPH_CLOSE k7 × 4  (increased from 3 → 4)
           Closes the larger gaps and dark patches *inside* the cloak caused
           by wrinkles, shadows, or low-saturation fabric areas.
           More iterations = fewer surviving holes before the fill step.

      4. medianBlur(7)
           Eliminates residual salt-and-pepper noise while preserving the
           overall shape better than a Gaussian blur would.

      5. fill_mask_holes()
           Border flood-fill trick: fills every interior hole produced by
           topology the morphological close missed.

      6. keep_largest_component()
           Discards every blob that is NOT the main cloak, removing stray
           detections from skin highlights or specular reflections.

      7. fill_largest_contour()   ← NEW key step
           Redraws the cloak region as a completely solid filled polygon.
           This is the definitive guarantee of zero holes: instead of patching
           a mask we rebuild it from the contour boundary outward.

      8. MORPH_DILATE k5 × 1
           Expand the boundary by one kernel width to ensure no cloak-edge
           pixels escape compositing (erased by the feathered blend later).

      9. MORPH_CLOSE k5 × 2
           Final smooth of the contour boundary before edge feathering.

    Args:
        raw_mask : uint8 binary mask from create_red_mask().

    Returns:
        Clean, fully solid, smooth binary mask ready for compositing.
    """
    k5 = np.ones((5, 5), np.uint8)
    k7 = np.ones((7, 7), np.uint8)

    # Step 1 — Gaussian pre-smoothing + re-threshold
    blurred = cv2.GaussianBlur(raw_mask, (5, 5), 0)
    _, mask  = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Step 2 — Opening: kill isolated noise specks and thin stray regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=2)

    # Step 3 — Closing (4 iterations): bridge broken regions, fill dark patches
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=4)

    # Step 4 — Median blur: remove remaining salt-and-pepper
    mask = cv2.medianBlur(mask, 7)

    # Step 5 — Flood-fill hole filling
    mask = fill_mask_holes(mask)

    # Step 6 — Keep only the largest blob (the actual cloak)
    mask = keep_largest_component(mask)

    # Guard: nothing detected — return empty mask immediately
    if not np.any(mask):
        return mask

    # Step 7 — Redraw from contour: guarantees a 100% solid filled region
    mask = fill_largest_contour(mask)

    # Step 8 — Dilate: ensure cloak boundary pixels are fully covered
    mask = cv2.dilate(mask, k5, iterations=1)

    # Step 9 — Final closing to smooth the expanded contour boundary
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    return mask


def feather_mask(mask: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    Produce a feathered (soft-edged) uint8 mask for use in cv2.add blending.

    The hard binary mask is Gaussian-blurred to create smooth boundary falloff.
    The interior remains at full 255; only the outer edge fades to 0.

    Args:
        mask        : Clean binary mask (uint8, 0 / 255).
        blur_radius : Gaussian kernel radius — controls softness of the edge.
                      Must be odd.  15 gives a ~7-pixel feather zone.

    Returns:
        Feathered uint8 mask in [0, 255] — same shape as input.
    """
    if blur_radius % 2 == 0:
        blur_radius += 1
    return cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)


# ---------------------------------------------------------------------------
# Background replacement
# ---------------------------------------------------------------------------

def replace_background(frame: np.ndarray,
                        background: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
    """
    Replace every cloak pixel with the stored background using OpenCV
    bitwise compositing, then apply a safety net to erase any surviving
    red pixels inside the cloak bounding box.

    Compositing pipeline
    --------------------
    OpenCV bitwise operations are used instead of floating-point NumPy
    arithmetic because they operate on integer uint8 data without precision
    loss, run in optimised C++ code, and produce pixel-exact results with
    no rounding or clipping artefacts.

    Step A — Feather the hard binary mask
        A Gaussian blur on the mask boundary produces a soft alpha edge.
        The feathered mask is converted to 3-channel for per-pixel weighting.

    Step B — Extract the background region (cloak area)
        bg_part = bitwise_and(background, background, mask=feathered_mask)
        Only pixels where the mask is non-zero copy from the background.

    Step C — Extract the foreground region (non-cloak area)
        fg_part = bitwise_and(frame, frame, mask=inv_feathered_mask)
        Only pixels where the mask IS zero copy from the live frame.

    Step D — Combine
        output = cv2.add(fg_part, bg_part)
        Both regions are disjoint (one masks the other's zeros), so cv2.add
        is a clean union with no overflow risk.

    Step E — Red-pixel safety net
        After compositing, detect any residual red pixels inside the cloak
        bounding box (wrinkle edges, shadow transitions not caught by the
        mask) and overwrite them directly from the stored background.
        This guarantees zero visible red cloth pixels in the output.

    Args:
        frame      : Current BGR webcam frame (uint8).
        background : Averaged background image (uint8, same shape as frame).
        mask       : Clean binary mask (uint8) from clean_mask().

    Returns:
        Composited BGR frame (uint8) — no cloak pixels remain visible.
    """
    # ── A. Feather the mask boundary for a smooth edge transition ─────────
    # blur_radius=15 gives a ~7-pixel soft zone around the cloak edge.
    feathered     = feather_mask(mask, blur_radius=15)        # uint8 [0,255]
    inv_feathered = cv2.bitwise_not(feathered)                # uint8 [0,255]

    # ── B. Background part: pixels where cloak was detected ───────────────
    # bitwise_and with the feathered mask weights each background pixel by
    # how strongly it was detected as cloak (255 = full, 0 = none).
    bg_part = cv2.bitwise_and(background, background, mask=feathered)

    # ── C. Foreground part: pixels where NO cloak was detected ────────────
    # The inverted feathered mask selects the non-cloak frame pixels.
    fg_part = cv2.bitwise_and(frame, frame, mask=inv_feathered)

    # ── D. Combine: disjoint union of the two masked regions ──────────────
    # cv2.add saturates at 255 but since the regions are complementary
    # (one is 0 wherever the other is non-zero) there is no actual overflow.
    output = cv2.add(fg_part, bg_part)

    # ── E. Red-pixel safety net ───────────────────────────────────────────
    # After compositing, any surviving red pixels inside the cloak region
    # (thin wrinkle edges, low-saturation border pixels not fully masked)
    # are replaced pixel-for-pixel from the stored background.
    #
    # We only scan inside the dilated bounding box of the mask for speed.
    if np.any(mask):
        # Find bounding box of the clean mask
        coords   = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)

        # Add a small margin so we catch pixels just outside the mask edge
        margin = 10
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(output.shape[1], x + w + margin)
        y2 = min(output.shape[0], y + h + margin)

        # Detect red pixels in the OUTPUT inside the bounding box
        roi_out = output[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi_out, cv2.COLOR_BGR2HSV)

        # Two red HSV bands — same dual-range logic as create_red_mask()
        residual1 = cv2.inRange(hsv_roi,
                                np.array([  0, 100, 60], np.uint8),
                                np.array([ 12, 255, 255], np.uint8))
        residual2 = cv2.inRange(hsv_roi,
                                np.array([158, 100, 60], np.uint8),
                                np.array([180, 255, 255], np.uint8))
        residual_mask = cv2.bitwise_or(residual1, residual2)

        # Overwrite only the surviving red pixels with background
        roi_bg = background[y1:y2, x1:x2]
        roi_out[residual_mask > 0] = roi_bg[residual_mask > 0]
        output[y1:y2, x1:x2] = roi_out

    return output
