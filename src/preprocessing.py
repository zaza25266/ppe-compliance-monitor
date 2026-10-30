# Purpose: Load, validate, resize, and convert images and video frames.

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image(image_file):
    # Load uploaded image and convert it to RGB NumPy array.
    try:
        image = Image.open(image_file).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Invalid or unsupported image.") from error


def validate_frame(frame):
    # Ensure the input is a valid three-channel image.
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a NumPy array.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have three color channels.")

    if frame.size == 0:
        raise ValueError("Frame is empty.")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def bgr_to_rgb(frame):
    # Convert OpenCV BGR frame to RGB.
    return cv2.cvtColor(validate_frame(frame), cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame):
    # Convert RGB frame to OpenCV BGR.
    return cv2.cvtColor(validate_frame(frame), cv2.COLOR_RGB2BGR)


def resize_frame(frame, max_width=1280, max_height=720):
    # Resize while preserving aspect ratio.
    frame = validate_frame(frame)

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA
        )

    return frame
