
# src/preprocessing.py
# Purpose: Validate uploaded images and standardize images/video frames
# before sending them to the YOLO object detection models.

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image(image_file):
    """Load an uploaded image and return it as an RGB NumPy array."""

    # Read the uploaded file and ensure it is a valid image.
    try:
        image = Image.open(image_file)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    # Convert grayscale, RGBA, and other supported modes to RGB.
    image = image.convert("RGB")

    # Convert the PIL image into a NumPy array with shape (height, width, 3).
    return np.asarray(image, dtype=np.uint8)


def validate_frame(frame):
    """Validate an image or video frame represented as a NumPy array."""

    # Ensure the input is a NumPy array.
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array.")

    # Accept only color images with three channels.
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have three color channels.")

    # Reject empty frames.
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Frame cannot be empty.")

    # Convert unexpected numeric types and values to uint8 image pixels.
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def bgr_to_rgb(frame):
    """Convert an OpenCV BGR frame into an RGB frame."""

    # Validate the frame before converting its channel order.
    frame = validate_frame(frame)

    # OpenCV normally reads images and video frames in BGR order.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame):
    """Convert an RGB frame into OpenCV's BGR format."""

    # Validate the frame before converting its channel order.
    frame = validate_frame(frame)

    # OpenCV video writers commonly expect BGR frames.
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def resize_frame(frame, max_width=1280, max_height=720):
    """Resize a frame while preserving its original aspect ratio."""

    # Validate the frame before resizing.
    frame = validate_frame(frame)

    # Read the original dimensions.
    height, width = frame.shape[:2]

    # Calculate a scale factor that fits both maximum dimensions.
    scale = min(max_width / width, max_height / height, 1.0)

    # Calculate the new dimensions without distorting the image.
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    # Avoid unnecessary resizing when the image already fits.
    if new_width == width and new_height == height:
        return frame

    # Use area interpolation when shrinking an image.
    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
