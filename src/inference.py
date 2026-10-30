
# src/inference.py
# Purpose: Load the person and PPE YOLO models, run inference,
# and return consistent detection dictionaries for the analytics module.

from functools import lru_cache

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.preprocessing import validate_frame


# Hugging Face repository containing our trained PPE detector.
PPE_MODEL_REPO = "ZubairAli25266/ppe_detector"
PPE_MODEL_FILENAME = "best.pt"


@lru_cache(maxsize=1)
def load_models():
    """Load the person detector and PPE detector once per app process."""

    # Download the trained PPE weights from Hugging Face.
    ppe_model_path = hf_hub_download(
        repo_id=PPE_MODEL_REPO,
        filename=PPE_MODEL_FILENAME,
        repo_type="model",
    )

    # Load a pretrained COCO model; class ID 0 represents a person.
    person_model = YOLO("yolov8n.pt")

    # Load our custom model trained to detect PPE items.
    ppe_model = YOLO(ppe_model_path)

    return person_model, ppe_model


def _extract_detections(result, model_type, confidence_threshold):
    """Convert one Ultralytics result into a list of dictionaries."""

    # Return an empty list if the model found no boxes.
    if result.boxes is None or len(result.boxes) == 0:
        return []

    detections = []

    # Read each predicted bounding box, class ID, and confidence score.
    for box in result.boxes:
        confidence = float(box.conf[0].item())

        # Ignore detections below the requested confidence threshold.
        if confidence < confidence_threshold:
            continue

        # Convert coordinates to ordinary Python integers.
        x1, y1, x2, y2 = (
            int(value) for value in box.xyxy[0].tolist()
        )

        # Resolve the predicted class ID to its model-provided name.
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]

        # Keep the two detector outputs distinguishable for analytics.
        detections.append(
            {
                "model_type": model_type,
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections


def detect_frame(
    frame: np.ndarray,
    confidence_threshold: float = 0.25,
    person_image_size: int = 640,
    ppe_image_size: int = 640,
):
    """Detect people and PPE in one RGB image or video frame."""

    # Validate the incoming RGB frame.
    frame = validate_frame(frame)

    # Load cached model instances instead of downloading on every frame.
    person_model, ppe_model = load_models()

    # Run person detection. The COCO class ID for person is 0.
    person_result = person_model.predict(
        source=frame,
        imgsz=person_image_size,
        conf=confidence_threshold,
        classes=[0],
        verbose=False,
    )[0]

    # Run custom PPE detection on the same frame.
    ppe_result = ppe_model.predict(
        source=frame,
        imgsz=ppe_image_size,
        conf=confidence_threshold,
        verbose=False,
    )[0]

    # Convert both outputs into the same predictable data structure.
    people = _extract_detections(
        person_result,
        model_type="person",
        confidence_threshold=confidence_threshold,
    )

    ppe_items = _extract_detections(
        ppe_result,
        model_type="ppe",
        confidence_threshold=confidence_threshold,
    )

    # Return detections without yet claiming which person wears which item.
    return {
        "people": people,
        "ppe_items": ppe_items,
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
    }
