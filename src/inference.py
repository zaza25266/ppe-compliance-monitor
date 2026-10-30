# Purpose: Load YOLO models and detect people and PPE items.

from functools import lru_cache

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.preprocessing import validate_frame


PPE_MODEL_REPO = "ZubairAli25266/ppe_detector"
PPE_MODEL_FILENAME = "best.pt"


@lru_cache(maxsize=1)
def load_models():
    # Download the trained PPE model and load the person detector.
    ppe_path = hf_hub_download(
        repo_id=PPE_MODEL_REPO,
        filename=PPE_MODEL_FILENAME,
        repo_type="model"
    )

    person_model = YOLO("yolov8n.pt")
    ppe_model = YOLO(ppe_path)

    return person_model, ppe_model


def extract_detections(result, model_type):
    # Convert YOLO results into a consistent dictionary format.
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        class_id = int(box.cls[0].item())

        detections.append({
            "model_type": model_type,
            "class_id": class_id,
            "class_name": str(result.names[class_id]),
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })

    return detections


def detect_frame(frame, confidence_threshold=0.25):
    # Run person detection and PPE detection on the same frame.
    frame = validate_frame(frame)

    person_model, ppe_model = load_models()

    person_result = person_model.predict(
        source=frame,
        conf=confidence_threshold,
        classes=[0],
        imgsz=640,
        verbose=False
    )[0]

    ppe_result = ppe_model.predict(
        source=frame,
        conf=confidence_threshold,
        imgsz=640,
        verbose=False
    )[0]

    return {
        "people": extract_detections(person_result, "person"),
        "ppe_items": extract_detections(ppe_result, "ppe"),
        "image_width": frame.shape[1],
        "image_height": frame.shape[0]
    }
