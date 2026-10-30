
# src/visualization.py
# Purpose: Draw person and PPE detections on RGB images so both
# image and video workflows can reuse the same visualization logic.

import cv2


def draw_detections(frame, detections, analytics):
    """Draw detection boxes and per-person status labels on an RGB frame."""

    # Copy the frame so the source image remains unchanged.
    annotated = frame.copy()

    # Draw the detected people and their current potential compliance status.
    for person in analytics["person_results"]:
        x1, y1, x2, y2 = person["bbox"]
        color = (0, 180, 0) if person["compliant"] else (255, 140, 0)

        # Draw the person bounding box.
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Add a concise status label above the person box.
        status = (
            "Potentially compliant"
            if person["compliant"]
            else "PPE check required"
        )
        label = f"Person {person['person_id']}: {status}"

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Draw PPE boxes and model confidence scores.
    for item in detections["ppe_items"]:
        x1, y1, x2, y2 = item["bbox"]
        color = (0, 200, 255)
        label = f"{item['class_name']} {item['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Return the annotated RGB image.
    return annotated
