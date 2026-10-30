# Purpose: Draw person and PPE bounding boxes on RGB images.

import cv2


def draw_detections(frame, detections, analytics):
    # Draw person boxes and PPE boxes with confidence labels.
    output = frame.copy()

    for person, result in zip(
        detections["people"],
        analytics["person_results"]
    ):
        x1, y1, x2, y2 = person["bbox"]

        if result["potentially_compliant"]:
            color = (0, 200, 0)
            label = "Potentially compliant"
        else:
            color = (255, 150, 0)
            label = "PPE check required"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # Draw detected PPE objects.
    for item in detections["ppe_items"]:
        x1, y1, x2, y2 = item["bbox"]

        label = (
            f"{item['class_name']} "
            f"{item['confidence']:.2f}"
        )

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            2
        )

    return output
