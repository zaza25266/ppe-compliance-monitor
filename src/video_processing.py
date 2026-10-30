# Purpose: Process uploaded videos, run PPE detection on sampled frames,
# generate an annotated video, and collect frame-level analytics.

import os
import tempfile
import time

import cv2

from src.inference import detect_frame
from src.analytics import analyze_compliance
from src.visualization import draw_detections
from src.preprocessing import bgr_to_rgb, rgb_to_bgr, resize_frame


def process_video(
    video_file,
    confidence_threshold=0.25,
    required_ppe=None,
    frame_interval=3,
    max_frames=300
):
    # Save uploaded video temporarily for OpenCV.
    input_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    input_file.write(video_file.getvalue())
    input_file.close()

    output_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    output_path = output_file.name
    output_file.close()

    cap = cv2.VideoCapture(input_file.name)

    if not cap.isOpened():
        os.remove(input_file.name)
        os.remove(output_path)
        raise ValueError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25

    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    frame_results = []
    processed = 0
    start_time = time.time()

    try:
        while processed < max_frames:
            success, frame = cap.read()

            if not success:
                break

            # Process only selected frames.
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_number % frame_interval != 0:
                continue

            frame = resize_frame(frame)
            rgb_frame = bgr_to_rgb(frame)

            detections = detect_frame(
                rgb_frame,
                confidence_threshold
            )

            analytics = analyze_compliance(
                detections,
                required_ppe
            )

            annotated = draw_detections(
                rgb_frame,
                detections,
                analytics
            )

            output_frame = rgb_to_bgr(annotated)

            # Initialize output writer using the processed frame size.
            if writer is None:
                height, width = output_frame.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(fps / frame_interval, 1),
                    (width, height)
                )

            writer.write(output_frame)

            frame_results.append({
                "frame": frame_number,
                "people": analytics["total_people"],
                "compliant": analytics["compliant_people"],
                "non_compliant": analytics["non_compliant_people"],
                "compliance_rate": analytics["compliance_rate"]
            })

            processed += 1

    finally:
        # Release video resources before Streamlit reads the output.
        cap.release()

        if writer is not None:
            writer.release()

        os.remove(input_file.name)

    if processed == 0:
        os.remove(output_path)
        raise ValueError("No frames could be processed.")

    elapsed = time.time() - start_time

    return {
        "output_path": output_path,
        "frame_results": frame_results,
        "source_frames": source_frames,
        "processed_frames": processed,
        "processing_seconds": elapsed,
        "processing_fps": processed / elapsed if elapsed else 0,
        "truncated": processed >= max_frames
    }
