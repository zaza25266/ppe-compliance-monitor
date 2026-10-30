
# src/video_processing.py
# Purpose: Process uploaded videos frame by frame, run PPE detection,
# annotate the frames, and collect frame-level analytics.

import os
import tempfile
import time

import cv2

from src.analytics import analyze_compliance
from src.inference import detect_frame
from src.preprocessing import bgr_to_rgb, resize_frame, rgb_to_bgr


def process_video(
    video_file,
    confidence_threshold=0.25,
    required_ppe=None,
    frame_interval=3,
    max_frames=600,
):
    """Process a video and return an annotated video plus frame analytics.

    Args:
        video_file: Uploaded video file-like object.
        confidence_threshold: Minimum YOLO detection confidence.
        required_ppe: PPE categories required for this inspection.
        frame_interval: Process every Nth source frame.
        max_frames: Maximum number of frames to process.

    Returns:
        A dictionary containing the annotated video path, frame results,
        processing statistics, and summary analytics.
    """

    # Validate the frame interval and processing limit.
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1.")

    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")

    # Save the uploaded video to a temporary file for OpenCV.
    suffix = os.path.splitext(
        getattr(video_file, "name", "uploaded_video.mp4")
    )[1] or ".mp4"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_input:
        temp_input.write(video_file.getvalue())
        input_path = temp_input.name

    output_path = None
    capture = None
    writer = None
    frame_results = []
    total_source_frames = 0
    processed_frames = 0
    processing_start = time.perf_counter()

    try:
        # Open the uploaded video with OpenCV.
        capture = cv2.VideoCapture(input_path)

        if not capture.isOpened():
            raise ValueError(
                "Could not open this video. Try uploading a valid MP4 file."
            )

        # Read source video properties.
        fps = capture.get(cv2.CAP_PROP_FPS)
        source_frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Use a fallback FPS if the file does not report a valid value.
        if fps <= 0:
            fps = 25.0

        # Create a temporary output video using a broadly supported codec.
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as temp_output:
            output_path = temp_output.name

        writer = None

        # Read and process frames until the video ends or the limit is reached.
        while processed_frames < max_frames:
            success, frame_bgr = capture.read()

            if not success:
                break

            source_frame_index = total_source_frames
            total_source_frames += 1

            # Skip frames according to the configured interval.
            if source_frame_index % frame_interval != 0:
                continue

            # Resize large frames to control inference latency.
            frame_bgr = resize_frame(
                frame_bgr,
                max_width=1280,
                max_height=720,
            )

            # Convert OpenCV's BGR frame into RGB for our inference pipeline.
            frame_rgb = bgr_to_rgb(frame_bgr)

            # Detect people and PPE in this sampled frame.
            detections = detect_frame(
                frame_rgb,
                confidence_threshold=confidence_threshold,
            )

            # Calculate potential compliance for the current frame.
            analytics = analyze_compliance(
                detections,
                required_ppe=required_ppe,
            )

            # Draw the bounding boxes and per-person status labels.
            from app import draw_detections

            annotated_rgb = draw_detections(
                frame_rgb,
                detections,
                analytics,
            )

            # Convert the annotated frame back to BGR for video writing.
            annotated_bgr = rgb_to_bgr(annotated_rgb)

            # Initialize the output writer using the actual processed size.
            if writer is None:
                height, width = annotated_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    max(1.0, fps / frame_interval),
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create the annotated output video."
                    )

            # Write this annotated frame to the output video.
            writer.write(annotated_bgr)

            # Save a compact analytics record for this processed frame.
            frame_results.append(
                {
                    "source_frame": source_frame_index,
                    "people_detected": analytics["total_people"],
                    "potentially_compliant": analytics["compliant_people"],
                    "ppe_check_required": analytics["non_compliant_people"],
                    "compliance_rate": analytics["compliance_rate"],
                }
            )

            processed_frames += 1

        # Reject files that produced no processable frames.
        if processed_frames == 0:
            raise ValueError(
                "No frames were processed. Check the video file "
                "and the frame interval."
            )

        # Compute processing statistics after the video is finished.
        elapsed_seconds = time.perf_counter() - processing_start
        processing_fps = (
            processed_frames / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        # Compute mean frame-level values; these are not unique-person counts.
        mean_people = sum(
            row["people_detected"] for row in frame_results
        ) / processed_frames

        mean_compliance = sum(
            row["compliance_rate"] for row in frame_results
        ) / processed_frames

        return {
            "output_path": output_path,
            "frame_results": frame_results,
            "source_frame_count": source_frame_count,
            "frames_read": total_source_frames,
            "processed_frames": processed_frames,
            "processing_seconds": round(elapsed_seconds, 2),
            "processing_fps": round(processing_fps, 2),
            "mean_people_per_frame": round(mean_people, 2),
            "mean_frame_compliance_rate": round(mean_compliance, 2),
            "truncated": processed_frames >= max_frames,
        }

    finally:
        # Release video resources even when processing raises an exception.
        if capture is not None:
            capture.release()

        if writer is not None:
            writer.release()

        # Remove the temporary input file; preserve output for the UI.
        if os.path.exists(input_path):
            os.remove(input_path)
