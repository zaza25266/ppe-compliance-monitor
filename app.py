# Purpose: Streamlit interface for image-based and video-based
# PPE detection and potential compliance monitoring.

import os

import pandas as pd
import streamlit as st

from src.preprocessing import load_image, resize_frame
from src.inference import detect_frame
from src.analytics import analyze_compliance, PPE_CLASSES
from src.visualization import draw_detections
from src.video_processing import process_video


# Configure the Streamlit page.
st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 PPE Compliance Monitor")

st.caption(
    "YOLOv8-based PPE detection and potential compliance monitoring."
)

st.warning(
    "Compliance results are approximate. PPE-to-person association "
    "uses bounding-box geometry and does not verify actual wearing."
)


# Sidebar settings.
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10, 0.90, 0.25, 0.05
)

selected_ppe = st.sidebar.multiselect(
    "Required PPE",
    options=list(PPE_CLASSES.keys()),
    default=list(PPE_CLASSES.keys())
)


# Image processing section.
st.header("Image Detection")

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="image"
)

if image_file:
    try:
        image = resize_frame(
            load_image(image_file),
            max_width=1280,
            max_height=1280
        )

        with st.spinner("Running YOLO inference..."):
            detections = detect_frame(image, confidence)
            analytics = analyze_compliance(
                detections,
                selected_ppe
            )

            annotated = draw_detections(
                image,
                detections,
                analytics
            )

        st.image(annotated, caption="Detection Results")

        # Display image analytics.
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("People", analytics["total_people"])
        col2.metric("Potentially Compliant", analytics["compliant_people"])
        col3.metric("PPE Check Required", analytics["non_compliant_people"])
        col4.metric(
            "Potential Compliance",
            f"{analytics['compliance_rate']:.1f}%"
        )

        st.subheader("PPE Detection Counts")
        st.json(analytics["assigned_ppe_counts"])

        st.subheader("Person-Level Results")
        st.dataframe(
            pd.DataFrame(analytics["person_results"]),
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Image processing failed: {error}")


st.divider()


# Video processing section.
st.header("Video Detection")

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"],
    key="video"
)

frame_interval = st.slider(
    "Process every Nth frame",
    1, 10, 3
)

max_frames = st.slider(
    "Maximum processed frames",
    30, 1000, 300, 30
)

if video_file and st.button("Process Video"):
    output_path = None

    try:
        with st.spinner("Processing video. This may take some time..."):
            result = process_video(
                video_file,
                confidence_threshold=confidence,
                required_ppe=selected_ppe,
                frame_interval=frame_interval,
                max_frames=max_frames
            )

            output_path = result["output_path"]

        # Read the completed output video.
        with open(output_path, "rb") as video:
            video_bytes = video.read()

        st.video(video_bytes)

        frame_df = pd.DataFrame(result["frame_results"])

        st.subheader("Video Analytics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Processed Frames", result["processed_frames"])
        col2.metric(
            "Processing Time",
            f"{result['processing_seconds']:.1f}s"
        )
        col3.metric(
            "Processing FPS",
            f"{result['processing_fps']:.2f}"
        )

        st.subheader("Frame-Level Compliance Trend")

        st.line_chart(
            frame_df.set_index("frame")["compliance_rate"]
        )

        st.dataframe(frame_df, use_container_width=True)

        if result["truncated"]:
            st.warning("Processing stopped at the configured frame limit.")

    except Exception as error:
        st.error(f"Video processing failed: {error}")

    finally:
        # Remove temporary output after Streamlit has read it.
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
