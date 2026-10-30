
# app.py
# Purpose: Provide a Streamlit interface for PPE detection,
# annotated image visualization, and per-person compliance analytics.

import cv2
import streamlit as st

from src.analytics import PPE_CLASSES, analyze_compliance
from src.inference import detect_frame
from src.preprocessing import load_image, resize_frame


# Configure the browser tab and overall application layout.
st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="🦺",
    layout="wide",
)

# Display the application header.
st.title("PPE Compliance Monitor")
st.caption(
    "Computer vision prototype for detecting personal protective "
    "equipment and identifying potentially missing items."
)

# Explain the current limitations of the compliance estimates.
st.info(
    "This is a prototype. PPE-to-person matching uses bounding-box "
    "heuristics, so missing-item flags require human verification."
)

# Define the supported PPE categories and their display labels.
PPE_LABELS = {
    "helmet": "Helmet",
    "mask": "Mask",
    "safety vest": "Safety Vest",
    "boots": "Boots",
    "glove": "Gloves",
}


def draw_detections(frame, detections, analytics):
    """Draw person/PPE boxes and per-person status on an RGB frame."""

    # Copy the input so visualization does not mutate the original frame.
    annotated = frame.copy()

    # Draw each person's bounding box and compliance status.
    for person in analytics["person_results"]:
        x1, y1, x2, y2 = person["bbox"]

        # Green indicates all configured items were detected.
        # Orange indicates one or more required items were not detected.
        color = (0, 180, 0) if person["compliant"] else (255, 140, 0)

        # Draw the person's bounding box.
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # Build a concise label for the person's current status.
        status = (
            "Potentially compliant"
            if person["compliant"]
            else "PPE check required"
        )
        label = f"Person {person['person_id']}: {status}"

        # Position the label above the bounding box when possible.
        text_y = max(20, y1 - 8)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Draw all PPE detections, including items not associated with a person.
    for item in detections["ppe_items"]:
        x1, y1, x2, y2 = item["bbox"]

        # Use cyan for PPE object detections.
        color = (0, 200, 255)

        # Include the class name and detector confidence in the label.
        label = f"{item['class_name']} {item['confidence']:.2f}"

        # Draw the PPE box and its label.
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )
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

    # Return the annotated RGB image for Streamlit display.
    return annotated


# Create the sidebar for inference and compliance settings.
st.sidebar.header("Inspection settings")

confidence_threshold = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Higher values filter out more low-confidence detections.",
)

selected_ppe = st.sidebar.multiselect(
    "Required PPE",
    options=list(PPE_LABELS.keys()),
    default=list(PPE_LABELS.keys()),
    format_func=lambda value: PPE_LABELS[value],
    help="A person is potentially compliant when all selected items "
    "are associated with that person.",
)

# Provide a separate upload area for the initial image-based workflow.
st.header("Image inspection")

uploaded_image = st.file_uploader(
    "Upload a workplace image",
    type=["jpg", "jpeg", "png", "webp"],
)

# Process the image only after the user uploads a file.
if uploaded_image is not None:
    try:
        # Decode the uploaded image and standardize it to RGB.
        image = load_image(uploaded_image)

        # Resize very large images to reduce inference latency.
        image = resize_frame(
            image,
            max_width=1280,
            max_height=1280,
        )

        # Run both models and collect person/PPE detections.
        with st.spinner("Detecting people and PPE..."):
            detections = detect_frame(
                image,
                confidence_threshold=confidence_threshold,
            )

            # Calculate person-level statuses using selected PPE requirements.
            analytics = analyze_compliance(
                detections,
                required_ppe=selected_ppe,
            )

        # Draw boxes and status labels over the original image.
        annotated_image = draw_detections(
            image,
            detections,
            analytics,
        )

        # Display the processed image alongside its analytics.
        image_column, analytics_column = st.columns(
            [1.4, 1],
            gap="large",
        )

        with image_column:
            st.subheader("Detection results")
            st.image(
                annotated_image,
                caption="Person and PPE detections",
                width="stretch",
            )

        with analytics_column:
            st.subheader("Inspection summary")

            # Present the main runtime analytics as metric cards.
            metric_column_1, metric_column_2 = st.columns(2)
            metric_column_1.metric(
                "People detected",
                analytics["total_people"],
            )
            metric_column_2.metric(
                "Potentially compliant",
                analytics["compliant_people"],
            )

            metric_column_3, metric_column_4 = st.columns(2)
            metric_column_3.metric(
                "PPE check required",
                analytics["non_compliant_people"],
            )
            metric_column_4.metric(
                "Estimated compliance",
                f"{analytics['compliance_rate']:.1f}%",
            )

            st.caption(
                "Compliance is an estimate based on detected PPE "
                "and the selected requirements, not a verified "
                "safety-compliance measurement."
            )

        # Show the per-category breakdown.
        st.subheader("PPE breakdown")

        if not selected_ppe:
            st.warning(
                "No required PPE categories are selected. "
                "Select at least one category to calculate compliance."
            )
        else:
            rows = []

            for item in selected_ppe:
                rows.append(
                    {
                        "PPE item": PPE_LABELS[item],
                        "People with item detected": (
                            analytics["people_with_ppe"].get(item, 0)
                        ),
                        "People potentially missing item": (
                            analytics["people_missing_ppe"].get(item, 0)
                        ),
                    }
                )

            st.dataframe(
                rows,
                hide_index=True,
                width="stretch",
            )

        # Display the status and missing items for each detected person.
        st.subheader("Person-level results")

        if analytics["person_results"]:
            person_rows = []

            for person in analytics["person_results"]:
                person_rows.append(
                    {
                        "Person": person["person_id"],
                        "Status": (
                            "Potentially compliant"
                            if person["compliant"]
                            else "PPE check required"
                        ),
                        "Detected PPE": ", ".join(
                            PPE_LABELS.get(item, item)
                            for item in person["detected_ppe"]
                        ) or "None detected",
                        "Potentially missing": ", ".join(
                            PPE_LABELS.get(item, item)
                            for item in person["missing_ppe"]
                        ) or "None",
                    }
                )

            st.dataframe(
                person_rows,
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning(
                "No people were detected. Try a clearer image or "
                "adjust the confidence threshold."
            )

        # Allow inspection of the raw model output when debugging.
        with st.expander("Technical detection details"):
            st.json(
                {
                    "people": detections["people"],
                    "ppe_items": detections["ppe_items"],
                    "image_width": detections["image_width"],
                    "image_height": detections["image_height"],
                }
            )

    except Exception as exc:
        # Report inference or image-processing errors without crashing the UI.
        st.error(
            "Could not process this image. Check the image format, "
            "model availability, and application logs."
        )
        st.exception(exc)

else:
    # Show instructions before an image has been uploaded.
    st.markdown(
        """
        ### How to use the monitor

        1. Select the PPE items required for the inspection.
        2. Adjust the detection confidence threshold if necessary.
        3. Upload a workplace image.
        4. Review the bounding boxes and per-person results.
        5. Verify potential missing-PPE flags manually.
        """
    )

# Display a small footer.
st.divider()
st.caption(
    "PPE Compliance Monitor | YOLOv8 · OpenCV · Streamlit"
)
