# PPE Compliance Monitor

A computer vision application for detecting personal protective equipment (PPE) and monitoring potential compliance issues using **YOLOv8, OpenCV, and Streamlit**.

The application combines a custom-trained PPE detection model with a pretrained person detector to analyze workplace images and videos, visualize detections, associate PPE items with detected people, and generate runtime analytics.

## Features

- **PPE Detection:** Detects helmets, masks, safety vests, boots, and gloves.
- **Person Detection:** Uses a pretrained YOLOv8 model to locate people in an image or video frame.
- **Image Inspection:** Upload workplace images and visualize bounding boxes, class labels, and confidence scores.
- **Video Inspection:** Process uploaded videos frame by frame and generate annotated video output.
- **Person-Level Analytics:** Associate detected PPE items with people and flag potentially missing equipment.
- **Interactive Dashboard:** View detected people, potential compliance status, per-item breakdowns, and frame-level analytics.
- **Configurable Requirements:** Select the PPE categories required for an inspection.
- **Hugging Face Integration:** Load the custom PPE model from the Hugging Face Hub.
- **Streamlit Deployment:** Run the application through a browser-based interface.

## Application Preview

The application provides an image and video inspection workflow with annotated detections and a compliance analytics dashboard.

> Add screenshots of the running application here after deployment.

Suggested screenshots:
- `assets/image-detection.png` — annotated workplace image
- `assets/compliance-dashboard.png` — PPE breakdown and person-level results
- `assets/video-analytics.png` — frame-level video analytics

## Architecture

```text
              Workplace Image / Video
                        |
                        v
             OpenCV Preprocessing
                        |
                        v
              +-------------------+
              |   YOLOv8 Models   |
              +-------------------+
                 /             \
                v               v
        Person Detector     PPE Detector
          (YOLOv8n)          (Custom)
                |               |
                v               v
           Person Boxes      PPE Boxes
                 \             /
                  \           /
                   v         v
               PPE-to-Person
                Association
                      |
                      v
             Compliance Analytics
                      |
                      v
              Streamlit Dashboard
```

### Detection Pipeline

1. Decode the uploaded image or video using the preprocessing module.
2. Run the pretrained person detector to identify people.
3. Run the custom PPE model to detect protective equipment.
4. Apply geometric association rules to associate PPE detections with people.
5. Compare associated items against the selected PPE requirements.
6. Display annotated images, potential missing-PPE flags, and analytics.

## Models

### Custom PPE Detection Model

The custom YOLOv8 model was trained to detect five PPE categories:

| Class | Description |
|---|---|
| Helmet | Protective headgear |
| Mask | Face mask |
| Safety Vest | High-visibility safety vest |
| boots | Protective boots |
| glove | Protective gloves |

Model weights: [ZubairAli25266/ppe_detector](https://huggingface.co/ZubairAli25266/ppe_detector)

Checkpoint filename: `best.pt`

The application downloads this checkpoint from the Hugging Face Hub when the model is loaded.

### Person Detection Model

The application uses the pretrained `yolov8n.pt` model for person detection. The person detector and custom PPE detector have separate responsibilities: one detects people, while the other detects protective equipment.

## Model Evaluation

The following results are from the reported **test-set evaluation** of the custom PPE detector.

- Test images: **1,177**
- Ground-truth instances: **3,751**
- Training duration: **30 epochs**

### Overall Metrics

| Metric | Score |
|---|---:|
| Precision | 0.8649 |
| Recall | 0.7449 |
| mAP@50 | 0.8229 |
| mAP@50–95 | 0.4789 |

### Per-Class Results

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Helmet | 0.900 | 0.746 | 0.805 | 0.478 |
| Mask | 0.889 | 0.838 | 0.898 | 0.522 |
| Safety Vest | 0.881 | 0.718 | 0.809 | 0.517 |
| boots | 0.798 | 0.742 | 0.822 | 0.474 |
| glove | 0.856 | 0.680 | 0.781 | 0.404 |

### Metric Definitions

- **Precision:** Fraction of predicted positive detections that are correct.
- **Recall:** Fraction of ground-truth objects successfully detected.
- **mAP@50:** Mean average precision using an IoU threshold of 0.50.
- **mAP@50–95:** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.

The object detector's evaluation metrics measure PPE detection performance. They do not measure the accuracy of person-to-PPE association or the final compliance classification. Those components require separate validation.

## Project Structure

```text
ppe-compliance-monitor/
├── notebooks/
│   ├── 01_yolov8_training.ipynb
│   └── 02_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── inference.py
│   ├── analytics.py
│   ├── visualization.py
│   └── video_processing.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Module Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, image/video uploads, dashboard rendering |
| `src/preprocessing.py` | Image validation, color conversion, and resizing |
| `src/inference.py` | Model loading and person/PPE inference |
| `src/analytics.py` | PPE-to-person association and compliance calculations |
| `src/visualization.py` | Annotated detection images and status labels |
| `src/video_processing.py` | Video decoding, frame processing, and annotated video output |
| `notebooks/01_yolov8_training.ipynb` | Training workflow |
| `notebooks/02_model_evaluation.ipynb` | Evaluation and metric analysis |

## Installation

### Requirements

- Python 3.10 or a compatible Python version supported by the selected dependencies
- pip
- Internet access for downloading model weights
- CPU or compatible GPU for inference

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ppe-compliance-monitor.git
cd ppe-compliance-monitor
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL for opening the application in a browser.

## Usage

### Image Inspection

1. Open the application.
2. Select the PPE categories required for the inspection.
3. Adjust the confidence threshold if needed.
4. Upload a workplace image.
5. Review person and PPE detections.
6. Inspect the per-person results and PPE breakdown.
7. Verify potential missing-PPE flags manually.

### Video Inspection

1. Upload a supported workplace video.
2. Choose the frame sampling interval.
3. Set the maximum number of frames to process.
4. Start video analysis.
5. Review the annotated output video and frame-level analytics.

Processing every Nth frame can reduce inference time, but may miss brief events or objects visible only between sampled frames.

## Analytics

The dashboard provides the following metrics:

| Metric | Meaning |
|---|---|
| People detected | Number of person detections in the current image or processed frame |
| Potentially compliant | Detected people associated with every selected required PPE category |
| PPE check required | Detected people with one or more required items not detected |
| Estimated compliance | Percentage of detected people classified as potentially compliant |
| People with item detected | People associated with a particular PPE category |
| People potentially missing item | People for whom a required PPE category was not detected |

For video processing, these metrics are calculated per processed frame. They do not represent unique workers across the entire video.

## Person-to-PPE Association

The application uses bounding-box geometry and approximate body-region rules to associate detected PPE items with people.

Examples include:
- Helmets are expected near the upper part of a person's bounding box.
- Safety vests are expected around the torso.
- Boots are expected near the lower part of the bounding box.
- Gloves are expected around plausible arm and hand regions.

These rules are heuristic approximations. They can fail when people overlap, body parts are occluded, bounding boxes are inaccurate, or PPE items are detected near multiple people.

A PPE item not detected or not associated with a person is **not proof that the person is violating a safety rule**. The application should be treated as a decision-support prototype, with human verification of flagged cases.

## Limitations

- PPE detection performance depends on image quality, lighting, occlusion, camera angle, and object size.
- Person-to-PPE association is heuristic and has not been established as a validated compliance classifier.
- A missing detection does not necessarily mean an item is absent.
- Video analytics currently count detections per processed frame, not unique individuals.
- The pretrained person detector may miss people in crowded or unusual scenes.
- CPU-based inference may be slow on long videos.
- Live browser-camera streaming requires client-side camera integration; server-side OpenCV capture does not directly access a visitor's webcam on a hosted Space.
- The system is not a certified safety inspection or enforcement tool.

## Future Improvements

- Add multi-object tracking for unique worker counts across video frames.
- Improve person-to-PPE association using pose estimation or more robust spatial methods.
- Evaluate compliance association against manually annotated test examples.
- Add confidence calibration and error-analysis reports.
- Support configurable workplace PPE policies.
- Optimize video inference and add progress reporting.
- Add browser-side live-camera support.
- Add automated tests and continuous integration.
- Record inspection summaries and export analytics reports.
- Improve the dashboard's accessibility and error handling.

## Technology Stack

- **Python** — application and model integration
- **Ultralytics YOLOv8** — object detection
- **PyTorch** — deep learning runtime used by Ultralytics
- **OpenCV** — image and video processing
- **NumPy** — image arrays and numerical operations
- **Pillow** — image decoding and conversion
- **Streamlit** — interactive application interface
- **Hugging Face Hub** — model-weight hosting and retrieval
- **Pandas** — tabular frame-level analytics

## License

Add a `LICENSE` file before distributing this project. Choose a license that matches your intended use and the licenses of the dataset, model, and dependencies. Do not assume the dataset or pretrained model licenses automatically permit every form of redistribution or commercial use.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework.
- [Hugging Face](https://huggingface.co/) for model hosting and Spaces deployment.
- [Streamlit](https://streamlit.io/) for the application interface.

---

**Disclaimer:** This project is intended for educational and prototyping purposes. It does not replace formal workplace safety inspections, applicable safety regulations, or qualified human judgment.
