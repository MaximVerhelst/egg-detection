# Egg Detection & Classification

A computer vision prototype for automated egg classification using a ZED 2i stereo camera. Built as a job application project.

## Features

- **Egg detection** — YOLOv8n object detection (mAP50: 97.2%)
- **Crack detection** — YOLOv8n object detection (mAP50: 98.0%)
- **Colour classification** — K-means clustering (white / brown)
- **Size measurement** — ZED depth map + ellipse fitting → volume in cm³ (S / M / L)

## Hardware

- ZED 2i stereo camera
- Black box with LED strips
- Camera mounted above egg carton

## Project Structure

```
Egg Detection/
├── classification_eggs.py   # Main pipeline (detection + colour + volume + crack)
├── calibrate.py             # Camera & carton level calibration
├── calibration.json         # Saved calibration (generated at runtime)
├── color_analysis.py        # Colour detection test on saved images
├── measure_eggs.py          # Volume measurement test with live camera
├── test_detection.py        # Detection test on test set images
├── train.py                 # YOLOv8 training script
├── models/
│   ├── egg_detection/       # Trained egg detection model
│   └── crack_detection/     # Trained crack detection model
├── datasets/
│   ├── egg_detection/       # Annotated egg dataset (Roboflow export)
│   └── crack_detection/     # Annotated crack dataset (Roboflow export)
└── data_collection/
    ├── capture.py           # Live image capture with ZED camera
    ├── camera_tune.py       # Camera settings tuning
    ├── calculate_scores.py  # Calculate best camera settings
    ├── capture_settings.py  # Camera settings configuration
    ├── export_report.py     # Export results to report
    └── images/              # Captured training images
```

## Installation

```bash
pip install -r requirements.txt
```

The ZED SDK must be installed separately: https://www.stereolabs.com/developers/release

## Usage

### 1. Calibration (first time or after moving the camera)
```bash
python calibrate.py
```
- Click 4 corner points on the carton → camera tilt correction
- Click 1 empty spot on the carton → carton level reference
- Calibration is saved to `calibration.json`

### 2. Run main pipeline
```bash
python classification_eggs.py
```
- Prompts whether to recalibrate
- Loads calibration and starts live detection

### 3. Test scripts (no camera required)
```bash
python color_analysis.py    # Test colour detection on saved images
python test_detection.py    # Test egg & crack detection on test set
```

## Models

Model weights are not included in this repository due to file size.  
Download from: *(link to be added)*

Place the weights in:
- `models/egg_detection/weights/best.pt`
- `models/crack_detection/weights/best.pt`

## Results

| Model | mAP50 | mAP50-95 |
|---|---|---|
| Egg detection | 97.2% | 93.2% |
| Crack detection | 98.0% | 92.4% |

## Size Classification

Based on egg volume (cm³):

| Class | Volume |
|---|---|
| S (Small) | < 40 cm³ |
| M (Medium) | 40 – 50 cm³ |
| L (Large) | > 50 cm³ |