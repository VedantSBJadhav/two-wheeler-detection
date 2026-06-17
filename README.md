# Two-Wheeler Detection for Intelligent Transportation Systems

Real-time detection of motorcycles, scooters, and mopeds using YOLOv8. Handles occlusion and dense traffic scenarios.

## Architecture

`src/
+-- data/          # Dataset pipeline
+-- models/        # Training & config
+-- inference/     # Video/image inference
+-- metrics/       # Evaluation & reporting
``n
## Quick Start

`ash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Train
python src/models/train.py

# Inference
python src/inference/inference.py --source path/to/video.mp4
``n
## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | TBD |
| mAP@0.5:0.95 | TBD |
| FPS | TBD |

## Demo

![Demo](assets/demo.gif)
