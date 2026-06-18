# 🛵 Two-Wheeler Detection for Intelligent Transportation Systems

A computer vision-based traffic monitoring system for detecting and localizing two-wheelers in real-world road environments using YOLOv8.

Developed as part of an **Intelligent Transportation Systems (ITS)** initiative, this project focuses on reliable detection of motorcycles, scooters, and mopeds under challenging urban traffic conditions. The system is designed to operate effectively in scenarios involving traffic congestion, partial occlusions, varying viewpoints, and complex road scenes commonly observed in Indian cities.

Accurate two-wheeler detection serves as a foundational component for several ITS applications including vehicle counting, traffic flow analysis, congestion monitoring, transportation planning, and smart-city infrastructure.

---

## Project Context

This project was developed during an internship/research initiative focused on applying Computer Vision and Deep Learning techniques to transportation analytics.

The primary objective was to build a lightweight yet accurate detection pipeline capable of identifying two-wheelers in dense traffic environments. Particular emphasis was placed on achieving real-time performance while maintaining robustness against occlusions and background clutter.

The system can be extended into larger transportation solutions involving:

- Traffic Density Estimation
- Vehicle Counting
- Traffic Flow Analytics
- Automatic Number Plate Recognition (ANPR)
- Traffic Violation Detection
- Smart Surveillance Systems

---

## Key Features

- Real-time two-wheeler detection
- Lightweight YOLOv8-based architecture
- Optimized for dense urban traffic scenes
- Robust against partial occlusions
- Modular training and inference pipeline
- Evaluation and performance reporting utilities
- Edge-device friendly deployment

---

## Performance

| Metric          | Score        |
| --------------- | ------------ |
| mAP@0.5         | 95.6%        |
| mAP@0.5:0.95    | 62.5%        |
| Precision       | 93.2%        |
| Recall          | 88.6%        |
| Inference Speed | 6.3 ms/image |

These results were obtained on the test dataset after training the model for 10 epochs using the YOLOv8n architecture.

---

## Dataset

**Source:** Roboflow Universe – Two-Wheeler Detection Dataset

### Classes

- Motorcycle
- Scooter
- Moped

### Dataset Split

| Split      | Images |
| ---------- | ------ |
| Train      | 1,792  |
| Validation | 384    |
| Test       | 384    |

The dataset contains diverse traffic scenes with varying illumination conditions, camera angles, and vehicle densities, making it suitable for real-world ITS applications.

---

## Model Configuration

| Parameter        | Value     |
| ---------------- | --------- |
| Architecture     | YOLOv8n   |
| Input Resolution | 640 × 640 |
| Batch Size       | 4         |
| Epochs           | 10        |
| Optimizer        | AdamW     |

The configuration was selected to balance inference speed and detection accuracy while remaining compatible with resource-constrained GPUs such as the NVIDIA GTX 1650 (4GB VRAM).

---

## Project Structure

```text
src/
├── data/              # Dataset preparation and preprocessing
├── models/            # Training scripts and model configurations
├── inference/         # Image and video inference pipelines
└── metrics/           # Evaluation and reporting utilities

assets/
├── demo.gif           # Demo output

requirements.txt
README.md
```

---

## Installation

Clone the repository and install the required dependencies.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

Train the model using:

```bash
python src/models/train.py
```

The training pipeline handles dataset loading, preprocessing, augmentation, model training, validation, and checkpoint generation.

---

## Running Inference

### Image Inference

```bash
python src/inference/inference.py --source path/to/image.jpg
```

### Video Inference

```bash
python src/inference/inference.py --source path/to/video.mp4
```

The output contains detected two-wheelers with bounding boxes and confidence scores.

---

## Demo

Generate a demonstration GIF using:

```bash
python src/inference/generate_demo.py
```

Example output:

```text
assets/demo.gif
```

---

## Applications

Potential use cases for this system include:

- Intelligent Transportation Systems (ITS)
- Urban Traffic Monitoring
- Vehicle Counting and Classification
- Smart City Infrastructure
- Transportation Analytics
- Congestion Analysis
- Road Safety Monitoring
- Automated Traffic Surveillance

---

## Future Work

Planned improvements include:

- Multi-class vehicle detection
- Vehicle tracking using ByteTrack or DeepSORT
- Traffic analytics dashboard
- Edge deployment with TensorRT
- CCTV integration for continuous monitoring
- Traffic density estimation and visualization
- ANPR integration for end-to-end traffic intelligence

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Roboflow

---

## Development Workflow

This repository follows semantic commit conventions:

```text
feat:      New feature
fix:       Bug fix
docs:      Documentation updates
refactor:  Code restructuring
chore:     Maintenance tasks
```

---

## Acknowledgements

This work was carried out as part of an internship/research initiative in the field of Intelligent Transportation Systems (ITS).

The project leverages the Ultralytics YOLOv8 framework and publicly available datasets for experimentation, evaluation, and model development. Special thanks to the open-source community and dataset contributors whose resources enabled the development and validation of this system.
