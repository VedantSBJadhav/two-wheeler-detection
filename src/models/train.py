from ultralytics import YOLO
import yaml
from pathlib import Path

def train_model():
    # Load YOLOv8 nano (fastest, smallest)
    model = YOLO('yolov8n.pt')
    
    # Load data config
    data_path = Path('configs/data.yaml').absolute()
    
    # Train
    results = model.train(
        data=str(data_path),
        epochs=10,  # Quick test run
        imgsz=640,
        batch=16,
        name='two_wheeler_test',
        project='runs/train',
        exist_ok=True,
        patience=5,
        save=True,
        plots=True
    )
    
    print(f'Training complete. Results: {results.results_dict}')

if __name__ == '__main__':
    train_model()

