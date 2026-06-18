from ultralytics import YOLO
from pathlib import Path

def train_model():
    model = YOLO('yolov8n.pt')
    data_path = Path('configs/data.yaml').absolute()
    
    results = model.train(
        data=str(data_path),
        epochs=10,
        imgsz=640,
        batch=4,
        name='two_wheeler_test',
        project='runs/train',
        exist_ok=True,
        patience=5,
        save=True,
        plots=True,
        device=0,
        workers=0
    )
    
    print(f'Training complete. Results: {results.results_dict}')

if __name__ == '__main__':
    train_model()

