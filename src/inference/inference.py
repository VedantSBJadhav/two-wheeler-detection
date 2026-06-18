from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

def run_inference(source, weights='runs/train/two_wheeler_test/weights/best.pt', conf=0.5):
    model = YOLO(weights)
    source_path = Path(source)
    
    if source_path.is_file() and source_path.suffix in ['.jpg', '.jpeg', '.png']:
        results = model.predict(source, conf=conf, save=True)
        print(f'Image inference complete. Saved to {results[0].save_dir}')
    elif source_path.is_file() and source_path.suffix in ['.mp4', '.avi', '.mov']:
        results = model.predict(source, conf=conf, save=True)
        print(f'Video inference complete. Saved to {results[0].save_dir}')
    else:
        print('Unsupported file format. Use .jpg, .png, .mp4, .avi')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    parser.add_argument('--weights', type=str, default='runs/train/two_wheeler_test/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()
    
    run_inference(args.source, args.weights, args.conf)

