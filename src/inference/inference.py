from ultralytics import YOLO
import argparse
from pathlib import Path

def run_inference(source, weights='runs/detect/runs/train/two_wheeler_test/weights/best.pt', conf=0.5):
    model = YOLO(weights)
    results = model.predict(source, conf=conf, save=True)
    print(f'Inference complete. Saved to {results[0].save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--weights', type=str, default='runs/detect/runs/train/two_wheeler_test/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()
    run_inference(args.source, args.weights, args.conf)

