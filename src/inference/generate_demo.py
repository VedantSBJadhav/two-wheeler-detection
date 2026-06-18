from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import imageio

def generate_demo_gif(weights='runs/detect/runs/train/two_wheeler_test/weights/best.pt', num_images=10, output='demos/demo.gif'):
    model = YOLO(weights)
    test_images = sorted(list(Path('data/splits/test/images').glob('*.jpg')))[:num_images]
    
    frames = []
    target_size = (640, 480)
    
    for img_path in test_images:
        results = model.predict(str(img_path), conf=0.5, verbose=False)
        annotated = results[0].plot()
        resized = cv2.resize(annotated, target_size)
        frames.append(resized)
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=500, loop=0)
    print(f'Demo GIF saved to {output}')

if __name__ == '__main__':
    generate_demo_gif()

