from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

def evaluate_model(weights='runs/detect/runs/train/two_wheeler_test/weights/best.pt', data='configs/data.yaml'):
    model = YOLO(weights)
    metrics = model.val(data=data, split='test', save=True)
    
    results = {
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'per_class': {
            name: {
                'precision': float(p),
                'recall': float(r),
                'mAP50': float(ap50)
            }
            for name, p, r, ap50 in zip(metrics.names.values(), metrics.box.p, metrics.box.r, metrics.box.ap50)
        }
    }
    
    output_dir = Path('metrics_output')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'mAP@0.5: {results["mAP50"]:.3f}')
    print(f'mAP@0.5:0.95: {results["mAP50_95"]:.3f}')
    print(f'Precision: {results["precision"]:.3f}')
    print(f'Recall: {results["recall"]:.3f}')
    print(f'Results saved to {output_dir}/results.json')

if __name__ == '__main__':
    evaluate_model()

