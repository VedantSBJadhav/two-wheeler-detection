import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path('data/raw/train')
SPLIT_DIR = Path('data/splits')
IMAGES_DIR = RAW_DIR / 'images'
LABELS_DIR = RAW_DIR / 'labels'

for split in ['train', 'val', 'test']:
    (SPLIT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

images = sorted([f for f in IMAGES_DIR.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png']])

train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

def copy_files(image_list, split_name):
    for img_path in image_list:
        label_path = LABELS_DIR / (img_path.stem + '.txt')
        shutil.copy2(img_path, SPLIT_DIR / split_name / 'images' / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, SPLIT_DIR / split_name / 'labels' / label_path.name)

copy_files(train_images, 'train')
copy_files(val_images, 'val')
copy_files(test_images, 'test')

for split in ['train', 'val', 'test']:
    count = len(list((SPLIT_DIR / split / 'images').iterdir()))
    print(f'{split}: {count} images')

print('Dataset split complete!')

