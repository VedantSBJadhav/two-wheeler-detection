import os
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path('data/raw')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Roboflow Universe - Two Wheeler Dataset (public, no API key needed)
# URL placeholder - we'll use a direct download link
ROBOFLOW_URL = 'https://universe.roboflow.com/ds/two-wheeler'

def download_roboflow_dataset():
    print('Downloading Roboflow two-wheeler dataset...')
    print('Note: For production, use roboflow API or manual download')
    print('URL:', ROBOFLOW_URL)
    print('Save to:', DATA_DIR)

if __name__ == '__main__':
    download_roboflow_dataset()

