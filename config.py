import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Flask configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.getenv('FLASK_ENV') == 'development'
TESTING = False

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

# Model configuration
MODEL_NAME = 'vgg19'
FEATURE_DIMENSION = 4096
DEVICE = 'cpu'  # Use 'cuda' for GPU

# FAISS configuration
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.json')

# Dataset configuration
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
DATASET_CATEGORIES = [
    'cars', 'motorbikes', 'pandas', 'manchester_united_jersey',
    'laptops', 'orange', 'burger', 'jeans', 'xrays', 'dogs'
]

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
