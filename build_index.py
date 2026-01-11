#!/usr/bin/env python3
"""Build FAISS index from downloaded dataset."""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure backend modules are importable
sys.path.insert(0, '/workspaces/image-retrieval-system-based-on-machine-learning')

from backend.feature_extractor import FeatureExtractor
from backend.faiss_index import FAISSIndex
from config import DATASET_FOLDER, DATASET_CATEGORIES


def build_dataset_index():
    """Build FAISS index from dataset."""
    print("="*60)
    print("Building FAISS Index from Dataset")
    print("="*60)
    
    # Initialize components
    print("\nInitializing FeatureExtractor (VGG19)...")
    extractor = FeatureExtractor()
    
    print("Initializing FAISS Index...")
    index = FAISSIndex()
    
    # Collect all images
    image_files = []
    print("\nScanning dataset...")
    for category in DATASET_CATEGORIES:
        category_path = os.path.join(DATASET_FOLDER, category)
        if not os.path.exists(category_path):
            print(f"  ✗ Category not found: {category}")
            continue
        
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.extend([(os.path.join(category_path, img), category) for img in images])
        print(f"  ✓ {category}: {len(images)} images")
    
    if not image_files:
        print("\n✗ No images found in dataset!")
        return False
    
    print(f"\n✓ Total images to index: {len(image_files)}")
    
    # Extract features and build index
    print("\nExtracting features and building index...")
    features_list = []
    urls_list = []
    failed = 0
    
    for i, (image_path, category) in enumerate(image_files, 1):
        try:
            # Extract features
            features = extractor.extract_features(image_path)
            features_list.append(features)
            
            # Store relative URL-like path as identifier
            relative_path = os.path.relpath(image_path, DATASET_FOLDER)
            urls_list.append(relative_path)
            
            if i % 50 == 0:
                print(f"  ✓ Processed {i}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {e}")
            failed += 1
    
    if not features_list:
        print("✗ No features extracted!")
        return False
    
    # Add to FAISS index
    print(f"\nAdding {len(features_list)} features to FAISS index...")
    features_array = np.array(features_list, dtype=np.float32)
    index.add_features(features_array, urls_list)
    
    # Save index
    print("\nSaving FAISS index...")
    index.save()
    
    print(f"\n{'='*60}")
    print(f"✓ Index built successfully!")
    print(f"  Total images indexed: {len(features_list)}")
    print(f"  Failed to process: {failed}")
    print(f"  Index size: {index.get_index_size()}")
    print(f"{'='*60}")
    
    return True


if __name__ == '__main__':
    success = build_dataset_index()
    sys.exit(0 if success else 1)
