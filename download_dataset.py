#!/usr/bin/env python3
"""Download dataset using public domain images from Wikimedia and synthetic fallback."""

import os
import urllib.request
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import DATASET_FOLDER, DATASET_CATEGORIES

# High-quality public domain images from Wikimedia Commons
# Using images that don't require authentication
IMAGE_SOURCES = {
    'cars': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/2024_Honda_Accord.jpg/1280px-2024_Honda_Accord.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/2023_Toyota_Camry_%28XV70%29%2C_front_9.9.23.jpg/1280px-2023_Toyota_Camry_%28XV70%29%2C_front_9.9.23.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/2024_Tesla_Model_Y%2C_front_8.27.23.jpg/1280px-2024_Tesla_Model_Y%2C_front_8.27.23.jpg',
    ],
    'motorbikes': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Harley-Davidson_FLSTF_Fat_Boy_2007.jpg/1280px-Harley-Davidson_FLSTF_Fat_Boy_2007.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Kawasaki_Ninja_H2_2015%2C_front_left.jpg/1280px-Kawasaki_Ninja_H2_2015%2C_front_left.jpg',
    ],
    'pandas': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/1024px-Grosser_Panda.JPG',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Giant_Panda_2004-03-2.jpg/1024px-Giant_Panda_2004-03-2.jpg',
    ],
    'laptops': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/MacBook_Pro_15%27_with_Retina_display_2013.png/1024px-MacBook_Pro_15%27_with_Retina_display_2013.png',
    ],
    'orange': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Oranges_-_whole-_along_-_and_piece_05.jpg/1280px-Oranges_-_whole-_along_-_and_piece_05.jpg',
    ],
    'burger': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Receita_Burguer_Artesanal.jpg/1280px-Receita_Burguer_Artesanal.jpg',
    ],
    'jeans': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Jeans_%286263824654%29.jpg/1024px-Jeans_%286263824654%29.jpg',
    ],
    'xrays': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/SotxRayHand.jpg/1024px-SotxRayHand.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Chest_X-ray_PA_view.jpg/1024px-Chest_X-ray_PA_view.jpg',
    ],
    'dogs': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1024px-YellowLabradorLooking_new.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1280px-Collage_of_Nine_Dogs.jpg',
    ],
    'manchester_united_jersey': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Manchester_United_FC_kit_2022-23.svg/800px-Manchester_United_FC_kit_2022-23.svg.png',
    ]
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def create_directories():
    """Create dataset directories for each category."""
    for category in DATASET_CATEGORIES:
        category_path = os.path.join(DATASET_FOLDER, category)
        os.makedirs(category_path, exist_ok=True)
        print(f"✓ Created directory: {category}")


def generate_synthetic_image(category, index):
    """Generate a synthetic image for a category."""
    category_path = os.path.join(DATASET_FOLDER, category)
    filename = f"{category}_{index:03d}.jpg"
    filepath = os.path.join(category_path, filename)
    
    # Create synthetic image with category label
    width, height = 256, 256
    
    # Generate random color based on category hash
    h = hash(f"{category}_{index}") % 256
    colors = [
        (h, h+50, h+100),
        (h+100, h, h+50),
        (h+50, h+100, h),
        (h, h+50, h),
    ]
    bg_color = colors[index % len(colors)]
    
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add text label
    text = f"{category}"
    try:
        draw.text((20, 20), text, fill=(255, 255, 255))
    except:
        pass
    
    # Add some random shapes
    for i in range(5):
        x = (index * 17 + i * 23) % (width - 30)
        y = (index * 13 + i * 19) % (height - 30)
        size = 10 + (index * i) % 20
        draw.ellipse([x, y, x+size, y+size], fill=(255-h, 255-h//2, 255))
    
    img.save(filepath, 'JPEG', quality=85)
    return filepath


def download_image(url, category, index):
    """Download image from URL."""
    category_path = os.path.join(DATASET_FOLDER, category)
    filename = f"{category}_{index:03d}.jpg"
    filepath = os.path.join(category_path, filename)
    
    try:
        urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=5).read()
        urllib.request.urlretrieve(url, filepath)
        
        # Validate image
        img = Image.open(filepath).convert('RGB')
        w, h = img.size
        if w < 100 or h < 100:
            os.remove(filepath)
            return False
        
        # Resize if too large
        if w > 1000 or h > 1000:
            img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            img.save(filepath, 'JPEG', quality=85)
        
        return True
    except Exception as e:
        return False


def build_dataset():
    """Build dataset with real and synthetic images."""
    print("="*60)
    print("Building Dataset (Real + Synthetic Images)")
    print("="*60)
    
    create_directories()
    
    total_images = 0
    
    for category in DATASET_CATEGORIES:
        print(f"\n{category.upper()}")
        print("-" * 40)
        
        downloaded = 0
        urls = IMAGE_SOURCES.get(category, [])
        
        # Try downloading real images first
        for idx, url in enumerate(urls):
            if download_image(url, category, downloaded):
                print(f"  ✓ Downloaded image {downloaded+1}")
                downloaded += 1
                time.sleep(0.5)
        
        # Fill remaining with synthetic images
        for idx in range(downloaded, 50):
            generate_synthetic_image(category, idx)
            print(f"  ✓ Generated synthetic image {idx+1}")
        
        total_images += 50
        print(f"  Completed: {category} ({50} images)")
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset built successfully!")
    print(f"  Total images: {total_images}")
    print(f"{'='*60}")


if __name__ == '__main__':
    build_dataset()
