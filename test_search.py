#!/usr/bin/env python3
"""Test script to debug search endpoint."""

import sys
import os

# Add project to path
sys.path.insert(0, '/workspaces/image-retrieval-system-based-on-machine-learning')

from app import app
import json

# Create a test client
client = app.test_client()

print("Testing /health endpoint...")
response = client.get('/health')
print(f"Status: {response.status_code}")
print(f"Data: {response.get_json()}")
print()

# First, upload a test image
print("Creating a test image...")
from PIL import Image
import numpy as np

# Create a simple test image
img = Image.new('RGB', (224, 224), color=(73, 109, 137))
test_image_path = '/tmp/test_image.jpg'
img.save(test_image_path)

print("Uploading test image...")
with open(test_image_path, 'rb') as f:
    response = client.post('/upload', data={'file': f})

print(f"Upload Status: {response.status_code}")
print(f"Upload Response: {response.get_json()}")
print()

# Try searching
print("Testing /search endpoint...")
search_data = {
    'filename': 'test_image.jpg',
    'query': 'cat'
}

response = client.post('/search', 
                       data=json.dumps(search_data),
                       content_type='application/json')

print(f"Search Status: {response.status_code}")
print(f"Search Response Type: {type(response.data)}")
print(f"Search Response Raw: {response.data[:500]}")  # First 500 bytes

try:
    data = response.get_json()
    print(f"Search JSON: {data}")
except Exception as e:
    print(f"JSON Parse Error: {e}")
