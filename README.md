# Content-Based Image Retrieval System

A Flask-based CBIR (Content-Based Image Retrieval) system that uses VGG19 for feature extraction, FAISS for similarity search, and DuckDuckGo as the image source.

## Features

- **VGG19 Feature Extraction**: Uses pre-trained VGG19 to extract 4096-dimensional feature vectors from images
- **FAISS Indexing**: Efficient similarity search using Facebook's FAISS library
- **DuckDuckGo Integration**: Real-time image fetching from DuckDuckGo search
- **Cosine Similarity**: Ranks results using cosine similarity scores (0.000 to 1.000)
- **Web Interface**: Modern, responsive UI with drag-and-drop upload
- **No Local Dataset**: Fetches and indexes images on-the-fly from DuckDuckGo

## System Architecture

```
┌─────────────────────┐
│  Web Interface      │  (HTML/CSS/JS)
├─────────────────────┤
│  Flask Backend      │  (app.py)
├─────────────────────┤
│  Feature Extractor  │  (VGG19)
├─────────────────────┤
│  FAISS Index        │  (Similarity Search)
├─────────────────────┤
│  DuckDuckGo Engine  │  (Image Fetching)
└─────────────────────┘
```

## Project Structure

```
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
├── backend/
│   ├── __init__.py                # Package initialization
│   ├── feature_extractor.py       # VGG19 feature extraction
│   ├── faiss_index.py             # FAISS index management
│   └── duckduckgo_cbir.py         # DuckDuckGo CBIR engine
├── static/
│   ├── css/
│   │   └── style.css              # Styling
│   ├── js/
│   │   └── app.js                 # Frontend logic
│   └── uploads/                   # Uploaded images (temporary)
└── templates/
    └── index.html                 # Main HTML page
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone or download the project**:
```bash
cd /workspaces/image-retrieval-system-based-on-machine-learning
```

2. **Create a virtual environment** (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Interface

1. **Upload Image**: Drag and drop an image into the upload area or click to browse
2. **Preview**: Image preview appears immediately after upload
3. **Search**: Click "Search Similar Images" button
4. **Results**: View 10 most similar images with cosine similarity scores (0.000-1.000)

### API Endpoints

#### Upload Image
- **POST** `/upload`
- **Parameters**: `file` (image file)
- **Response**: `{ success: true, filename: string, filepath: string }`

#### Search Similar Images
- **POST** `/search`
- **Body**: `{ filename: string, query: string }`
- **Response**: 
```json
{
  "success": true,
  "results": [
    {
      "url": "image_url",
      "similarity": 0.945
    }
  ],
  "indexed_count": 50
}
```

#### Health Check
- **GET** `/health`
- **Response**: `{ status: "ok", model: "VGG19+FAISS+DuckDuckGo" }`

## Configuration

Edit `config.py` to customize:

```python
# Model Settings
MODEL_NAME = 'vgg19'              # Feature extraction model
FEATURE_DIMENSION = 4096          # Feature vector dimension
DEVICE = 'cpu'                    # Use 'cuda' for GPU

# Upload Settings
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # Max file size (50MB)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

# Search Settings
MAX_RESULTS_TO_FETCH = 50         # Images to fetch from DuckDuckGo
FINAL_RESULTS = 10                # Results to display
```

## Technical Details

### Feature Extraction
- **Model**: VGG19 (pre-trained on ImageNet)
- **Input Size**: 224×224 RGB images
- **Feature Vector**: 4096-dimensional
- **Normalization**: L2 normalization for cosine similarity

### Similarity Search
- **Algorithm**: FAISS IndexFlatL2
- **Metric**: Cosine similarity (converted from L2 distance)
- **Similarity Range**: 0.000 (dissimilar) to 1.000 (identical)

### Image Source
- **Provider**: DuckDuckGo
- **Fetching Method**: Real-time search and download
- **Validation**: Image format and size validation
- **Parallel Processing**: Multi-threaded image fetching

## Performance

- **Feature Extraction**: ~100-500ms per image (CPU)
- **FAISS Search**: <10ms for 50 images
- **DuckDuckGo Fetch**: 5-15 seconds for 50 images
- **Total Search Time**: 10-20 seconds per query

## Requirements

### Python Packages
- Flask 2.3.0 - Web framework
- PyTorch 2.0.0 - Deep learning framework
- torchvision 0.15.0 - Computer vision utilities
- FAISS 1.7.4 - Similarity search library
- Pillow 10.0.0 - Image processing
- requests 2.31.0 - HTTP library
- beautifulsoup4 4.12.0 - HTML parsing

## Troubleshooting

### Issue: Model download takes long
- **Solution**: VGG19 model is downloaded automatically on first run. This may take a few minutes.

### Issue: DuckDuckGo images not loading
- **Solution**: Some images may fail to load. The system gracefully handles errors and continues with valid images.

### Issue: High memory usage
- **Solution**: Reduce `MAX_RESULTS_TO_FETCH` in config.py

### Issue: Slow search on CPU
- **Solution**: For GPU acceleration, install `faiss-gpu` and set `DEVICE = 'cuda'` in config.py

## Browser Compatibility

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Limitations

- Search time depends on DuckDuckGo response time
- Image quality varies based on DuckDuckGo results
- No persistent index between sessions (optional: implement save/load)
- Single-user testing recommended for deployment

## Future Enhancements

- [ ] Persistent FAISS index saving/loading
- [ ] GPU acceleration support
- [ ] Additional models (ResNet, EfficientNet)
- [ ] Batch search support
- [ ] Image upload history
- [ ] Search result caching
- [ ] Mobile app version
- [ ] Advanced filtering options

## License

This project is provided as-is for educational and research purposes.

## References

- VGG19: https://arxiv.org/abs/1409.1556
- FAISS: https://github.com/facebookresearch/faiss
- DuckDuckGo: https://duckduckgo.com/

## Contact & Support

For issues or questions, please refer to the project documentation.

---

**Built with**: Flask, PyTorch, FAISS, DuckDuckGo  
**Last Updated**: January 2024