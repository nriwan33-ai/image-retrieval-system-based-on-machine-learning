from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, SECRET_KEY, DEBUG
from backend.feature_extractor import FeatureExtractor
from backend.duckduckgo_cbir import DuckDuckGoCBIR


app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize components
feature_extractor = FeatureExtractor()
cbir_system = DuckDuckGoCBIR()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        features = feature_extractor.extract_features(filepath)
        
        # Return success with features
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'/uploads/{filename}',
            'features_shape': features.shape,
            'message': 'Image uploaded successfully'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_similar():
    """Search for similar images from DuckDuckGo."""
    try:
        data = request.get_json()
        
        if 'filename' not in data:
            return jsonify({'error': 'No image specified'}), 400
        
        filename = secure_filename(data['filename'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Extract features from uploaded image
        query_features = feature_extractor.extract_features(filepath)
        
        # Get image description/query (use extracted keywords or default)
        query_text = data.get('query', 'similar images')
        
        # Index images from DuckDuckGo
        num_indexed = cbir_system.index_images_from_query(query_text)
        
        if num_indexed == 0:
            return jsonify({
                'error': 'No images found on DuckDuckGo for this search',
                'results': []
            }), 200
        
        # Search for similar images in the index
        results = cbir_system.search_similar_images(query_features, k=10)
        
        return jsonify({
            'success': True,
            'results': results,
            'indexed_count': num_indexed,
            'message': f'Found {len(results)} similar images'
        }), 200
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def get_upload(filename):
    """Serve uploaded files."""
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': 'VGG19+FAISS+DuckDuckGo'}), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File is too large. Maximum size is 50MB'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=DEBUG,
        threaded=True
    )
