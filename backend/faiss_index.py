import faiss
import numpy as np
import json
import os
from config import FAISS_INDEX_PATH, METADATA_PATH, FEATURE_DIMENSION


class FAISSIndex:
    """FAISS index for similarity search."""
    
    def __init__(self):
        self.index = None
        self.metadata = {}
        self.dimension = FEATURE_DIMENSION
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        # Create L2 index for cosine similarity
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_features(self, features, image_urls):
        """
        Add features to the index.
        
        Args:
            features: numpy array of shape (n_samples, dimension)
            image_urls: list of image URLs corresponding to features
        """
        if len(features) == 0:
            return
        
        # Ensure features are float32
        features = np.asarray(features, dtype=np.float32)
        
        # Get current index size
        current_size = self.index.ntotal
        
        # Add to index
        self.index.add(features)
        
        # Update metadata
        for i, url in enumerate(image_urls):
            self.metadata[current_size + i] = {
                'url': url,
                'index': current_size + i
            }
    
    def search(self, query_features, k=10):
        """
        Search for similar images.
        
        Args:
            query_features: numpy array of shape (dimension,)
            k: number of results to return
            
        Returns:
            list of tuples (url, similarity_score)
        """
        if self.index.ntotal == 0:
            return []
        
        query_features = np.asarray([query_features], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_features, min(k, self.index.ntotal))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Convert L2 distance to cosine similarity
            # For L2 distance on normalized vectors: similarity = 1 - (distance^2 / 2)
            similarity = max(0, 1 - (distance ** 2) / 2)
            
            metadata = self.metadata.get(int(idx), {})
            url = metadata.get('url', '')
            
            results.append({
                'url': url,
                'similarity': float(similarity),
                'distance': float(distance)
            })
        
        return results
    
    def clear(self):
        """Clear the index."""
        self.index = None
        self.metadata = {}
        self._initialize_index()
    
    def get_index_size(self):
        """Get current index size."""
        return self.index.ntotal
    
    def save(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w') as f:
            # Convert keys to strings for JSON serialization
            metadata_str = {str(k): v for k, v in self.metadata.items()}
            json.dump(metadata_str, f)
    
    def load(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        """Load index and metadata from disk."""
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self._initialize_index()
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_str = json.load(f)
                self.metadata = {int(k): v for k, v in metadata_str.items()}
        else:
            self.metadata = {}
