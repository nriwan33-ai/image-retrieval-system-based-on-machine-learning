"""CBIR system using FAISS index with local dataset."""

from .faiss_index import FAISSIndex
from .feature_extractor import FeatureExtractor
from config import FAISS_INDEX_PATH, METADATA_PATH, FINAL_RESULTS
import os


class DuckDuckGoCBIR:
    """Content-Based Image Retrieval system using local dataset."""
    
    def __init__(self):
        """Initialize CBIR system with FAISS index."""
        self.faiss_index = FAISSIndex()
        self.faiss_index.load(FAISS_INDEX_PATH, METADATA_PATH)
        self.feature_extractor = FeatureExtractor()
        index_size = self.faiss_index.get_index_size()
        print(f"✓ CBIR initialized with {index_size} indexed images")
    
    def search_similar_images(self, query_features, k=FINAL_RESULTS):
        """
        Search for similar images in the index.
        
        Args:
            query_features: numpy array of query image features
            k: number of results to return
            
        Returns:
            list of dictionaries with 'url' and 'similarity' keys
        """
        results = self.faiss_index.search(query_features, k=k)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Format results
        formatted_results = [
            {
                'url': result['url'],
                'similarity': round(result['similarity'], 3)
            }
            for result in results
        ]
        
        return formatted_results[:k]
    
    def get_index_size(self):
        """Get number of images in the index."""
        return self.faiss_index.get_index_size()
    
    def reset_index(self):
        """Clear the FAISS index."""
        self.faiss_index.clear()
        self.faiss_index.clear()


