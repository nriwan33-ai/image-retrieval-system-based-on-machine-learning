import requests
from PIL import Image
from io import BytesIO
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .feature_extractor import FeatureExtractor
from .faiss_index import FAISSIndex
from config import MAX_RESULTS_TO_FETCH, FINAL_RESULTS, DUCKDUCKGO_TIMEOUT


class DuckDuckGoCBIR:
    """DuckDuckGo Content-Based Image Retrieval system using FAISS."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.faiss_index = FAISSIndex()
    
    def search_duckduckgo_images(self, query, num_results=MAX_RESULTS_TO_FETCH):
        """
        Search for images on DuckDuckGo.
        
        Args:
            query: search query string
            num_results: number of results to fetch
            
        Returns:
            list of image URLs
        """
        try:
            # DuckDuckGo image search endpoint
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            image_urls = []
            
            # Try using DuckDuckGo's API
            url = f"https://duckduckgo.com/api/add_search"
            params = {
                'q': query,
                'ia': 'images'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=DUCKDUCKGO_TIMEOUT)
            
            if response.status_code == 200:
                # Parse JSON response for image URLs
                try:
                    data = response.json()
                    if 'results' in data:
                        for result in data['results']:
                            if 'image' in result:
                                image_urls.append(result['image'])
                except:
                    pass
            
            # Fallback: Use simple scraping
            if len(image_urls) < num_results:
                image_urls.extend(self._scrape_duckduckgo_images(query, num_results))
            
            return list(set(image_urls[:num_results]))
        except Exception as e:
            print(f"Error searching DuckDuckGo: {str(e)}")
            return []
    
    def _scrape_duckduckgo_images(self, query, num_results):
        """Scrape image URLs from DuckDuckGo search page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # DuckDuckGo image search URL
            search_url = f"https://duckduckgo.com/?q={query}&iax=images&ia=images"
            response = requests.get(search_url, headers=headers, timeout=DUCKDUCKGO_TIMEOUT)
            
            if response.status_code != 200:
                return []
            
            # Simple regex to extract image URLs from HTML
            import re
            image_pattern = r'"image":\s*"([^"]+)"'
            matches = re.findall(image_pattern, response.text)
            
            # Decode escaped URLs
            urls = []
            for match in matches:
                try:
                    # DuckDuckGo encodes URLs, attempt to decode
                    decoded_url = match.replace('\\/', '/')
                    if decoded_url.startswith('http'):
                        urls.append(decoded_url)
                except:
                    pass
            
            return urls[:num_results]
        except Exception as e:
            print(f"Error scraping DuckDuckGo: {str(e)}")
            return []
    
    def _fetch_and_validate_image(self, url):
        """
        Fetch image from URL and validate it.
        
        Returns:
            tuple (url, features) or (url, None) if invalid
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            # Validate image
            image = Image.open(BytesIO(response.content))
            image.verify()
            
            # Extract features
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Use feature extractor to get features
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name, 'JPEG')
                features = self.feature_extractor.extract_features(tmp.name)
                import os
                os.unlink(tmp.name)
            
            return (url, features)
        except Exception as e:
            print(f"Error fetching/validating image from {url}: {str(e)}")
            return (url, None)
    
    def search_similar_images(self, query_features, k=FINAL_RESULTS):
        """
        Search for similar images from indexed CBIR data.
        
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
    
    def index_images_from_query(self, query_text):
        """
        Search DuckDuckGo for images and index them.
        
        Args:
            query_text: search query
            
        Returns:
            number of images indexed
        """
        try:
            print(f"Searching DuckDuckGo for: {query_text}")
            image_urls = self.search_duckduckgo_images(query_text, MAX_RESULTS_TO_FETCH)
            
            if not image_urls:
                print("No images found")
                return 0
            
            print(f"Found {len(image_urls)} images, processing...")
            
            # Fetch and process images in parallel
            valid_features = []
            valid_urls = []
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._fetch_and_validate_image, url): url 
                    for url in image_urls
                }
                
                for future in as_completed(futures):
                    url, features = future.result()
                    if features is not None:
                        valid_features.append(features)
                        valid_urls.append(url)
            
            print(f"Successfully processed {len(valid_features)} images")
            
            # Add to FAISS index
            if valid_features:
                self.faiss_index.add_features(valid_features, valid_urls)
                print(f"Index size: {self.faiss_index.get_index_size()}")
            
            return len(valid_features)
        except Exception as e:
            print(f"Error indexing images: {str(e)}")
            return 0
    
    def reset_index(self):
        """Clear the FAISS index."""
        self.faiss_index.clear()
