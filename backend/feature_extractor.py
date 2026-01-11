import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import numpy as np
from config import DEVICE, MODEL_NAME


class FeatureExtractor:
    """Extract features from images using VGG19 pre-trained model."""
    
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        self.model = self._load_model()
        self.transform = self._get_transforms()
    
    def _load_model(self):
        """Load pre-trained VGG19 model."""
        model = vgg19(pretrained=True)
        # Remove the classification head, keep only feature extraction layers
        model = torch.nn.Sequential(*list(model.features.children()))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_path):
        """
        Extract features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of features (4096-dim vector)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                # Global average pooling
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                # L2 normalization
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            raise
    
    def extract_features_from_url(self, image_url):
        """
        Extract features from an image URL.
        
        Args:
            image_url: URL of the image
            
        Returns:
            numpy array of features (4096-dim vector)
        """
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess and extract
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"Error extracting features from URL {image_url}: {str(e)}")
            return None
