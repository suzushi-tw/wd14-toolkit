import numpy as np
from PIL import Image
import logging

class ColorAnalyzer:
    def __init__(self):
        self.grayscale_threshold = 0.05  # RGB difference threshold
        self.logger = logging.getLogger(__name__)
        
    def is_grayscale(self, image_path: str) -> bool:
        """Check if image is grayscale/B&W"""
        try:
            # Load image from path
            with Image.open(image_path) as image:
                if image.mode == "L":  # Already grayscale
                    return True
                
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Convert to numpy array
                img_array = np.array(image)
                
                if len(img_array.shape) < 3:  # Not an RGB image
                    return True
                    
                # Check if R,G,B channels are similar
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                
                # Check if the channels are very similar
                diff_rg = np.abs(r - g).mean()
                diff_rb = np.abs(r - b).mean()
                diff_gb = np.abs(g - b).mean()
                
                return (diff_rg < self.grayscale_threshold * 255 and
                        diff_rb < self.grayscale_threshold * 255 and
                        diff_gb < self.grayscale_threshold * 255)
                        
        except Exception as e:
            self.logger.error(f"Error analyzing {image_path}: {e}")
            return False