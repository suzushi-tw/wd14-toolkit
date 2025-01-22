from transformers import pipeline
import torch
from PIL import Image
import logging

class AestheticScorer:
    def __init__(self):
        self.model_name = "shadowlilac/aesthetic-shadow-v2"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=30)
        
        # Quality thresholds based on score
        self.quality_mapping = {
            (0.95, 1.00): "masterpiece",
            (0.80, 0.95): "high quality", 
            (0.60, 0.80): "good quality",
            (0.25, 0.60): "normal quality",
            (0.5, 0.25): "low quality",
            (0.00, 0.5): "worst quality"
        }
        
        # Aesthetic thresholds
        self.aesthetic_mapping = {
            (0.71, 1.00): "very aesthetic",
            (0.45, 0.71): "aesthetic",
            (0.27, 0.45): "pleasent",
            (0.00, 0.27): "unpleasent"
        }
        
        # Force CUDA setup
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = 0  # Use first CUDA device
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            self.device = -1  # Use CPU
            self.logger.warning("CUDA not available")
        self.pipe = None

    def get_quality_tag(self, score):
        for (min_score, max_score), tag in self.quality_mapping.items():
            if min_score <= score <= max_score:
                return tag
        return "unknown quality"
        
    def get_aesthetic_tag(self, score):
        for (min_score, max_score), tag in self.aesthetic_mapping.items():
            if min_score <= score <= max_score:
                return tag
        return "unknown aesthetic"

    def get_score(self, image):
        if image is None:
            return 0.0, "unknown", "unknown", "unknown"

        try:
            if self.pipe is None:
                self.pipe = pipeline(
                    "image-classification",
                    model=self.model_name,
                    device=self.device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.logger.info(f"Model loaded on device: {self.pipe.device}")

            result = self.pipe(images=image)
            
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                raw_score = prediction["score"]
                label = prediction["label"]
                final_score = raw_score if label == "hq" else 1 - raw_score
                
                quality_tag = self.get_quality_tag(final_score)
                aesthetic_tag = self.get_aesthetic_tag(final_score)
                
                self.logger.info(f"Raw score: {raw_score:.4f}, Label: {label}, "
                               f"Final score: {final_score:.4f}, Quality: {quality_tag}, "
                               f"Aesthetic: {aesthetic_tag}")
                
                return round(final_score, 4), quality_tag, aesthetic_tag, label.upper()
            
            return 0.0, "unknown", "unknown", "unknown"
            
        except Exception as e:
            self.logger.error(f"Error during scoring: {str(e)}")
            return 0.0, "error", "error", "error"