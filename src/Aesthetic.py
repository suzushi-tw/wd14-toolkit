import torch
from PIL import Image
import logging
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

class AestheticScorer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=30)
        self.model = None
        self.preprocessor = None
        
        # Quality mapping for V2.5 scores
        self.quality_mapping = {
            (7.5, 10.0): ("masterpiece", "best quality"),
            (6.5, 7.5): ("high quality", "aesthetic"),
            (5.5, 6.5): ("good quality", "aesthetic"),
            (4.5, 5.5): ("normal quality", "average"),
            (3.5, 4.5): ("low quality", "unaesthetic"),
            (0.0, 3.5): ("worst quality", "unaesthetic")
        }
        
    def load_model(self):
        if self.model is None:
            try:
                self.model, self.preprocessor = convert_v2_5_from_siglip(
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                if torch.cuda.is_available():
                    self.model = self.model.to(torch.bfloat16).cuda()
            except Exception as e:
                self.logger.error(f"Error loading V2.5 model: {str(e)}")

    def get_quality_tag(self, score):
        for (min_score, max_score), (quality, aesthetic) in self.quality_mapping.items():
            if min_score <= score <= max_score:
                return quality, aesthetic
        return "unknown quality", "unknown"

    def get_score(self, image):
        if image is None:
            return 0.0, "unknown quality", "unknown"
            
        if self.model is None:
            self.load_model()
            
        try:
            pixel_values = (
                self.preprocessor(images=image, return_tensors="pt")
                .pixel_values.to(torch.bfloat16)
                .cuda() if torch.cuda.is_available() else 
                self.preprocessor(images=image, return_tensors="pt")
                .pixel_values
            )
            
            with torch.inference_mode():
                score = self.model(pixel_values).logits.squeeze().float().cpu().numpy()
            
            quality, aesthetic = self.get_quality_tag(float(score))
            self.logger.info(f"Score: {score:.4f}, Quality: {quality}")
            
            return float(score), quality, aesthetic
            
        except Exception as e:
            self.logger.error(f"Error during scoring: {str(e)}")
            return 0.0, "error", "error"