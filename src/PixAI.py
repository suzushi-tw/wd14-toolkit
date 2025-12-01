"""
PixAI Tagger v0.9 Implementation
Standalone module for both single image and batch processing
"""

import json
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import huggingface_hub
import os


class TaggingHead(nn.Module):
    """Tagging head for PixAI model - matches the exact architecture"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        # Note: head is wrapped in Sequential to match the saved model structure
        self.head = nn.Sequential(nn.Linear(input_dim, num_classes))

    def forward(self, x):
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return probs


class PixAITagger:
    """PixAI Tagger v0.9 - PyTorch-based anime image tagger"""
    
    MODEL_FILENAME = "model_v0.9.pth"
    LABEL_FILENAME = "tags_v0.9_13k.json"
    BASE_MODEL_REPO = "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"
    
    def __init__(self, model_repo="pixai-labs/pixai-tagger-v0.9", device=None):
        """
        Initialize PixAI Tagger
        
        Args:
            model_repo: HuggingFace repository ID
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_repo = model_repo
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tag_names = []
        self.gen_tag_count = 0
        self.character_tag_count = 0
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def load_model(self, token=None):
        """
        Load the PixAI model and tags
        
        Args:
            token: HuggingFace token (if needed for private repos)
        """
        if self.model is not None:
            return  # Already loaded
            
        # Use token from env if not provided
        if token is None:
            token = os.environ.get("HF_TOKEN", True)
        
        # Download model files
        print(f"Downloading PixAI model from {self.model_repo}...")
        label_path = huggingface_hub.hf_hub_download(
            self.model_repo,
            self.LABEL_FILENAME,
            token=token,
        )
        model_path = huggingface_hub.hf_hub_download(
            self.model_repo,
            self.MODEL_FILENAME,
            token=token,
        )
        
        # Load tags
        with open(label_path, 'r', encoding='utf-8') as f:
            tag_info = json.load(f)
        
        tag_map = tag_info["tag_map"]
        tag_split = tag_info["tag_split"]
        self.gen_tag_count = tag_split["gen_tag_count"]
        self.character_tag_count = tag_split["character_tag_count"]
        
        # Create index to tag mapping
        self.tag_names = [''] * len(tag_map)
        for tag, idx in tag_map.items():
            self.tag_names[idx] = tag
        
        # Build model architecture
        print("Building PixAI model architecture...")
        encoder = timm.create_model(self.BASE_MODEL_REPO, pretrained=False)
        encoder.reset_classifier(0)
        
        decoder = TaggingHead(1024, len(self.tag_names))
        self.model = nn.Sequential(encoder, decoder)
        
        # Load weights
        print(f"Loading weights on {self.device}...")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"PixAI Tagger loaded successfully on {self.device}")
    
    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare image for inference
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB
        if image.mode == "RGBA":
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.split()[3])
            image = canvas
        elif image.mode == "P":
            image = image.convert("RGBA")
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.split()[3])
            image = canvas
        else:
            image = image.convert("RGB")
        
        # Apply transform and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(
        self,
        image: Image.Image,
        general_threshold: float = 0.30,
        character_threshold: float = 0.85,
    ) -> dict:
        """
        Predict tags for an image
        
        Args:
            image: PIL Image
            general_threshold: Threshold for general tags
            character_threshold: Threshold for character tags
            
        Returns:
            Dictionary with 'general_tags', 'character_tags', and 'all_probs'
        """
        if self.model is None:
            self.load_model()
        
        # Prepare image
        image_tensor = self.prepare_image(image)
        
        # Run inference with fallback to CPU if CUDA fails
        try:
            with torch.inference_mode():
                probs = self.model(image_tensor)[0].cpu().numpy()
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device == "cuda":
                print(f"PixAI: CUDA error during inference, falling back to CPU: {e}")
                self.device = "cpu"
                self.model = self.model.cpu()
                image_tensor = image_tensor.cpu()
                with torch.inference_mode():
                    probs = self.model(image_tensor)[0].cpu().numpy()
            else:
                raise
        
        # Split into general and character tags
        general_probs = probs[:self.gen_tag_count]
        character_probs = probs[self.gen_tag_count:]
        
        # Filter by threshold
        general_tags = {}
        for idx, prob in enumerate(general_probs):
            if prob > general_threshold:
                tag = self.tag_names[idx]
                general_tags[tag] = float(prob)
        
        character_tags = {}
        for idx, prob in enumerate(character_probs):
            if prob > character_threshold:
                tag = self.tag_names[self.gen_tag_count + idx]
                character_tags[tag] = float(prob)
        
        return {
            'general_tags': general_tags,
            'character_tags': character_tags,
            'all_probs': probs,
        }
    
    def format_tags(
        self,
        general_tags: dict,
        character_tags: dict,
        include_scores: bool = False
    ) -> tuple[str, str, str]:
        """
        Format tags for output
        
        Args:
            general_tags: Dictionary of general tags and scores
            character_tags: Dictionary of character tags and scores
            include_scores: Whether to include scores in output
            
        Returns:
            Tuple of (sorted_general_string, character_string, general_string)
        """
        # Sort by score (descending)
        sorted_general = sorted(general_tags.items(), key=lambda x: x[1], reverse=True)
        sorted_character = sorted(character_tags.items(), key=lambda x: x[1], reverse=True)
        
        if include_scores:
            general_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted_general])
            character_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted_character])
        else:
            general_str = ", ".join([k for k, v in sorted_general])
            character_str = ", ".join([k for k, v in sorted_character])
        
        # Sorted general string (for compatibility with existing code)
        sorted_general_string = ", ".join([k.replace("_", " ") for k, v in sorted_general])
        
        return sorted_general_string, character_str, general_str
