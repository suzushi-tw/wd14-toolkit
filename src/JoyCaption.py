from PIL import Image
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Please install it with: pip install llama-cpp-python")


class JoyCaptioner:
    """JoyCaption using llama-cpp-python with GGUF models for efficient inference"""
    
    CAPTION_TYPE_MAP = {
        "Descriptive": [
            "Write a descriptive caption for this image in a formal tone.",
            "Write a descriptive caption for this image in a formal tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a formal tone.",
        ],
        "Descriptive (Informal)": [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
        "Training Prompt": [
            "Write a stable diffusion prompt for this image.",
            "Write a stable diffusion prompt for this image within {word_count} words.",
            "Write a {length} stable diffusion prompt for this image.",
        ],
        "MidJourney": [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
        "Booru tag list": [
            "Write a list of Booru tags for this image.",
            "Write a list of Booru tags for this image within {word_count} words.",
            "Write a {length} list of Booru tags for this image.",
        ],
        "Booru-like tag list": [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
        "Art Critic": [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
        "Product Listing": [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
        "Social Media Post": [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    }

    def __init__(
        self, 
        model_repo="mradermacher/llama-joycaption-beta-one-hf-llava-GGUF",
        model_file="llama-joycaption-beta-one-hf-llava.Q4_K_M.gguf",
        mmproj_repo="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf", 
        mmproj_file="llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
        n_ctx=2048,  # Reduced from 4096
        n_gpu_layers=32  # Increased from 20 - you have headroom with 8GB
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required. Install it with: pip install llama-cpp-python")
        
        self.model_repo = model_repo
        self.model_file = model_file
        self.mmproj_repo = mmproj_repo
        self.mmproj_file = mmproj_file
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        
        self.model = None
        self.chat_handler = None
        self.model_path = None
        self.mmproj_path = None
    
    def _download_model_files(self, token=None):
        """Download GGUF model and mmproj files from Hugging Face if not cached"""
        print("Checking for model files...")
        
        try:
            # Download the main GGUF model
            print(f"Downloading GGUF model: {self.model_file}")
            self.model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                token=token
            )
            print(f"✓ Model downloaded to: {self.model_path}")
            
            # Download the mmproj file (vision encoder)
            print(f"Downloading mmproj file: {self.mmproj_file}")
            self.mmproj_path = hf_hub_download(
                repo_id=self.mmproj_repo,
                filename=self.mmproj_file,
                token=token
            )
            print(f"✓ mmproj downloaded to: {self.mmproj_path}")
            
        except Exception as e:
            print(f"Error downloading model files: {e}")
            raise
        
    def load_model(self, token=None):
        """Load the GGUF model with llama-cpp-python"""
        if self.model is not None:
            return

        print(f"Loading JoyCaption GGUF model with 4-bit quantization")
        
        try:
            # Download model files if needed
            if self.model_path is None or self.mmproj_path is None:
                self._download_model_files(token=token)
            
            # Initialize the chat handler with the vision projector
            print("Initializing LLaVA chat handler with mmproj...")
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=self.mmproj_path,
                verbose=False
            )
            
            # Load the GGUF model
            print(f"Loading GGUF model with {self.n_gpu_layers} GPU layers...")
            self.model = Llama(
                model_path=self.model_path,
                chat_handler=self.chat_handler,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                logits_all=True,
                verbose=False,
                n_batch=512,  # Add batch size for better memory management
            )
            
            print(f"✓ Model loaded successfully (using {self.n_gpu_layers} GPU layers, ~3-8GB VRAM depending on layers)")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def unload_model(self):
        """Unload the model and free memory"""
        if self.model is not None:
            del self.model
            del self.chat_handler
            self.model = None
            self.chat_handler = None
            print("JoyCaption model unloaded.")
    def predict(self, image, tags=None, caption_type="Descriptive", caption_length="long", custom_prompt=None):
        """
        Generate a caption for an image using JoyCaption.
        
        Args:
            image: PIL Image or path to image file
            tags: Optional tags to condition the caption (not used in this implementation)
            caption_type: Type of caption from CAPTION_TYPE_MAP
            caption_length: Caption length ("any", "very short", "short", "medium-length", "long", "very long") 
                          or a number as string for word count
            custom_prompt: Optional custom prompt (overrides all other settings)
        
        Returns:
            Generated caption string
        """
        if self.model is None:
            self.load_model()

        if self.model is None:
            raise RuntimeError("Failed to load JoyCaption model.")

        # Handle image input
        if isinstance(image, str):
            image = Image.open(image)
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build the prompt following HF Space logic
        if custom_prompt and custom_prompt.strip():
            prompt_str = custom_prompt.strip()
        else:
            # Determine which prompt template to use based on length
            length = None if caption_length == "any" else caption_length
            
            if isinstance(length, str):
                try:
                    length = int(length)
                except ValueError:
                    pass
            
            # Select the correct prompt template
            if length is None:
                map_idx = 0
            elif isinstance(length, int):
                map_idx = 1
            else:  # descriptive length like "short", "long", etc.
                map_idx = 2
            
            prompt_str = self.CAPTION_TYPE_MAP[caption_type][map_idx]
            
            # Format the prompt with length/word_count
            prompt_str = prompt_str.format(length=caption_length, word_count=caption_length)

        # Build the conversation following HF Space format
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        # For llama-cpp-python with LLaVA, format message with image
        messages = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_to_data_uri(image)}},
                    {"type": "text", "text": prompt_str}
                ]
            },
        ]

        # Generate the caption using the same parameters as HF Space
        # HF Space uses: max_new_tokens=300, do_sample=True, temp=0.6 (default), top_p=0.9 (default)
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=300,  # Match HF Space
                temperature=0.6,
                top_p=0.9,
                stop=["<|eot_id|>", "\nASSISTANT:", "\nUSER:", "ASSISTANT:", "USER:"],  # Stop at end of turn or role markers
            )
            
            caption = response['choices'][0]['message']['content']
            
            # Clean up the caption
            caption = caption.strip()
            
            # Remove any leaked role markers or conversation artifacts
            for marker in ["ASSISTANT:", "USER:", "<|eot_id|>", "<|im_end|>", "<|end|>"]:
                if marker in caption:
                    caption = caption.split(marker)[0].strip()
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            raise
        except Exception as e:
            print(f"Error generating caption: {e}")
            raise
    
    def _image_to_data_uri(self, image):
        """Convert PIL Image to data URI for llama-cpp-python"""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

