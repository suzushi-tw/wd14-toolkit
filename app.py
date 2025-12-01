import argparse
import os
from dotenv import load_dotenv

import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

from src.validator import ImageValidator 
from src.Aesthetic import AestheticScorer
from src.JoyCaption import JoyCaptioner
from src.PixAI import PixAITagger
from src.config.config import dropdown_list, SWINV2_MODEL_DSV3_REPO, PIXAI_TAGGER_V09_REPO
from frontend import launch_interface

# Load environment variables from .env file
load_dotenv()

TITLE = "WaifuDiffusion Tagger"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    return parser.parse_args()

def load_labels(dataframe) -> tuple[list[str], list[int], list[int], list[int]]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

class Predictor:
    """Unified predictor for ONNX and PyTorch models"""
    
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None
        self.model = None
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []
        
        # PixAI tagger instance (lazy loaded)
        self.pixai_tagger = None

    def is_pixai_model(self, model_repo):
        """Check if the model is a PixAI PyTorch model"""
        return "pixai" in model_repo.lower()

    def load_model(self, model_repo):
        """Load model (either ONNX or PixAI PyTorch)"""
        if model_repo == self.last_loaded_repo:
            return

        if self.is_pixai_model(model_repo):
            # Load PixAI model
            if self.pixai_tagger is None:
                self.pixai_tagger = PixAITagger(model_repo)
            self.pixai_tagger.load_model(token=HF_TOKEN)
            self.last_loaded_repo = model_repo
        else:
            # Load ONNX model
            import huggingface_hub
            
            self.model_target_size = 448
            
            # Download files
            token = HF_TOKEN if HF_TOKEN else True
            csv_path = huggingface_hub.hf_hub_download(
                model_repo,
                LABEL_FILENAME,
                token=token,
            )
            model_path = huggingface_hub.hf_hub_download(
                model_repo,
                MODEL_FILENAME,
                token=token,
            )
            
            # Load tags
            tags_df = pd.read_csv(csv_path)
            sep_tags = load_labels(tags_df)

            self.tag_names = sep_tags[0]
            self.rating_indexes = sep_tags[1]
            self.general_indexes = sep_tags[2]
            self.character_indexes = sep_tags[3]

            # Load ONNX model
            available_providers = rt.get_available_providers()
            print(f"Available providers: {available_providers}")

            if 'CUDAExecutionProvider' in available_providers:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ]
            else:
                providers = ['CPUExecutionProvider']

            self.model = rt.InferenceSession(model_path, providers=providers)
            self.last_loaded_repo = model_repo
    
    def prepare_image(self, image):
        """Prepare image for ONNX inference"""
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

        # Pad image to square
        target_size = self.model_target_size
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.Resampling.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        """Predict tags for an image"""
        self.load_model(model_repo)

        if self.is_pixai_model(model_repo):
            # Use PixAI tagger
            result = self.pixai_tagger.predict(
                image,
                general_threshold=general_thresh,
                character_threshold=character_thresh,
            )
            
            # Format output to match expected format
            sorted_general_string, character_str, general_str = self.pixai_tagger.format_tags(
                result['general_tags'],
                result['character_tags'],
                include_scores=True
            )
            
            # PixAI doesn't have ratings, return empty
            rating_str = "No rating available for PixAI model"
            
            return sorted_general_string, rating_str, character_str, general_str
        else:
            # ONNX inference
            image_array = self.prepare_image(image)
            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name
            preds = self.model.run([label_name], {input_name: image_array})[0][0]

            labels = list(zip(self.tag_names, preds.astype(float)))

            # Ratings
            ratings_names = [labels[i] for i in self.rating_indexes]
            rating = dict(ratings_names)

            # General tags
            general_names = [labels[i] for i in self.general_indexes]

            if general_mcut_enabled:
                general_probs = np.array([x[1] for x in general_names])
                general_thresh = mcut_threshold(general_probs)

            general_res = [x for x in general_names if x[1] > general_thresh]
            general_res = dict(general_res)

            # Character tags
            character_names = [labels[i] for i in self.character_indexes]

            if character_mcut_enabled:
                character_probs = np.array([x[1] for x in character_names])
                character_thresh = mcut_threshold(character_probs)
                character_thresh = max(0.15, character_thresh)

            character_res = [x for x in character_names if x[1] > character_thresh]
            character_res = dict(character_res)

            # Format output
            sorted_general_strings = sorted(
                general_res.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_general_strings = [x[0] for x in sorted_general_strings]
            sorted_general_strings = (
                ", ".join(sorted_general_strings).replace("(", "\\(").replace(")", "\\)")
            )

            rating_str = ", ".join([f"{k}: {v:.3f}" for k, v in rating.items()])
            character_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted(character_res.items(), key=lambda x: x[1], reverse=True)])
            general_str = ", ".join([f"{k}: {v:.3f}" for k, v in sorted(general_res.items(), key=lambda x: x[1], reverse=True)])

            return sorted_general_strings, rating_str, character_str, general_str 

def main():
    args = parse_args()
    predictor = Predictor()
    validator = ImageValidator()
    aesthetic_scorer = AestheticScorer()
    joy_captioner = JoyCaptioner()

    launch_interface(
        predictor, 
        validator, 
        aesthetic_scorer,
        joy_captioner,
        args
    )

if __name__ == "__main__":
    main()