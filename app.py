import argparse
import os

import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
import glob
from pathlib import Path
from tqdm import tqdm
from batch_process import batch_process
from src.validator import ImageValidator 
from src.Merge import DatasetMerger
from src.Aesthetic import AestheticScorer
from src.Batch_aestheitc import AestheticSorter

TITLE = "WaifuDiffusion Tagger"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# Dataset v2 series of models:
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# IdolSankaku series of models:
EVA02_LARGE_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-eva02-large-tagger-v1"
SWINV2_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-swinv2-tagger-v1"

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

def load_labels(dataframe) -> list[str]:
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
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None

    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
            use_auth_token=HF_TOKEN,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
            use_auth_token=HF_TOKEN,
        )
        return csv_path, model_path

    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)
        
        # Set model target size (this is a key fix)
        self.model_target_size = 448  # or 224, depending on the model
        
        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        # First check available execution providers
        available_providers = rt.get_available_providers()
        print(f"Available providers: {available_providers}")

        # Choose providers based on availability
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
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
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
                Image.BICUBIC,
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
        self.load_model(model_repo)

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", "\\(").replace(")", "\\)")
        )

        return sorted_general_strings, rating, character_res, general_res 

def main():
    args = parse_args()
    predictor = Predictor()
    validator = ImageValidator()
    aesthetic_scorer = AestheticScorer()

    dropdown_list = [
        SWINV2_MODEL_DSV3_REPO,
        CONV_MODEL_DSV3_REPO,
        VIT_MODEL_DSV3_REPO,
        VIT_LARGE_MODEL_DSV3_REPO,
        EVA02_LARGE_MODEL_DSV3_REPO,
        MOAT_MODEL_DSV2_REPO,
        SWIN_MODEL_DSV2_REPO,
        CONV_MODEL_DSV2_REPO,
        CONV2_MODEL_DSV2_REPO,
        VIT_MODEL_DSV2_REPO,
        SWINV2_MODEL_IS_DSV1_REPO,
        EVA02_LARGE_MODEL_IS_DSV1_REPO,
    ]

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>")
        
        with gr.Tabs():
            # Single image processing tab
            with gr.TabItem("Single Image Processing"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        image = gr.Image(type="pil", image_mode="RGBA", label="Input")
                        model_repo = gr.Dropdown(
                            dropdown_list,
                            value=SWINV2_MODEL_DSV3_REPO,
                            label="Model",
                        )
                        with gr.Row():
                            general_thresh = gr.Slider(
                                0, 1,
                                step=args.score_slider_step,
                                value=args.score_general_threshold,
                                label="General Tags Threshold",
                                scale=3,
                            )
                            general_mcut_enabled = gr.Checkbox(
                                value=False,
                                label="Use MCut threshold",
                                scale=1,
                            )
                        with gr.Row():
                            character_thresh = gr.Slider(
                                0, 1,
                                step=args.score_slider_step,
                                value=args.score_character_threshold,
                                label="Character Tags Threshold",
                                scale=3,
                            )
                            character_mcut_enabled = gr.Checkbox(
                                value=False,
                                label="Use MCut threshold",
                                scale=1,
                            )
                        with gr.Row():
                            clear = gr.ClearButton(
                                components=[image, model_repo, general_thresh, 
                                          general_mcut_enabled, character_thresh, 
                                          character_mcut_enabled],
                                variant="secondary",
                                size="lg",
                            )
                            submit = gr.Button(value="Submit", variant="primary", size="lg")
                    
                    with gr.Column(variant="panel"):
                        sorted_general_strings = gr.Textbox(label="Output (string)")
                        rating = gr.Label(label="Rating")
                        character_res = gr.Label(label="Output (characters)")
                        general_res = gr.Label(label="Output (tags)")
                        clear.add([sorted_general_strings, rating, character_res, general_res])

                submit.click(
                    predictor.predict,
                    inputs=[image, model_repo, general_thresh, general_mcut_enabled,
                           character_thresh, character_mcut_enabled],
                    outputs=[sorted_general_strings, rating, character_res, general_res],
                )

            with gr.TabItem("Aesthetic Scoring"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        aesthetic_image = gr.Image(
                            type="pil",
                            image_mode="RGB",
                            label="Input Image"
                        )
                        score_button = gr.Button(
                            value="Get Aesthetic Score",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            aesthetic_score = gr.Number(
                                label="Score",
                                precision=4,
                                value=None
                            )
                            quality_tag = gr.Textbox(
                                label="Quality Level",
                                value=None
                            )
                            aesthetic_tag = gr.Textbox(
                                label="Aesthetic Category",
                                value=None
                            )
                        
                score_button.click(
                    aesthetic_scorer.get_score,
                    inputs=[aesthetic_image],
                    outputs=[aesthetic_score, quality_tag, aesthetic_tag]
                )

            # Batch processing tab
            with gr.TabItem("Batch Processing"):
                with gr.Column(variant="panel"):
                    folder_path = gr.Textbox(
                        label="Image Folder Path",
                        placeholder="Enter the folder path containing images",
                        info="Supports jpg, jpeg, png, webp, webm formats"
                    )
                    manual_tags = gr.Textbox(
                        label="Manual Tags",
                        placeholder="Enter tags to add, separated by commas (e.g., masterpiece, aesthetic)",
                        info="These tags will be added to the beginning of all image tags"
                    )
                    batch_model = gr.Dropdown(
                        dropdown_list,
                        value=SWINV2_MODEL_DSV3_REPO,
                        label="Model",
                    )
                    with gr.Row():
                        batch_general_thresh = gr.Slider(
                            0, 1,
                            step=args.score_slider_step,
                            value=args.score_general_threshold,
                            label="General Tags Threshold"
                        )
                        batch_general_mcut = gr.Checkbox(
                            value=False,
                            label="Use MCut threshold"
                        )
                    with gr.Row():
                        batch_character_thresh = gr.Slider(
                            0, 1,
                            step=args.score_slider_step,
                            value=args.score_character_threshold,
                            label="Character Tags Threshold"
                        )
                        batch_character_mcut = gr.Checkbox(
                            value=False,
                            label="Use MCut threshold"
                        )
                    batch_submit = gr.Button(value="Start Batch Processing", variant="primary")
                    batch_output = gr.Textbox(label="Processing Result")
                    batch_results = gr.Dataframe(
                        headers=["file", "tags", "rating", "characters"],
                        label="Processing Details"
                    )

                batch_submit.click(
                    lambda *args: batch_process(predictor, *args),
                    inputs=[
                        folder_path,
                        batch_model,
                        batch_general_thresh,
                        batch_general_mcut,
                        batch_character_thresh,
                        batch_character_mcut,
                        manual_tags,
                    ],
                    outputs=[batch_output, batch_results]
                )

            with gr.TabItem("Batch Aesthetic Processing"):
                with gr.Column(variant="panel"):
                    batch_source = gr.Textbox(
                        label="Source Folder Path",
                        placeholder="Enter folder containing images to sort",
                        info="Supports jpg, jpeg, png, webp formats including subfolders"
                    )
                    batch_target = gr.Textbox(
                        label="Target Folder Path",
                        placeholder="Enter folder to sort images into"
                    )
                    batch_aesthetic_button = gr.Button(
                        value="Sort Images by Quality",
                        variant="primary"
                    )
                    batch_aesthetic_output = gr.Textbox(
                        label="Processing Results",
                        lines=10
                    )

                    def format_sort_results(stats):
                        output = [
                            f"Processed {stats['total_processed']} images\n",
                            "\nSorted counts:",
                        ]
                        for folder, count in stats['sorted_counts'].items():
                            output.append(f"  {folder}: {count}")
                            
                        if stats['errors']:
                            output.append("\nErrors:")
                            for error in stats['errors'][:5]:
                                output.append(f"  {error}")
                            if len(stats['errors']) > 5:
                                output.append(f"  ...and {len(stats['errors'])-5} more errors")
                                
                        return "\n".join(output)

                    def sort_and_format(source, target):
                        if not source or not target:
                            return "Error: Please enter source and target paths"
                        try:
                            sorter = AestheticSorter()
                            stats = sorter.sort_images(source, target)
                            return format_sort_results(stats)
                        except Exception as e:
                            return f"Error during sorting: {str(e)}"

                    batch_aesthetic_button.click(
                        sort_and_format,
                        inputs=[batch_source, batch_target],
                        outputs=[batch_aesthetic_output]
                    )
            
            with gr.TabItem("Dataset Validation"):
                with gr.Column(variant="panel"):
                    validate_path = gr.Textbox(
                        label="Dataset Path",
                        placeholder="Enter the folder path containing images",
                        info="Supports jpg, jpeg, png, webp formats, including subfolders"
                    )
                    grayscale_target = gr.Textbox(
                        label="Grayscale Images Target Folder (optional)",
                        placeholder="Enter folder path to move grayscale images to",
                        info="Leave empty to keep files in place"
                    )
                    validate_button = gr.Button(value="Validate Dataset", variant="primary")
                    validate_output = gr.Textbox(
                        label="Validation Result",
                        lines=10
                    )

                def validate_and_move(path: str, target: str) -> str:
                    analysis = validator.analyze_dataset(path)
                    moved_stats = None
                    
                    if target and analysis['grayscale_files']:
                        moved_stats = validator.move_grayscale_images(
                            analysis['grayscale_files'], 
                            target
                        )
                        
                    return validator.format_results(analysis, moved_stats)

                validate_button.click(
                    validate_and_move,
                    inputs=[validate_path, grayscale_target],
                    outputs=[validate_output]
                )
            
            with gr.TabItem("Dataset Merge"):
                with gr.Column(variant="panel"):
                    source_path = gr.Textbox(
                        label="Source Folder Path",
                        placeholder="Enter the source path containing subfolders"
                    )
                    target_path = gr.Textbox(
                        label="Target Folder Path",
                        placeholder="Enter the target path to merge into"
                    )
                    merge_button = gr.Button(value="Start Merge", variant="primary")
                    merge_output = gr.Textbox(
                        label="Merge Result",
                        lines=10
                    )

                    def format_merge_results(stats, target):
                        return f"""Merge complete!
                        
            Processed files: {stats['files_moved']}
            Processed characters: {stats['characters_merged']}
            Source folders: {', '.join(stats['source_folders'])}

            Files moved to: {target}
            Merged character list saved to: {os.path.join(target, 'merged_characters_list.json')}"""

                    def merge_and_format(source, target):
                        if not source or not target:
                            return "Error: Please enter source and target folder paths"
                        try:
                            stats = DatasetMerger().merge_dataset(source, target)
                            return format_merge_results(stats, target)
                        except Exception as e:
                            return f"Error: {str(e)}"

                    merge_button.click(
                        merge_and_format,
                        inputs=[source_path, target_path],
                        outputs=[merge_output]
                    )

    demo.queue(max_size=10)
    demo.launch()

if __name__ == "__main__":
    main()