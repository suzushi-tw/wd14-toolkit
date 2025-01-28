import gradio as gr
import os
from src.config.config import dropdown_list, SWINV2_MODEL_DSV3_REPO
from src.Batch_aestheitc import AestheticSorter
from batch_process import batch_process
from src.Merge import DatasetMerger
TITLE = "WaifuDiffusion Tagger"

def create_interface(predictor, validator, aesthetic_scorer, anime_hand_detector=None, joy_captioner=None, args=None):
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>")
        
        with gr.Tabs():
            with gr.TabItem("Hand Gesture Detection"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        hand_image = gr.Image(
                            type="pil",
                            image_mode="RGB",
                            label="Input Image"
                        )
                        detect_gesture_button = gr.Button(
                            value="Detect Anime Hand Gestures",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(variant="panel"):
                        output_image = gr.Image(
                            type="pil",
                            label="Detected Hands"
                        )
                        detected_gestures = gr.Json(
                            label="Detected Gestures"
                        )
                        confidence_display = gr.Label(
                            label="Confidence Scores"
                        )

                # Update click handler
                detect_gesture_button.click(
                    lambda img: anime_hand_detector.detect_gesture(img, debug=True),
                    inputs=[hand_image],
                    outputs=[detected_gestures, confidence_display, output_image]
                )

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
    return demo

def launch_interface(predictor, validator, aesthetic_scorer, anime_hand_detector, joy_captioner, args):
    demo = create_interface(predictor, validator, aesthetic_scorer, anime_hand_detector, joy_captioner, args)
    demo.launch(
        server_name="0.0.0.0",  # Required for Runpod
        server_port=3000,       # Runpod default port
        share=True             # Enable public URL
    )