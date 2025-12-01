import gradio as gr
import os
from src.config.config import dropdown_list, SWINV2_MODEL_DSV3_REPO
from src.Batch_aestheitc import AestheticSorter
from batch_process import batch_process
from batch_joy_caption import batch_joy_caption
from src.Merge import DatasetMerger
TITLE = "WaifuDiffusion Tagger"

def create_interface(predictor, validator, aesthetic_scorer, joy_captioner, args=None):
    # Ensure args is not None (should always be provided)
    if args is None:
        raise ValueError("args parameter is required")
    
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>")
        
        with gr.Tabs():
            with gr.Tab("Single Image Processing"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        image = gr.Image(type="pil", label="Input")
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
                        
                        use_joy_caption = gr.Checkbox(
                            value=False,
                            label="Use JoyCaption instead of tagger",
                            info="When checked, only JoyCaption runs (saves VRAM by not loading tagger)"
                        )
                        
                        with gr.Row(visible=False) as joy_settings:
                            joy_caption_type = gr.Dropdown(
                                choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", 
                                        "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                                value="Descriptive",
                                label="Caption Type"
                            )
                            joy_caption_length = gr.Dropdown(
                                choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                                        [str(i) for i in range(20, 261, 10)],
                                value="long",
                                label="Caption Length",
                                info="Select length preset or specific word count"
                            )

                        with gr.Row():
                            clear = gr.Button("Clear", variant="secondary", size="lg")
                            submit = gr.Button(value="Submit", variant="primary", size="lg")
                    
                    with gr.Column(variant="panel"):
                        sorted_general_strings = gr.Textbox(label="Output (string)")
                        rating = gr.Textbox(label="Rating")
                        character_res = gr.Textbox(label="Output (characters)")
                        general_res = gr.Textbox(label="Output (tags)", lines=5)
                        joy_caption_output = gr.Textbox(label="JoyCaption Output", lines=5)

                def process_image(image, model_repo, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled, 
                                use_joy, joy_type, joy_length):
                    if use_joy:
                        # Only run JoyCaption
                        try:
                            caption = joy_captioner.predict(image, caption_type=joy_type, caption_length=joy_length)
                            return "", "", "", "", caption
                        except Exception as e:
                            return "", "", "", "", f"Error: {str(e)}"
                    else:
                        # Only run tagger
                        sorted_general_strings, rating_str, character_str, general_str = predictor.predict(
                            image, model_repo, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled
                        )
                        return sorted_general_strings, rating_str, character_str, general_str, ""

                def clear_inputs():
                    return None, SWINV2_MODEL_DSV3_REPO, args.score_general_threshold, False, args.score_character_threshold, False, \
                           False, "Descriptive", "long", "", "", "", "", ""
                
                use_joy_caption.change(
                    lambda x: gr.update(visible=x),
                    inputs=[use_joy_caption],
                    outputs=[joy_settings]
                )

                submit.click(
                    process_image,
                    inputs=[image, model_repo, general_thresh, general_mcut_enabled,
                           character_thresh, character_mcut_enabled, use_joy_caption, joy_caption_type, joy_caption_length],
                    outputs=[sorted_general_strings, rating, character_res, general_res, joy_caption_output],
                )
                
                clear.click(
                    clear_inputs,
                    inputs=[],
                    outputs=[image, model_repo, general_thresh, general_mcut_enabled, 
                            character_thresh, character_mcut_enabled, use_joy_caption, joy_caption_type, joy_caption_length,
                            sorted_general_strings, rating, character_res, general_res, joy_caption_output],
                )

            with gr.Tab("Aesthetic Scoring"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        aesthetic_image = gr.Image(
                            type="pil",
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
            with gr.Tab("Batch Processing"):
                with gr.Tabs():
                    # Danbooru Tags tab
                    with gr.Tab("Danbooru Tags"):
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
                            batch_output = gr.Textbox(label="Processing Result", lines=20)

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
                            outputs=[batch_output]
                        )
                    
                    # Natural Language (Newbie) tab
                    with gr.Tab("Natural Language (Newbie)"):
                        with gr.Column(variant="panel"):
                            nl_folder_path = gr.Textbox(
                                label="Image Folder Path",
                                placeholder="Enter the folder path containing images",
                                info="Supports jpg, jpeg, png, webp formats"
                            )
                            
                            with gr.Row():
                                nl_caption_type = gr.Dropdown(
                                    choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", 
                                            "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                                    value="Descriptive",
                                    label="Caption Type"
                                )
                                nl_caption_length = gr.Dropdown(
                                    choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                                            [str(i) for i in range(20, 261, 10)],
                                    value="long",
                                    label="Caption Length"
                                )
                            
                            nl_custom_prompt = gr.Textbox(
                                label="Custom Prompt (optional)",
                                placeholder="Enter a custom prompt to override caption type",
                                info="Leave empty to use the selected caption type",
                                lines=2
                            )
                            
                            nl_batch_submit = gr.Button(
                                value="Start JoyCaption Batch Processing", 
                                variant="primary"
                            )
                            nl_batch_output = gr.Textbox(
                                label="Processing Result", 
                                lines=20
                            )
                        
                        # Placeholder function - to be implemented later
                        def nl_batch_process_wrapper(folder, caption_type, caption_length, custom_prompt):
                            if not folder:
                                return "Error: Please enter a folder path"
                            try:
                                return batch_joy_caption(
                                    joy_captioner,
                                    folder,
                                    caption_type=caption_type,
                                    caption_length=caption_length,
                                    custom_prompt=custom_prompt if custom_prompt.strip() else None
                                )
                            except Exception as e:
                                return f"Error during batch processing: {str(e)}"
                        
                        nl_batch_submit.click(
                            nl_batch_process_wrapper,
                            inputs=[nl_folder_path, nl_caption_type, nl_caption_length, nl_custom_prompt],
                            outputs=[nl_batch_output]
                        )

            with gr.Tab("Batch Aesthetic Processing"):
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
            
            with gr.Tab("Dataset Validation"):
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
            
            with gr.Tab("Dataset Merge"):
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

    demo.queue()
    return demo

def launch_interface(predictor, validator, aesthetic_scorer, joy_captioner, args):
    demo = create_interface(predictor, validator, aesthetic_scorer, joy_captioner, args)
    demo.launch(
        server_name="0.0.0.0",  # Required for Runpod
        server_port=3000,       # Runpod default port
        share=True             # Enable public URL
    )