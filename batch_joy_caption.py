"""
Batch processing for JoyCaption natural language image captioning.
"""

import os
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gradio as gr


def batch_joy_caption(
    joy_captioner,
    folder_path,
    caption_type="Descriptive",
    caption_length="long",
    custom_prompt=None,
    progress=gr.Progress()
):
    """
    Batch process images with JoyCaption to generate natural language captions.
    
    Args:
        joy_captioner: JoyCaptioner instance
        folder_path: Path to folder containing images
        caption_type: Type of caption to generate
        caption_length: Length of caption
        custom_prompt: Optional custom prompt
        progress: Gradio progress tracker
    
    Returns:
        String with processing results
    """
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' does not exist."
    
    # Load the model first
    joy_captioner.load_model()
    
    # Supported image formats
    supported_formats = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", fmt), recursive=True))
    
    if not image_files:
        return f"No images found in '{folder_path}'"
    
    print(f"Found {len(image_files)} images to process")
    
    # Statistics
    stats = {
        'total': len(image_files),
        'processed': 0,
        'errors': 0,
        'error_files': []
    }
    
    # Process each image
    for idx, img_path in enumerate(tqdm(image_files, desc="Generating captions")):
        try:
            # Update progress
            if progress:
                progress((idx + 1) / len(image_files), f"Processing {idx + 1}/{len(image_files)}")
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Generate caption
            caption = joy_captioner.predict(
                image,
                caption_type=caption_type,
                caption_length=caption_length,
                custom_prompt=custom_prompt
            )
            
            # Save caption to .txt file with same name as image
            txt_path = Path(img_path).with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            stats['processed'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            error_msg = f"{Path(img_path).name}: {str(e)}"
            stats['error_files'].append(error_msg)
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Format results
    result = f"""
Batch JoyCaption Processing Complete!

Total images: {stats['total']}
Successfully processed: {stats['processed']}
Errors: {stats['errors']}

Settings:
- Caption Type: {caption_type}
- Caption Length: {caption_length}
- Custom Prompt: {custom_prompt if custom_prompt else 'None'}

Output: Text files saved next to each image with .txt extension
"""
    
    if stats['error_files']:
        result += f"\n\nErrors encountered:\n"
        for error in stats['error_files'][:10]:  # Show first 10 errors
            result += f"  - {error}\n"
        if len(stats['error_files']) > 10:
            result += f"  ... and {len(stats['error_files']) - 10} more errors\n"
    
    return result
