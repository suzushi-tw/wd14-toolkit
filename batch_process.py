import os
import glob
from PIL import Image
import pandas as pd
import gradio as gr
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
from contextlib import contextmanager

@contextmanager
def image_loader(path):
    """Context manager for proper image handling"""
    try:
        img = Image.open(path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        yield img
    finally:
        img.close()

def process_single_image(img_path, predictor, model_repo, params, manual_tag_list):
    """Process a single image with error handling"""
    try:
        with image_loader(img_path) as img:
            tags, rating, chars, general = predictor.predict(
                img,
                model_repo,
                params['general_thresh'],
                params['general_mcut_enabled'],
                params['character_thresh'],
                params['character_mcut_enabled']
            )
            
            all_tags = []
            rating_tag = max(rating.items(), key=lambda x: x[1])[0]
            all_tags.append(rating_tag)
            if manual_tag_list:
                all_tags.extend(manual_tag_list)
            if chars:
                all_tags.extend([k for k, v in chars.items()])
            if tags:
                all_tags.extend(tags.split(", "))

            txt_path = os.path.splitext(img_path)[0] + '.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(', '.join(all_tags))

            return {
                "file": os.path.basename(img_path),
                "tags": tags,
                "rating": rating_tag,
                "characters": ", ".join([k for k,v in chars.items()]),
                "txt_file": os.path.basename(txt_path)
            }, set(chars.keys()) if chars else set()
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None, set()

def batch_process(
    predictor,
    folder_path,
    model_repo,
    general_thresh,
    general_mcut_enabled,
    character_thresh,
    character_mcut_enabled,
    manual_tags="",
    progress=gr.Progress()
):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    results = []
    characters_set = set()
    predictor.load_model(model_repo)
    
    manual_tag_list = [tag.strip() for tag in manual_tags.split(",") if tag.strip()] if manual_tags else []
    
    is_cuda = any('CUDA' in provider for provider in predictor.model.get_providers())
    cuda_status = "Using GPU" if is_cuda else "Using CPU"
    logging.info(f"Processing status: {cuda_status}")
    
    # Gather image files more efficiently
    supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".webm"}
    image_files = [
        f for f in glob.glob(os.path.join(folder_path, "**/*.*"), recursive=True)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    logging.info(f"Found {len(image_files)} image files")
    
    # Parameters dictionary for process_single_image
    params = {
        'general_thresh': general_thresh,
        'general_mcut_enabled': general_mcut_enabled,
        'character_thresh': character_thresh,
        'character_mcut_enabled': character_mcut_enabled
    }
    
    # Parallel processing
    max_workers = 4 if is_cuda else 2
    process_func = partial(process_single_image, 
                         predictor=predictor,
                         model_repo=model_repo,
                         params=params,
                         manual_tag_list=manual_tag_list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(process_func, img_path): img_path 
                        for img_path in image_files}
        
        for future in progress.tqdm(as_completed(future_to_img), total=len(image_files)):
            result, chars = future.result()
            if result:
                results.append(result)
                characters_set.update(chars)

    # Save results
    if results:
        output_path = os.path.join(folder_path, "tags_output.csv")
        characters_path = os.path.join(folder_path, "characters_list.json")
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        with open(characters_path, 'w', encoding='utf-8') as f:
            json.dump(list(characters_set), f, ensure_ascii=False, indent=2)
        
        status_msg = f"""Processing complete! 
        Processing mode: {cuda_status}
        Total processed: {len(results)} images
        CSV result: {output_path}
        Characters list: {characters_path}
        Created corresponding txt files for each image"""
    else:
        status_msg = "No images were successfully processed"
    
    return status_msg, df