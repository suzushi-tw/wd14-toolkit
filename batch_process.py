import os
import glob
from PIL import Image
import pandas as pd
import gradio as gr
import json

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
    results = []
    characters_set = set()  # Track unique characters
    predictor.load_model(model_repo)
    
    manual_tag_list = [tag.strip() for tag in manual_tags.split(",") if tag.strip()] if manual_tags else []
    
    is_cuda = any('CUDA' in provider for provider in predictor.model.get_providers())
    cuda_status = "Using GPU" if is_cuda else "Using CPU"
    print(f"Processing status: {cuda_status}")
    
    supported_formats = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.webm"]
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", fmt), recursive=True))
    
    print(f"Found {len(image_files)} image files")
    
    batch_size = 100 if is_cuda else 50
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        for img_path in progress.tqdm(batch_files):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                tags, rating, chars, general = predictor.predict(
                    img,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,  
                    character_mcut_enabled
                )
                
                if chars:
                    characters_set.update(chars.keys())
                
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                
                all_tags = []
                rating_tag = max(rating.items(), key=lambda x: x[1])[0]
                all_tags.append(rating_tag)
                if manual_tag_list:
                    all_tags.extend(manual_tag_list)
                if chars:
                    all_tags.extend([k for k, v in chars.items()])
                if tags:
                    all_tags.extend(tags.split(", "))

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(', '.join(all_tags))
                
                results.append({
                    "file": os.path.basename(img_path),
                    "tags": tags,
                    "rating": rating_tag,
                    "characters": ", ".join([k for k,v in chars.items()]),
                    "processing_mode": cuda_status,
                    "txt_file": os.path.basename(txt_path)
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    # Save CSV output
    output_path = os.path.join(folder_path, "tags_output.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Save characters list to JSON
    characters_path = os.path.join(folder_path, "characters_list.json")
    with open(characters_path, 'w', encoding='utf-8') as f:
        json.dump(list(characters_set), f, ensure_ascii=False, indent=2)
    
    status_msg = f"""Processing complete! 
    Processing mode: {cuda_status}
    Total processed: {len(results)} images
    CSV result: {output_path}
    Characters list: {characters_path}
    Created corresponding txt files for each image"""
    
    return status_msg, df