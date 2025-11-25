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
                
                # Initialize variables for character tags from JSON
                processed_chars_from_json = False
                json_char_list = []
                json_path = img_path + '.json' # Assumes JSON file is image_filename.json

                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f_json:
                            data = json.load(f_json)
                        
                        raw_char_tags_from_json = []
                        if 'tags_character' in data and isinstance(data['tags_character'], list):
                            raw_char_tags_from_json = data['tags_character']
                        elif 'tag_string_character' in data and isinstance(data['tag_string_character'], str):
                            raw_char_tags_from_json = data['tag_string_character'].split()
                        
                        if raw_char_tags_from_json:
                            for tag in raw_char_tags_from_json:
                                # Sanitize: replace underscores with spaces
                                sanitized_tag = tag.replace('_', ' ')
                                json_char_list.append(sanitized_tag)
                            processed_chars_from_json = True
                            # print(f"Loaded {len(json_char_list)} character tags from {json_path}")
                    except Exception as e_json:
                        print(f"Error reading or parsing JSON {json_path}: {e_json}")

                # Get predictions (rating, general tags, and fallback characters)
                tags, rating, pred_chars_dict, general = predictor.predict(
                    img,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,  
                    character_mcut_enabled
                )
                
                current_image_characters = []
                if processed_chars_from_json and json_char_list: # Prioritize JSON if tags were found
                    current_image_characters = json_char_list
                elif pred_chars_dict: # Fallback to predictor's characters
                    current_image_characters = list(pred_chars_dict.keys())
                
                if current_image_characters:
                    characters_set.update(current_image_characters)
                
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                
                all_tags_for_txt = []
                rating_tag = max(rating.items(), key=lambda x: x[1])[0]
                all_tags_for_txt.append(rating_tag)
                if manual_tag_list:
                    all_tags_for_txt.extend(manual_tag_list)
                
                if current_image_characters:
                    all_tags_for_txt.extend(current_image_characters)
                
                if tags: # General tags string from predictor
                    all_tags_for_txt.extend(tags.split(", "))

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(', '.join(all_tags_for_txt))
                
                results.append({
                    "file": os.path.basename(img_path),
                    "tags": tags, # General tags string from predictor
                    "rating": rating_tag,
                    "characters": ", ".join(current_image_characters),
                    "processing_mode": cuda_status,
                    "txt_file": os.path.basename(txt_path)
                })

                # Delete JSON file if it was successfully processed for characters
                if processed_chars_from_json and os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                        # print(f"Deleted JSON file: {json_path}")
                    except Exception as e_del:
                        print(f"Error deleting JSON file {json_path}: {e_del}")
                
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
    
    # Format detailed results for display
    detailed_results = []
    for r in results[:10]:  # Show first 10 as preview
        detailed_results.append(
            f"File: {r['file']}\n"
            f"  Rating: {r['rating']}\n"
            f"  Characters: {r['characters']}\n"
            f"  Tags: {r['tags'][:100]}{'...' if len(r['tags']) > 100 else ''}\n"
        )
    
    status_msg = f"""Processing complete! 
Processing mode: {cuda_status}
Total processed: {len(results)} images
CSV result: {output_path}
Characters list: {characters_path}
Created corresponding txt files for each image

Preview of first {min(len(results), 10)} results:
{''.join(detailed_results)}
{f'...and {len(results) - 10} more results in CSV' if len(results) > 10 else ''}"""
    
    return status_msg