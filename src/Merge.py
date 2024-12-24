import os
import shutil
import json
from pathlib import Path
from typing import Set, List, Dict

class DatasetMerger:
    def __init__(self):
        self.characters: Set[str] = set()
        self.stats: Dict = {
            "files_moved": 0,
            "characters_merged": 0,
            "source_folders": set()
        }
    
    def merge_character_lists(self, folder_path: str) -> None:
        """Merge all characters_list.json files found in subfolders"""
        found_files = False
        print(f"Searching for character lists in: {folder_path}")
        
        for json_file in Path(folder_path).rglob("characters_list.json"):
            found_files = True
            print(f"Found character list: {json_file}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    char_list = json.load(f)
                    if isinstance(char_list, list):
                        print(f"Adding {len(char_list)} characters from {json_file.parent.name}")
                        self.characters.update(char_list)
                        self.stats["source_folders"].add(json_file.parent.name)
                    else:
                        print(f"Warning: Invalid format in {json_file}")
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        if not found_files:
            print(f"Warning: No characters_list.json files found in {folder_path}")
        else:
            print(f"Total unique characters found: {len(self.characters)}")

    def save_merged_characters(self, target_path: str) -> None:
        """Save merged character list to JSON"""
        try:
            output_file = Path(target_path) / "merged_characters_list.json"
            sorted_chars = sorted(list(self.characters))
            print(f"Saving {len(sorted_chars)} characters to {output_file}")
            
            # Ensure target directory exists
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_chars, f, ensure_ascii=False, indent=2)
            
            print(f"Successfully saved character list to {output_file}")
            self.stats["characters_merged"] = len(sorted_chars)
        except Exception as e:
            print(f"Error saving character list: {e}")
            raise

    def move_files(self, source_path: str, target_path: str) -> None:
        """Move files from subfolders to target folder with artist prefix"""
        source_path = Path(source_path)
        target_path = Path(target_path)
        target_path.mkdir(exist_ok=True)

        for folder in source_path.iterdir():
            if folder.is_dir():
                artist_name = folder.name
                for file in folder.rglob("*"):
                    if file.is_file() and not file.name == "characters_list.json":
                        # Create new filename with artist prefix
                        new_name = f"{artist_name}_{file.name}"
                        new_path = target_path / new_name
                        
                        # Ensure no overwrites
                        counter = 1
                        while new_path.exists():
                            new_name = f"{artist_name}_{counter}_{file.name}"
                            new_path = target_path / new_name
                            counter += 1
                            
                        shutil.copy2(file, new_path)
                        self.stats["files_moved"] += 1

    def merge_dataset(self, source_path: str, target_path: str) -> Dict:
        """Main merge function"""
        self.merge_character_lists(source_path)
        self.move_files(source_path, target_path)
        self.save_merged_characters(target_path)
        return self.stats