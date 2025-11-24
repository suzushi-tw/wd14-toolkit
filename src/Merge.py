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
        self.target_path = None
    
    def merge_character_lists(self, folder_path: str) -> None:
        """Merge all characters_list.json files found in subfolders"""
        found_files = False
        print(f"Searching for character lists in: {folder_path}")
        
        for json_file in Path(folder_path).rglob("characters_list.json"):
            # Skip if file is in target directory
            if self.target_path and self.target_path in json_file.parents:
                continue
                
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

    def save_merged_characters(self, target_path: str) -> None:
        """Save merged character list to JSON file"""
        characters_path = Path(target_path) / "characters_list.json"
        with open(characters_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.characters), f, ensure_ascii=False, indent=2)
        print(f"Saved merged characters list to: {characters_path}")
        self.stats["characters_merged"] = len(self.characters)

    def move_files(self, source_path: str, target_path: str) -> None:
        """Move files from subfolders to target folder, keeping original names where possible"""
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        # Skip if target is within source
        if target_path.is_relative_to(source_path):
            raise ValueError("Target directory cannot be within source directory")
            
        target_path.mkdir(exist_ok=True)

        for folder in source_path.iterdir():
            if folder.is_dir() and folder != target_path:
                artist_name = folder.name
                for file in folder.rglob("*"):
                    if file.is_file() and not file.name == "characters_list.json":
                        # Try original filename first
                        new_path = target_path / file.name
                        
                        # Only add prefix if file already exists
                        if new_path.exists():
                            new_name = f"{artist_name}_{file.name}"
                            new_path = target_path / new_name
                            
                            # Handle multiple conflicts
                            counter = 1
                            while new_path.exists():
                                new_name = f"{artist_name}_{counter}_{file.name}"
                                new_path = target_path / new_name
                                counter += 1
                            
                        shutil.copy2(file, new_path)
                        self.stats["files_moved"] += 1

    def merge_dataset(self, source_path: str, target_path: str) -> Dict:
        """Main merge function"""
        self.target_path = Path(target_path)
        self.merge_character_lists(source_path)
        self.move_files(source_path, target_path)
        self.save_merged_characters(target_path)
        return self.stats