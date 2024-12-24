import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class ImageValidator:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        
    def get_image_files(self, folder_path: str) -> List[Path]:
        """Recursively get all supported image files"""
        files = []
        seen = set()
        for ext in self.supported_formats:
            for file in Path(folder_path).rglob(f'*{ext}'):
                if str(file) not in seen:
                    seen.add(str(file))
                    files.append(file)
            for file in Path(folder_path).rglob(f'*{ext.upper()}'):
                if str(file) not in seen:
                    seen.add(str(file))
                    files.append(file)
        return files

    def analyze_dataset(self, folder_path: str) -> Dict:
        """Analyze the dataset"""
        files = self.get_image_files(folder_path)
        
        # Check for duplicate filenames
        base_name_dict = defaultdict(list)
        format_stats = defaultdict(int)
        
        for file_path in files:
            # Count file extensions
            format_stats[file_path.suffix.lower()] += 1
            # Check filenames
            base_name = file_path.stem
            base_name_dict[base_name].append(file_path)
            
        # Organize results
        duplicates = {name: paths for name, paths in base_name_dict.items() if len(paths) > 1}
        
        return {
            'total_files': len(files),
            'format_distribution': dict(format_stats),
            'duplicates': duplicates
        }

    def format_results(self, analysis: Dict) -> str:
        """Format the output results"""
        result = []
        
        # Basic statistics
        result.append("=== Dataset Statistics ===")
        result.append(f"Total files: {analysis['total_files']}")
        
        result.append("\n=== Format Distribution ===")
        for fmt, count in analysis['format_distribution'].items():
            result.append(f"{fmt}: {count} files")
            
        # Duplicate files
        if analysis['duplicates']:
            result.append("\n=== Duplicate Filenames (may affect training) ===")
            for name, paths in analysis['duplicates'].items():
                result.append(f"\nFilename: {name}")
                for path in paths:
                    result.append(f"  - {path}")
            
            result.append("\nTip: During SDXL LoRA training, images with the same filename may be considered as the same image")
            result.append("Suggestion: Ensure each image has a unique filename")
        else:
            result.append("\n✓ No duplicate filenames found")
                    
        return "\n".join(result)