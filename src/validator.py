import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from .ColorAnalyzer import ColorAnalyzer
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import gc

class ImageValidator:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        self.color_analyzer = ColorAnalyzer()
        self.max_workers = max(1, min(4, multiprocessing.cpu_count() // 2))

    def move_grayscale_images(self, grayscale_files: List[Path], target_folder: str) -> Tuple[int, List[str]]:
        """Move grayscale images to target folder"""
        if not grayscale_files:
            return 0, []
            
        target_path = Path(target_folder)
        target_path.mkdir(parents=True, exist_ok=True)
        
        moved = 0
        errors = []
        
        for file_path in grayscale_files:
            try:
                new_path = target_path / file_path.name
                # Handle duplicate filenames
                counter = 1
                while new_path.exists():
                    new_path = target_path / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1
                file_path.rename(new_path)
                moved += 1
            except Exception as e:
                errors.append(f"Failed to move {file_path}: {str(e)}")
                
        return moved, errors

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
        
    def process_image(self, file_path):
        """Process single image for parallel execution"""
        try:
            is_gray = self.color_analyzer.is_grayscale(str(file_path))
            return {
                'path': file_path,
                'is_grayscale': is_gray,
                'format': file_path.suffix.lower(),
                'base_name': file_path.stem
            }
        except Exception as e:
            return {
                'path': file_path,
                'error': str(e)
            }

    def analyze_dataset(self, folder_path: str) -> Dict:
        files = self.get_image_files(folder_path)
        total_files = len(files)
        
        if total_files == 0:
            return {
                'total_files': 0,
                'format_distribution': {},
                'duplicates': {},
                'grayscale_files': [],
                'grayscale_count': 0,
                'errors': []
            }

        # Initialize statistics
        format_stats = defaultdict(int)
        base_name_dict = defaultdict(list)
        grayscale_files = []
        errors = []
        
        # Process in smaller batches
        batch_size = 2000  # Adjust this value based on your system
        for i in range(0, total_files, batch_size):
            batch = files[i:i + batch_size]
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_image, file_path) for file_path in batch]
                
                with tqdm(total=len(batch), desc=f"Analyzing batch {i//batch_size + 1}") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        pbar.update(1)
                        
                        if 'error' in result:
                            errors.append((result['path'], result['error']))
                            continue
                            
                        format_stats[result['format']] += 1
                        base_name_dict[result['base_name']].append(result['path'])
                        
                        if result['is_grayscale']:
                            grayscale_files.append(result['path'])

            gc.collect()

        # Find duplicates
        duplicates = {name: paths for name, paths in base_name_dict.items() if len(paths) > 1}
        
        return {
            'total_files': total_files,
            'format_distribution': dict(format_stats),
            'duplicates': duplicates,
            'grayscale_files': grayscale_files,
            'grayscale_count': len(grayscale_files),
            'errors': errors
        }
    
    def format_results(self, analysis: Dict, moved_stats: Tuple[int, List[str]] = None) -> str:
        """Format the output results"""
        result = []
        
        # Basic statistics
        result.append("=== Dataset Statistics ===")
        result.append(f"Total files: {analysis['total_files']}")
        result.append(f"Grayscale/B&W images: {analysis['grayscale_count']} ({(analysis['grayscale_count']/analysis['total_files']*100):.1f}%)")
        
        # Add move results if available
        if moved_stats:
            moved_count, move_errors = moved_stats
            result.append(f"\n=== Move Results ===")
            result.append(f"Successfully moved {moved_count} grayscale images")
            if move_errors:
                result.append("\nMove errors:")
                for error in move_errors:
                    result.append(f"  - {error}")
        
        result.append("\n=== Format Distribution ===")
        for fmt, count in analysis['format_distribution'].items():
            result.append(f"{fmt}: {count} files")
            
        # List grayscale files (only if not moved)
        if analysis['grayscale_files'] and not moved_stats:
            result.append("\n=== Grayscale Images ===")
            for path in analysis['grayscale_files']:
                result.append(f"  - {path}")
            
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