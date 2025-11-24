import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from .Aesthetic import AestheticScorer

class AestheticSorter:
    def __init__(self):
        self.scorer = AestheticScorer()
        self.quality_folders = [
            "masterpiece", "high_quality", "good_quality",
            "normal_quality", "low_quality", "worst_quality"
        ]
        # Pre-load model
        if self.scorer.pipe is None:
            self.scorer.get_score(Image.new('RGB', (224, 224)))

    def process_image_batch(self, batch):
        results = []
        for img_path in batch:
            try:
                image = Image.open(img_path).convert('RGB')
                score, quality, _, _ = self.scorer.get_score(image)
                results.append((img_path, quality.replace(" ", "_").lower()))
            except Exception as e:
                results.append((img_path, f"error: {str(e)}"))
        return results

    def sort_images(self, source_path: str, target_path: str) -> dict:
        stats = {
            "total_processed": 0,
            "sorted_counts": {folder: 0 for folder in self.quality_folders},
            "errors": []
        }

        # Create quality folders
        target_base = Path(target_path)
        for folder in self.quality_folders:
            (target_base / folder).mkdir(parents=True, exist_ok=True)

        # Gather all image files
        source_path = Path(source_path)
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            image_files.extend(source_path.rglob(ext))

        print(f"Found {len(image_files)} images to process")

        # Process in batches using multiple threads
        batch_size = 32
        num_workers = min(mp.cpu_count(), 8)  # Limit max workers
        batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_image_batch, batch) for batch in batches]
            
            with tqdm(total=len(image_files)) as pbar:
                for future in as_completed(futures):
                    for img_path, quality_folder in future.result():
                        try:
                            if quality_folder.startswith("error"):
                                stats["errors"].append(f"Error processing {img_path}: {quality_folder}")
                                continue

                            if quality_folder not in self.quality_folders:
                                quality_folder = "normal_quality"

                            target_file = target_base / quality_folder / Path(img_path).name
                            if target_file.exists():
                                base = target_file.stem
                                counter = 1
                                while target_file.exists():
                                    target_file = target_base / quality_folder / f"{base}_{counter}{target_file.suffix}"
                                    counter += 1

                            # Use copy instead of loading and saving
                            os.system(f'copy "{img_path}" "{target_file}"')
                            
                            stats["sorted_counts"][quality_folder] += 1
                            stats["total_processed"] += 1
                            pbar.update(1)

                        except Exception as e:
                            stats["errors"].append(f"Error processing {img_path}: {str(e)}")

        torch.cuda.empty_cache()  # Clear GPU memory
        return stats