# WD14 Toolkit

This standalone WD14 toolkit is designed to fix CUDA version mismatches and provides features for tagging datasets for Stable Diffusion. It includes batch processing and dataset validation to mitigate errors when training with Kohya SS. Additionally, it offers a simple tool to merge datasets into one folder while maintaining a character list.

## Features

- **Tagging Dataset for Stable Diffusion**: Tag individual images or entire datasets to prepare them for training.
- **Batch Processing**: Process multiple images in a folder, automatically tagging and saving results.
- **Dataset Validation**: Validate datasets to identify and resolve potential issues before training.
- **Dataset Merging**: Merge multiple datasets into a single folder, maintaining a comprehensive character list.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/suzushi-tw/wd14-toolkit.git
    cd wd14-toolkit
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## License

MIT