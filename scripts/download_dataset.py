"""
Download the Art and Advertising Visual Optimization dataset from Kaggle.

This dataset contains:
- Actual advertisement images
- Visual optimization data
- Performance metrics for ad creative testing
"""

import kagglehub
import shutil
from pathlib import Path

# Download latest version to cache
cache_path = kagglehub.dataset_download("zara2099/art-and-advertising-visual-optimization-dataset")
print(f"Downloaded to cache: {cache_path}")

# Copy to project data folder
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

target_path = data_dir / "ad-visual-optimization"

if target_path.exists():
    print(f"Dataset already exists at: {target_path}")
else:
    shutil.copytree(cache_path, target_path)
    print(f"Copied to project: {target_path}")

print(f"\nFinal path: {target_path}")
