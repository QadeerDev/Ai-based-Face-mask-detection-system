"""
Dataset Setup Helper
Downloads and organizes the Face Mask Detection dataset from Kaggle.

Requirements:
    pip install kaggle
    Set up ~/.kaggle/kaggle.json with your API credentials
    OR manually download from: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
"""

import os
import shutil
import zipfile
from pathlib import Path


def setup_from_kaggle():
    """Auto-download via Kaggle API."""
    try:
        import kaggle
        print("📥 Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            "omkargurav/face-mask-dataset",
            path="./data_raw",
            unzip=True
        )
        organize_dataset("./data_raw")
    except ImportError:
        print("❌ kaggle package not found. Install: pip install kaggle")
        print_manual_instructions()
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        print_manual_instructions()


def organize_dataset(raw_path: str):
    """
    Organizes dataset into:
      data/with_mask/
      data/without_mask/
    """
    print("📁 Organizing dataset structure...")

    # Common raw dataset folder names
    src_map = {
        "with_mask": ["with_mask", "Mask", "mask", "masked"],
        "without_mask": ["without_mask", "Without_Mask", "no_mask", "non-masked"]
    }

    os.makedirs("./data/with_mask", exist_ok=True)
    os.makedirs("./data/without_mask", exist_ok=True)

    moved = {"with_mask": 0, "without_mask": 0}

    for target, candidates in src_map.items():
        for candidate in candidates:
            src_dir = Path(raw_path) / candidate
            if src_dir.exists():
                for img_file in src_dir.glob("*.[jp][pn]*"):
                    dest = Path("./data") / target / img_file.name
                    shutil.copy2(img_file, dest)
                    moved[target] += 1
                break

    print(f"✅ Dataset organized:")
    print(f"   with_mask    : {moved['with_mask']} images")
    print(f"   without_mask : {moved['without_mask']} images")

    if moved["with_mask"] == 0 or moved["without_mask"] == 0:
        print("\n⚠️  Some classes may be empty. Check raw folder structure.")
        print_manual_instructions()


def print_manual_instructions():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Manual Dataset Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Go to: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
2. Download and unzip
3. Arrange as:

   face-mask-detection/
   └── data/
       ├── with_mask/
       │   ├── img_001.jpg
       │   └── ...
       └── without_mask/
           ├── img_001.jpg
           └── ...

4. Run: python src/train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def verify_dataset():
    """Quick dataset health check."""
    wm  = list(Path("./data/with_mask").glob("*.[jp][pn]*"))
    wom = list(Path("./data/without_mask").glob("*.[jp][pn]*"))

    print("\n📊 Dataset Verification:")
    print(f"   with_mask    : {len(wm)} images")
    print(f"   without_mask : {len(wom)} images")
    print(f"   Total        : {len(wm)+len(wom)} images")

    ratio = len(wm) / (len(wom) + 1e-9)
    if 0.8 <= ratio <= 1.25:
        print("   Balance      : ✅ Balanced dataset")
    else:
        print(f"   Balance      : ⚠️  Imbalanced ({ratio:.2f}x) — consider oversampling")

    return len(wm) > 0 and len(wom) > 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_dataset()
    elif Path("./data/with_mask").exists():
        verify_dataset()
    else:
        print("🔧 Setting up dataset...\n")
        setup_from_kaggle()
