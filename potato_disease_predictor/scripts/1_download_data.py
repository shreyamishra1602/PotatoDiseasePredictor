#!/usr/bin/env python3
"""
Data Download Script - Kaggle Datasets
Downloads weather and image datasets for potato disease prediction
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_kaggle():
    """Check Kaggle API setup"""
    kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_path.exists():
        print("Kaggle API key not found!")
        print("Setup instructions:")
        print("   1. Go to https://www.kaggle.com/settings/account")
        print("   2. Scroll to 'API' section and click 'Create New Token'")
        print("   3. Save kaggle.json to ~/.kaggle/")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def run_kaggle_download(dataset, output_path):
    """Run kaggle download with proper error handling"""
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", output_path, "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Download failed for {dataset}")
        print(f"stderr: {result.stderr}")
        return False
    return True


def download_weather_data():
    """Download weather-based disease dataset"""
    # Skip if already downloaded
    csv_files = list(Path("data/raw").glob("*.csv"))
    if csv_files:
        print(f"\nWeather data already exists ({csv_files[0].name}), skipping download.")
        return True

    print("\nDownloading Weather Dataset...")
    ok = run_kaggle_download(
        "tamima1530/potato-leaf-disease-based-on-weather-details",
        "data/raw"
    )
    if ok:
        print("Weather data downloaded!")
    return ok


def download_image_data():
    """Download PlantVillage image dataset"""
    # Skip if already downloaded
    raw_path = Path("data/raw")
    img_dirs = [d for d in raw_path.iterdir() if d.is_dir()] if raw_path.exists() else []
    if img_dirs:
        print(f"\nImage data already exists ({img_dirs[0].name}), skipping download.")
        return True

    print("\nDownloading PlantVillage Image Dataset (~344MB)...")
    print("   This may take a few minutes...")
    ok = run_kaggle_download("arjuntejaswi/plant-village", "data/raw")
    if ok:
        print("Image data downloaded!")
    return ok


def verify_downloads():
    """Verify downloaded data"""
    raw_path = Path("data/raw")
    files = list(raw_path.rglob("*"))

    print(f"\nDownloaded files: {len(files)}")

    # Check for weather CSV
    csv_files = list(raw_path.glob("*.csv"))
    if csv_files:
        print(f"Weather CSV found: {csv_files[0].name}")

    # Check for image folders
    img_folders = [f for f in raw_path.iterdir() if f.is_dir()]
    if img_folders:
        print(f"Image folders found: {len(img_folders)}")
        for folder in img_folders:
            print(f"   - {folder.name}")

    return True


def main():
    print("=" * 60)
    print("POTATO DISEASE DATASET DOWNLOADER")
    print("=" * 60)

    # Check Kaggle setup
    if not setup_kaggle():
        sys.exit(1)

    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    try:
        # Download datasets
        w_ok = download_weather_data()
        i_ok = download_image_data()

        # Verify
        verify_downloads()

        print("\n" + "=" * 60)
        if w_ok and i_ok:
            print("ALL DATASETS DOWNLOADED SUCCESSFULLY!")
        else:
            print("SOME DOWNLOADS FAILED - check errors above")
        print("=" * 60)
        print("Data location: data/raw/")
        print("Next: Run python scripts/2_preprocess_weather.py")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have Kaggle API configured correctly")
        sys.exit(1)


if __name__ == "__main__":
    main()
