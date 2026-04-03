#!/usr/bin/env python3
"""
Image Preprocessing Script for ViT
Prepares PlantVillage dataset for Vision Transformer training - MAC OPTIMIZED
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

# Potato disease classes - try both capitalization variants
POTATO_CLASSES = {
    'Potato___Early_blight': 0,
    'Potato___Late_blight': 1,
    'Potato___healthy': 2
}

# Alternate names (some dataset versions use different casing)
POTATO_CLASS_ALIASES = {
    'Potato___Early_Blight': 'Potato___Early_blight',
    'Potato___Late_Blight': 'Potato___Late_blight',
}

def find_plantvillage_data():
    """Locate PlantVillage dataset"""
    print("\n🔍 Searching for PlantVillage data...")
    
    possible_paths = [
        Path("data/raw/PlantVillage"),
        Path("data/raw/plant-village"),
        Path("data/raw/Plant_leave_diseases_dataset_with_augmentation"),
        Path("data/raw/New Plant Diseases Dataset(Augmented)"),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found at: {path}")
            return path
    
    # Search in subdirectories
    raw_path = Path("data/raw")
    if raw_path.exists():
        for item in raw_path.rglob("*Potato*"):
            if item.is_dir():
                print(f"✅ Found potato data at: {item}")
                return item.parent
    
    raise FileNotFoundError("❌ PlantVillage data not found! Run download script first.")

def extract_potato_images(source_path, output_path):
    """Extract only potato disease images"""
    print("\n🥔 Extracting potato images...")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {cls: 0 for cls in POTATO_CLASSES.keys()}
    
    # Search for potato folders (case-insensitive)
    for class_name in POTATO_CLASSES.keys():
        class_dir = None

        # Try exact match first
        exact = source_path / class_name
        if exact.is_dir():
            class_dir = exact
        else:
            # Case-insensitive search
            target_lower = class_name.lower()
            for path in source_path.iterdir():
                if path.is_dir() and path.name.lower() == target_lower:
                    class_dir = path
                    break
            # Also try rglob as last resort
            if class_dir is None:
                for path in source_path.rglob("*"):
                    if path.is_dir() and path.name.lower() == target_lower:
                        class_dir = path
                        break

        if class_dir is None:
            print(f"Warning: {class_name} folder not found")
            continue
        
        # Create output directory
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        # Copy images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
                 list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG"))
        
        print(f"   {class_name}: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"Copying {class_name}"):
            try:
                # Verify it's a valid image
                img = Image.open(img_path)
                img.verify()
                
                # Copy to output
                shutil.copy2(img_path, output_class_dir / img_path.name)
                stats[class_name] += 1
                
            except Exception as e:
                print(f"⚠️  Skipping corrupted image: {img_path.name}")
    
    return stats

def create_train_val_split(data_path, train_ratio=0.8):
    """Split data into train and validation sets"""
    print("\n✂️  Creating train/val split...")
    
    data_path = Path(data_path)
    train_path = data_path.parent / "train"
    val_path = data_path.parent / "val"
    
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    
    split_stats = {"train": {}, "val": {}}
    
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = (list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) +
                  list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG")) +
                  list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPEG")))
        
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class directories
        train_class_dir = train_path / class_name
        val_class_dir = val_path / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Move images
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
        
        split_stats["train"][class_name] = len(train_images)
        split_stats["val"][class_name] = len(val_images)
        
        print(f"   {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    return split_stats

def create_metadata(output_path, stats):
    """Create metadata file"""
    metadata = {
        "classes": POTATO_CLASSES,
        "num_classes": len(POTATO_CLASSES),
        "class_names": list(POTATO_CLASSES.keys()),
        "stats": stats,
        "image_size": 224,  # ViT standard input
        "normalization": {
            "mean": [0.485, 0.456, 0.406],  # ImageNet stats
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata saved: {output_path / 'metadata.json'}")

def main():
    print("=" * 60)
    print("🖼️  IMAGE DATA PREPROCESSING FOR ViT - MAC")
    print("=" * 60)
    
    try:
        # Find source data
        source_path = find_plantvillage_data()
        
        # Extract potato images
        output_path = Path("data/processed/potato_images")
        stats = extract_potato_images(source_path, output_path)
        
        print("\n📊 Extraction Summary:")
        total = 0
        for class_name, count in stats.items():
            print(f"   {class_name}: {count} images")
            total += count
        print(f"   TOTAL: {total} images")
        
        if total == 0:
            print("\n❌ No images extracted! Check dataset structure.")
            return 1
        
        # Create train/val split
        split_stats = create_train_val_split(output_path)
        
        # Create metadata
        processed_path = Path("data/processed")
        all_stats = {
            "extraction": stats,
            "split": split_stats
        }
        create_metadata(processed_path, all_stats)
        
        print("\n" + "=" * 60)
        print("✅ IMAGE PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"📁 Train images: data/processed/train/")
        print(f"📁 Val images: data/processed/val/")
        print(f"📊 Total: {total} images")
        print(f"▶️  Next: Run ViT training script")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
