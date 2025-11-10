"""
Prepare V3 cow data with 5 breeds for training - FINAL VERSION
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

def prepare_cow_data_v3():
    """Prepare V3 cow data with 5 breeds into train/val/test splits"""
    
    # Source directory
    source_dir = Path("data/final_organized_v3/cows")
    target_dir = Path("data/processed_v3/cows")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    for split in ["train", "val", "test"]:
        (target_dir / split).mkdir(exist_ok=True)
    
    print("🐄 Preparing V3 Cow Data (5 breeds)...")
    print("="*60)
    
    # Collect all images and their labels from existing splits
    all_images = []
    all_labels = []
    
    # Define the 5 breeds we want to include
    target_breeds = ["Gir", "Sahiwal", "Holstein Friesian", "Jersey", "Red Sindhi"]
    
    # Process each split (train, valid, test) and each breed within
    for split_name in ["train", "valid", "test"]:
        split_dir = source_dir / split_name
        if not split_dir.exists():
            continue
            
        print(f"📁 Processing {split_name} split...")
        
        # Process each breed directory
        for breed_dir in split_dir.iterdir():
            if breed_dir.is_dir() and breed_dir.name in target_breeds:
                breed_name = breed_dir.name
                image_count = len(list(breed_dir.glob('*.jpg'))) + len(list(breed_dir.glob('*.jpeg'))) + len(list(breed_dir.glob('*.png')))
                print(f"  🏷️ {breed_name}: {image_count} images")
                
                for img_file in breed_dir.glob("*.jpg"):
                    all_images.append(img_file)
                    all_labels.append(breed_name)
                for img_file in breed_dir.glob("*.jpeg"):
                    all_images.append(img_file)
                    all_labels.append(breed_name)
                for img_file in breed_dir.glob("*.png"):
                    all_images.append(img_file)
                    all_labels.append(breed_name)
    
    print(f"\n📊 Found {len(all_images)} total images")
    
    # Verify images
    print("\n🔍 Verifying images...")
    valid_images = []
    valid_labels = []
    
    for img_path, label in tqdm(zip(all_images, all_labels), total=len(all_images), desc="Checking images"):
        try:
            with Image.open(img_path) as img:
                # Verify image can be loaded
                img.verify()
                valid_images.append(img_path)
                valid_labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping corrupted image: {img_path}")
    
    print(f"✅ {len(valid_images)} valid images after verification")
    
    # Convert to arrays for splitting
    X = np.array(valid_images)
    y = np.array(valid_labels)
    
    # Get unique breeds
    unique_breeds = np.unique(y)
    print(f"🏷️ Breeds: {', '.join(unique_breeds)}")
    
    # Split data: 70% train, 15% val, 15% test
    print("\n📦 Splitting data (70/15/15)...")
    
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Second split: separate train and val (70/15 of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 15/85
    )
    
    print(f"✅ Train: {len(X_train)} images")
    print(f"✅ Validation: {len(X_val)} images") 
    print(f"✅ Test: {len(X_test)} images")
    
    # Copy images to target directories
    print("\n📋 Organizing splits...")
    
    splits = [
        (X_train, y_train, "train"),
        (X_val, y_val, "val"),
        (X_test, y_test, "test")
    ]
    
    for X_split, y_split, split_name in splits:
        split_dir = target_dir / split_name
        
        # Create breed directories
        for breed in unique_breeds:
            (split_dir / breed).mkdir(exist_ok=True)
        
        # Copy images
        for img_path, label in tqdm(zip(X_split, y_split), 
                                   total=len(X_split), 
                                   desc=f"Copying {split_name}"):
            target_path = split_dir / label / img_path.name
            shutil.copy2(img_path, target_path)
    
    # Generate statistics
    print("\n" + "="*60)
    print("📊 V3 Cow Dataset Statistics:")
    print("="*60)
    
    print(f"{'Breed':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'Pct':<6}")
    print("-" * 70)
    
    total_images = len(X_train) + len(X_val) + len(X_test)
    
    for breed in unique_breeds:
        train_count = len([x for x, y in zip(X_train, y_train) if y == breed])
        val_count = len([x for x, y in zip(X_val, y_val) if y == breed])
        test_count = len([x for x, y in zip(X_test, y_test) if y == breed])
        total = train_count + val_count + test_count
        percentage = (total / total_images) * 100
        
        print(f"{breed:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8} {percentage:<6.1f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {len(X_train):<8} {len(X_val):<8} {len(X_test):<8} {total_images:<8} {'100.0':<6}")
    
    # Save classes.json
    classes = sorted(unique_breeds.tolist())
    with open(target_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    
    print(f"\n✅ Classes saved to: {target_dir / 'classes.json'}")
    print(f"📁 Data ready at: {target_dir}")
    
    print(f"\n🎯 Ready for training!")
    print(f"📊 Total images: {total_images}")
    print(f"🏷️ Number of breeds: {len(unique_breeds)}")
    
    return {
        "train_size": len(X_train),
        "val_size": len(X_val), 
        "test_size": len(X_test),
        "total_size": total_images,
        "num_classes": len(unique_breeds),
        "classes": classes
    }

if __name__ == "__main__":
    stats = prepare_cow_data_v3()
    
    print("\n🚀 Next Steps:")
    print("1. Run: python scripts/train_cow_classifier_v3.py")
    print("2. Run: python scripts/evaluate_cow_model_v3.py")
    print("3. Update app.py to include V3 model")
