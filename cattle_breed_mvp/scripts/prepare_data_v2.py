"""
Prepare data for training - Version 2
Works with new organized structure
Creates train/val/test splits with proper organization
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

def prepare_dataset(source_dir, output_dir, breeds, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Organize images into train/val/test splits
    """
    print(f"Preparing dataset from: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Breeds: {breeds}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for breed in breeds:
            os.makedirs(os.path.join(output_dir, split, breed), exist_ok=True)
    
    all_images = []
    breed_counts = {}
    
    # Collect all images
    for breed in breeds:
        breed_dir = os.path.join(source_dir, breed)
        if not os.path.exists(breed_dir):
            print(f"Warning: {breed_dir} not found, skipping...")
            continue
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(Path(breed_dir).glob(ext))
        
        breed_counts[breed] = len(images)
        all_images.extend([(str(img), breed) for img in images])
    
    print(f"\nTotal images found: {len(all_images)}")
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count}")
    
    # Split by breed to maintain balance
    train_data = []
    val_data = []
    test_data = []
    
    for breed in breeds:
        breed_images = [img for img, b in all_images if b == breed]
        
        if len(breed_images) == 0:
            continue
        
        # First split: train vs (val+test)
        train_imgs, temp_imgs = train_test_split(
            breed_images, 
            train_size=train_ratio, 
            random_state=42
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            train_size=val_size,
            random_state=42
        )
        
        train_data.extend([(img, breed) for img in train_imgs])
        val_data.extend([(img, breed) for img in val_imgs])
        test_data.extend([(img, breed) for img in test_imgs])
    
    # Copy images to splits
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        print(f"\nProcessing {split_name} split: {len(split_data)} images")
        
        for img_path, breed in tqdm(split_data):
            src = img_path
            dst = os.path.join(output_dir, split_name, breed, Path(img_path).name)
            
            try:
                # Verify image can be opened
                with Image.open(src) as img:
                    img.verify()
                
                # Copy image
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print("\nDataset preparation complete!")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Print per-breed distribution
    print("\nPer-breed distribution:")
    for breed in breeds:
        train_count = len([1 for _, b in train_data if b == breed])
        val_count = len([1 for _, b in val_data if b == breed])
        test_count = len([1 for _, b in test_data if b == breed])
        total = train_count + val_count + test_count
        print(f"  {breed}:")
        print(f"    Train: {train_count}, Val: {val_count}, Test: {test_count}, Total: {total}")

# Example usage
if __name__ == "__main__":
    # Cow breeds
    COW_BREEDS = ['gir', 'sahiwal', 'red_sindhi']
    
    COW_SOURCE = "data/final_organized/cows/"
    COW_OUTPUT = "data/processed_v2/cows/"
    
    print("="*60)
    print("PREPARING COW DATASET")
    print("="*60)
    prepare_dataset(COW_SOURCE, COW_OUTPUT, COW_BREEDS)
    
    # Buffalo breeds (if data available)
    BUFFALO_BREEDS = ['murrah', 'jaffarabadi', 'mehsana']
    
    BUFFALO_SOURCE = "data/final_organized/buffaloes/"
    BUFFALO_OUTPUT = "data/processed_v2/buffaloes/"
    
    # Check if buffalo data exists
    buffalo_exists = any(
        Path(BUFFALO_SOURCE, breed).exists() and 
        len(list(Path(BUFFALO_SOURCE, breed).glob('*.jpg'))) > 0
        for breed in BUFFALO_BREEDS
    )
    
    if buffalo_exists:
        print("\n" + "="*60)
        print("PREPARING BUFFALO DATASET")
        print("="*60)
        prepare_dataset(BUFFALO_SOURCE, BUFFALO_OUTPUT, BUFFALO_BREEDS)
    else:
        print("\n" + "="*60)
        print("BUFFALO DATA NOT READY")
        print("="*60)
        print("Buffalo folders exist but no images found.")
        print("See BUFFALO_DATASET_GUIDE.md for data collection.")
