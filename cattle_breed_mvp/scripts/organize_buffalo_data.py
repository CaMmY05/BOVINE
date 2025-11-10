"""
Organize buffalo breed data from Roboflow downloads
Extract and structure buffalo breeds for training
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_buffalo_data():
    """
    Extract buffalo breeds from Roboflow dataset
    """
    print("="*60)
    print("ORGANIZING BUFFALO BREED DATA")
    print("="*60)
    
    # Source: Roboflow Indian Bovine Recognition dataset
    source_base = "data/research_datasets/roboflow/indian_bovine_recognition"
    
    # Destination: Final organized buffaloes
    dest_base = "data/final_organized/buffaloes"
    
    # Buffalo breeds to extract
    buffalo_breeds = {
        'Murrah': 'murrah',
        'Jaffrabadi': 'jaffarabadi',  # Note: spelled differently in dataset
        'Mehsana': 'mehsana',
        'Nili_Ravi': 'nili_ravi',
        'Surti': 'surti',
        'Bhadawari': 'bhadawari'
    }
    
    print(f"\nSource: {source_base}")
    print(f"Destination: {dest_base}")
    print(f"\nTarget breeds: {list(buffalo_breeds.values())}")
    
    total_copied = 0
    breed_counts = {}
    
    for roboflow_name, organized_name in buffalo_breeds.items():
        print(f"\n{'='*60}")
        print(f"Processing: {organized_name.upper()}")
        print(f"{'='*60}")
        
        # Create destination folder
        dest_breed_dir = os.path.join(dest_base, organized_name)
        os.makedirs(dest_breed_dir, exist_ok=True)
        
        breed_total = 0
        
        # Copy from train, valid, test splits
        for split in ['train', 'valid', 'test']:
            source_breed_dir = os.path.join(source_base, split, roboflow_name)
            
            if not os.path.exists(source_breed_dir):
                print(f"  {split}: Not found, skipping...")
                continue
            
            # Get all images
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(Path(source_breed_dir).glob(ext))
            
            if not images:
                print(f"  {split}: No images found")
                continue
            
            print(f"  {split}: Found {len(images)} images")
            
            # Copy images with prefix
            for img_path in tqdm(images, desc=f"  Copying {split}"):
                # Create unique filename with split prefix
                new_name = f"{split}_{img_path.name}"
                dest_path = os.path.join(dest_breed_dir, new_name)
                
                try:
                    shutil.copy2(str(img_path), dest_path)
                    breed_total += 1
                except Exception as e:
                    print(f"    Error copying {img_path.name}: {e}")
        
        breed_counts[organized_name] = breed_total
        total_copied += breed_total
        
        print(f"  ✓ Total for {organized_name}: {breed_total} images")
    
    # Summary
    print("\n" + "="*60)
    print("BUFFALO DATA ORGANIZATION COMPLETE!")
    print("="*60)
    print(f"\nTotal images organized: {total_copied}")
    print("\nBreakdown by breed:")
    for breed, count in breed_counts.items():
        print(f"  {breed:15s}: {count:4d} images")
    
    # Check if we have enough data
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    min_images = 100
    ready_breeds = []
    insufficient_breeds = []
    
    for breed, count in breed_counts.items():
        if count >= min_images:
            ready_breeds.append(breed)
            print(f"  ✓ {breed:15s}: {count:4d} images (READY)")
        else:
            insufficient_breeds.append(breed)
            print(f"  ⚠ {breed:15s}: {count:4d} images (LOW - need {min_images}+)")
    
    print(f"\nReady for training: {len(ready_breeds)} breeds")
    print(f"Need more data: {len(insufficient_breeds)} breeds")
    
    if len(ready_breeds) >= 3:
        print("\n✓ Sufficient data for 3+ buffalo breeds!")
        print(f"  Recommended breeds: {', '.join(ready_breeds[:3])}")
    else:
        print(f"\n⚠ Only {len(ready_breeds)} breeds have sufficient data")
        print("  Consider downloading additional buffalo images")
    
    return breed_counts, ready_breeds

if __name__ == "__main__":
    breed_counts, ready_breeds = organize_buffalo_data()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if len(ready_breeds) >= 3:
        print("1. Prepare buffalo data:")
        print("   python scripts\\prepare_buffalo_data.py")
        print()
        print("2. Train buffalo classifier:")
        print("   python scripts\\train_buffalo_classifier.py")
        print()
        print("3. Evaluate buffalo model:")
        print("   python scripts\\evaluate_buffalo_model.py")
    else:
        print("1. Download more buffalo images:")
        print("   python scripts\\download_buffalo_images.py")
        print()
        print("2. Then organize and train")
