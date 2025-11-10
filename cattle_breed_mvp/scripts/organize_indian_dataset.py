"""
Organize Indian Bovine Breeds dataset for training
Copies selected breeds to data/raw/ folder
"""

import os
import shutil
from pathlib import Path

# Selected breeds for MVP
SELECTED_BREEDS = {
    'Sahiwal': 'sahiwal',
    'Gir': 'gir',
    'Red_Sindhi': 'red_sindhi'
}

# Paths
SOURCE_BASE = Path('data/raw/indian_bovine/Indian_bovine_breeds/Indian_bovine_breeds')
DEST_BASE = Path('data/raw')

def organize_dataset():
    """Copy selected breeds to proper structure"""
    
    print("="*60)
    print("ORGANIZING INDIAN BOVINE BREEDS DATASET")
    print("="*60)
    
    for source_name, dest_name in SELECTED_BREEDS.items():
        source_path = SOURCE_BASE / source_name
        dest_path = DEST_BASE / dest_name
        
        if not source_path.exists():
            print(f"\n✗ Source not found: {source_path}")
            continue
        
        # Create destination directory
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        print(f"\n📂 Processing {source_name}...")
        image_count = 0
        
        for img_file in source_path.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest_file = dest_path / img_file.name
                shutil.copy2(img_file, dest_file)
                image_count += 1
        
        print(f"   ✓ Copied {image_count} images to {dest_name}/")
    
    print("\n" + "="*60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("="*60)
    
    # Summary
    print("\nBreed Distribution:")
    for dest_name in SELECTED_BREEDS.values():
        dest_path = DEST_BASE / dest_name
        if dest_path.exists():
            count = len(list(dest_path.glob('*.[jp][pn][g]*')))
            print(f"  - {dest_name:15s}: {count:4d} images")
    
    total = sum(len(list((DEST_BASE / dest_name).glob('*.[jp][pn][g]*'))) 
                for dest_name in SELECTED_BREEDS.values() 
                if (DEST_BASE / dest_name).exists())
    
    print(f"\n  Total: {total} images")
    print(f"  Expected split (70/15/15):")
    print(f"    - Train: ~{int(total * 0.7)} images")
    print(f"    - Val:   ~{int(total * 0.15)} images")
    print(f"    - Test:  ~{int(total * 0.15)} images")
    
    print("\n✅ Ready for data preparation!")
    print("   Next step: python scripts\\prepare_data.py")

if __name__ == "__main__":
    organize_dataset()
