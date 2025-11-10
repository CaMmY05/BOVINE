import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Base directories
base_dir = Path('data/research_datasets/roboflow/indian_bovine_recognition')
target_dir = Path('data/final_organized_v3/cows')

# Create target directories
for split in ['train', 'valid', 'test']:
    for breed in ['Gir', 'Sahiwal', 'Red Sindhi', 'Holstein Friesian', 'Jersey']:
        (target_dir / split / breed).mkdir(parents=True, exist_ok=True)

# Mapping from source breed names to target breed names
breed_mapping = {
    'sahiwal': 'Sahiwal',
    'gir': 'Gir',
    'holstein_friesian': 'Holstein Friesian',
    'jersey': 'Jersey',
    'red_sindhi': 'Red Sindhi'
}

# Process each split
for split in ['train', 'valid', 'test']:
    print(f"Processing {split} split...")
    split_dir = base_dir / split
    
    # Process each breed
    for breed_src, breed_dest in breed_mapping.items():
        breed_dir = split_dir / breed_src
        if not breed_dir.exists():
            print(f"Warning: {breed_dir} does not exist, skipping...")
            continue
            
        # Get all image files
        image_files = list(breed_dir.glob('*.jpg')) + list(breed_dir.glob('*.jpeg')) + list(breed_dir.glob('*.png'))
        
        # Copy images to target directory
        for img_path in tqdm(image_files, desc=f"Copying {breed_dest} images"):
            target_path = target_dir / split / breed_dest / img_path.name
            if not target_path.exists():
                shutil.copy2(img_path, target_path)
    
    print(f"Completed processing {split} split.\n")

# Print statistics
print("\nData organization complete. Summary:")
for split in ['train', 'valid', 'test']:
    print(f"\n{split.capitalize()}:")
    for breed in ['Gir', 'Sahiwal', 'Red Sindhi', 'Holstein Friesian', 'Jersey']:
        count = len(list((target_dir / split / breed).glob('*.*')))
        print(f"  {breed}: {count} images")
