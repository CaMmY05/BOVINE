"""
Remove duplicate images using perceptual hashing
"""

import os
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm
import shutil

def find_duplicates(image_dir, hash_size=8, threshold=5):
    """
    Find duplicate images using perceptual hashing
    
    Args:
        image_dir: Directory containing images
        hash_size: Size of hash (larger = more precise)
        threshold: Hamming distance threshold (0 = exact duplicates)
    
    Returns:
        List of duplicate image groups
    """
    print(f"\nScanning images in: {image_dir}")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(image_dir).rglob(ext))
    
    if not image_files:
        print("No images found!")
        return []
    
    print(f"Found {len(image_files)} images")
    
    # Calculate hashes
    print("\nCalculating image hashes...")
    hashes = {}
    failed = []
    
    for img_path in tqdm(image_files):
        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate perceptual hash
                img_hash = imagehash.phash(img, hash_size=hash_size)
                hashes[str(img_path)] = img_hash
        except Exception as e:
            failed.append((str(img_path), str(e)))
    
    if failed:
        print(f"\n⚠️  Failed to process {len(failed)} images:")
        for path, error in failed[:5]:  # Show first 5
            print(f"  - {Path(path).name}: {error}")
    
    # Find duplicates
    print("\nFinding duplicates...")
    duplicates = defaultdict(list)
    processed = set()
    
    hash_list = list(hashes.items())
    for i, (path1, hash1) in enumerate(tqdm(hash_list)):
        if path1 in processed:
            continue
        
        group = [path1]
        for path2, hash2 in hash_list[i+1:]:
            if path2 in processed:
                continue
            
            # Calculate Hamming distance
            distance = hash1 - hash2
            if distance <= threshold:
                group.append(path2)
                processed.add(path2)
        
        if len(group) > 1:
            duplicates[hash1].extend(group)
            processed.add(path1)
    
    return duplicates

def remove_duplicates_interactive(duplicates):
    """
    Interactively remove duplicates
    """
    if not duplicates:
        print("\n✓ No duplicates found!")
        return 0
    
    print(f"\n{'='*60}")
    print(f"Found {len(duplicates)} groups of duplicates")
    print(f"{'='*60}")
    
    total_to_remove = sum(len(group) - 1 for group in duplicates.values())
    print(f"Total duplicate images: {total_to_remove}")
    
    print("\nOptions:")
    print("1. Auto-remove (keep first, delete rest)")
    print("2. Review each group manually")
    print("3. Move duplicates to separate folder")
    print("4. Cancel")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    removed_count = 0
    
    if choice == '1':
        # Auto-remove
        print("\nRemoving duplicates...")
        for group in tqdm(duplicates.values()):
            # Keep first, remove rest
            for img_path in group[1:]:
                try:
                    os.remove(img_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {img_path}: {e}")
        
        print(f"\n✓ Removed {removed_count} duplicate images")
    
    elif choice == '2':
        # Manual review
        for idx, group in enumerate(duplicates.values(), 1):
            print(f"\n{'='*60}")
            print(f"Duplicate Group {idx}/{len(duplicates)}")
            print(f"{'='*60}")
            
            for i, img_path in enumerate(group):
                print(f"{i+1}. {Path(img_path).name} ({os.path.getsize(img_path) / 1024:.1f} KB)")
            
            keep = input(f"Keep which image? (1-{len(group)}, 'a' for all, 's' to skip): ").strip()
            
            if keep == 's':
                continue
            elif keep == 'a':
                continue
            else:
                try:
                    keep_idx = int(keep) - 1
                    if 0 <= keep_idx < len(group):
                        for i, img_path in enumerate(group):
                            if i != keep_idx:
                                os.remove(img_path)
                                removed_count += 1
                                print(f"  ✓ Removed {Path(img_path).name}")
                except:
                    print("Invalid input, skipping group")
        
        print(f"\n✓ Removed {removed_count} duplicate images")
    
    elif choice == '3':
        # Move to separate folder
        dup_dir = 'data/duplicates'
        os.makedirs(dup_dir, exist_ok=True)
        
        print(f"\nMoving duplicates to: {dup_dir}")
        for group in tqdm(duplicates.values()):
            for img_path in group[1:]:
                try:
                    dest = Path(dup_dir) / Path(img_path).name
                    shutil.move(img_path, dest)
                    removed_count += 1
                except Exception as e:
                    print(f"Error moving {img_path}: {e}")
        
        print(f"\n✓ Moved {removed_count} duplicate images to {dup_dir}")
    
    else:
        print("Cancelled")
    
    return removed_count

def main():
    """Main function"""
    print("="*60)
    print("DUPLICATE IMAGE REMOVER")
    print("="*60)
    
    # Get directory
    default_dir = 'data/raw_downloads'
    dir_path = input(f"\nEnter directory path (default: {default_dir}): ").strip() or default_dir
    
    if not os.path.exists(dir_path):
        print(f"✗ Directory not found: {dir_path}")
        return
    
    # Get threshold
    print("\nDuplicate detection threshold:")
    print("  0 = Exact duplicates only")
    print("  5 = Very similar images (recommended)")
    print("  10 = Similar images")
    threshold = int(input("Enter threshold (default: 5): ").strip() or "5")
    
    # Find duplicates
    duplicates = find_duplicates(dir_path, threshold=threshold)
    
    # Remove duplicates
    if duplicates:
        removed = remove_duplicates_interactive(duplicates)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Duplicate groups found: {len(duplicates)}")
        print(f"Images removed/moved: {removed}")
        print(f"\n✓ Done!")
    else:
        print("\n✓ No duplicates found!")

if __name__ == "__main__":
    # Check if imagehash is installed
    try:
        import imagehash
    except ImportError:
        print("✗ imagehash not installed!")
        print("Install with: pip install imagehash")
        exit(1)
    
    main()
