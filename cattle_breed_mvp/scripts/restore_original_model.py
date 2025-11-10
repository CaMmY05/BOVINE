"""
Restore original model and data setup
This script will:
1. Backup the current (new) model
2. Restore original data counts
3. Keep original model as the working model
"""

import os
import shutil
from pathlib import Path

def backup_current_model():
    """Backup the current model to a separate folder"""
    print("="*60)
    print("STEP 1: BACKING UP CURRENT MODEL")
    print("="*60)
    
    source = Path('models/classification/breed_classifier_v1')
    backup = Path('models/classification/breed_classifier_v2_expanded_data')
    
    if backup.exists():
        print(f"Backup already exists at: {backup}")
        return
    
    # Copy entire folder
    shutil.copytree(source, backup)
    print(f"✓ Backed up current model to: {backup}")
    print(f"  - This model has 67.91% accuracy (with expanded data)")
    print(f"  - Saved for future reference")

def check_original_model():
    """Check if we need to restore the original model"""
    print(f"\n{'='*60}")
    print("STEP 2: CHECKING ORIGINAL MODEL")
    print("="*60)
    
    model_path = Path('models/classification/breed_classifier_v1/best_model.pth')
    
    if not model_path.exists():
        print("✗ Original model not found!")
        print("  We need to retrain with original data")
        return False
    
    print(f"✓ Model exists at: {model_path}")
    print(f"  Note: This is currently the NEW model (67.91% accuracy)")
    print(f"  We need to retrain with ORIGINAL data to get 75.65% accuracy back")
    return True

def restore_original_data():
    """Restore original data by removing new downloads"""
    print(f"\n{'='*60}")
    print("STEP 3: RESTORING ORIGINAL DATA")
    print("="*60)
    
    breeds = ['gir', 'sahiwal', 'red_sindhi']
    
    for breed in breeds:
        breed_dir = Path(f'data/raw/{breed}')
        if not breed_dir.exists():
            continue
        
        # Count current images
        current_count = len(list(breed_dir.glob('*.jpg'))) + len(list(breed_dir.glob('*.png')))
        
        # Remove images with "download_" prefix (these are new)
        removed = 0
        for img in breed_dir.glob('download_*'):
            img.unlink()
            removed += 1
        
        final_count = len(list(breed_dir.glob('*.jpg'))) + len(list(breed_dir.glob('*.png')))
        
        print(f"\n{breed}:")
        print(f"  Before: {current_count} images")
        print(f"  Removed: {removed} new images")
        print(f"  After: {final_count} images (original)")

def main():
    """Main restoration process"""
    print("="*60)
    print("RESTORING ORIGINAL MODEL & DATA")
    print("="*60)
    print("\nThis will:")
    print("1. Backup current model (67.91% accuracy)")
    print("2. Remove new downloaded images")
    print("3. Restore original data (947 images)")
    print("4. Prepare for retraining with original data")
    print()
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Backup current model
    backup_current_model()
    
    # Step 2: Check original model
    check_original_model()
    
    # Step 3: Restore original data
    restore_original_data()
    
    print(f"\n{'='*60}")
    print("RESTORATION COMPLETE!")
    print("="*60)
    print("\nOriginal data restored:")
    
    breeds = ['gir', 'sahiwal', 'red_sindhi']
    total = 0
    for breed in breeds:
        breed_dir = Path(f'data/raw/{breed}')
        if breed_dir.exists():
            count = len(list(breed_dir.glob('*.jpg'))) + len(list(breed_dir.glob('*.png')))
            print(f"  {breed:15s}: {count} images")
            total += count
    
    print(f"  {'TOTAL':15s}: {total} images")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("="*60)
    print("1. Retrain with original data:")
    print("   python scripts\\prepare_data.py")
    print("   python scripts\\extract_roi.py")
    print("   python scripts\\train_classifier.py")
    print()
    print("2. This will restore 75.65% accuracy")
    print()
    print("3. Then we can download quality datasets")
    print("   and add them selectively")

if __name__ == "__main__":
    main()
