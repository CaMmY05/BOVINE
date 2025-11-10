"""
Organize 5 cow breeds from Roboflow dataset for V3 model
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_cow_data_v3():
    """Extract and organize 5 cow breeds from Roboflow dataset"""
    
    # Source directories
    base_dir = Path("data/research_datasets/roboflow/indian_bovine_recognition")
    target_dir = Path("data/final_organized_v3/cows")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Select top 5 cow breeds by data availability
    # Based on train folder counts: Sahiwal (313), Gir (239), Holstein_Friesian (222), Red_Sindhi (109), Jersey (140)
    cow_breeds = {
        "sahiwal": "Sahiwal",
        "gir": "Gir", 
        "holstein_friesian": "Holstein Friesian",
        "red_sindhi": "Red Sindhi",
        "jersey": "Jersey"
    }
    
    print("🐄 Organizing 5 Cow Breeds for V3 Model...")
    print("="*60)
    
    total_images = 0
    breed_stats = {}
    
    # Process each split
    for split in ["train", "valid", "test"]:
        print(f"\n📁 Processing {split} split...")
        split_dir = base_dir / split
        split_target = target_dir / split
        split_target.mkdir(exist_ok=True)
        
        for breed_key, breed_name in cow_breeds.items():
            breed_source = split_dir / breed_key
            breed_target = split_target / breed_key
            
            if breed_source.exists():
                breed_target.mkdir(exist_ok=True)
                
                # Copy images
                image_files = list(breed_source.glob("*.jpg")) + list(breed_source.glob("*.jpeg")) + list(breed_source.glob("*.png"))
                
                print(f"  📋 {breed_name}: {len(image_files)} images")
                
                for img_file in tqdm(image_files, desc=f"  Copying {breed_name}", leave=False):
                    shutil.copy2(img_file, breed_target / img_file.name)
                
                if breed_key not in breed_stats:
                    breed_stats[breed_key] = {"name": breed_name, "train": 0, "valid": 0, "test": 0}
                breed_stats[breed_key][split] = len(image_files)
                total_images += len(image_files)
            else:
                print(f"  ⚠️ {breed_name}: Source directory not found")
    
    print("\n" + "="*60)
    print("📊 V3 Cow Dataset Summary:")
    print("="*60)
    
    print(f"{'Breed':<20} {'Train':<8} {'Valid':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)
    
    for breed_key, stats in breed_stats.items():
        total = stats["train"] + stats["valid"] + stats["test"]
        print(f"{stats['name']:<20} {stats['train']:<8} {stats['valid']:<8} {stats['test']:<8} {total:<8}")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {sum(s['train'] for s in breed_stats.values()):<8} {sum(s['valid'] for s in breed_stats.values()):<8} {sum(s['test'] for s in breed_stats.values()):<8} {total_images:<8}")
    
    print(f"\n✅ Successfully organized {total_images} images for 5 cow breeds!")
    print(f"📁 Location: {target_dir}")
    
    # Verify minimum images per breed
    print("\n🔍 Data Quality Check:")
    for breed_key, stats in breed_stats.items():
        total = stats["train"] + stats["valid"] + stats["test"]
        if total < 100:
            print(f"⚠️ {stats['name']}: Only {total} images (may be insufficient)")
        else:
            print(f"✅ {stats['name']}: {total} images (sufficient)")
    
    return breed_stats

if __name__ == "__main__":
    breed_stats = organize_cow_data_v3()
    
    print("\n🎯 Next Steps:")
    print("1. Run: python scripts/prepare_cow_data_v3.py")
    print("2. Run: python scripts/train_cow_classifier_v3.py") 
    print("3. Run: python scripts/evaluate_cow_model_v3.py")
    print("4. Update app.py to include V3 model option")
