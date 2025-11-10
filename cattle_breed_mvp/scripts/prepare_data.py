import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import json

def organize_dataset(raw_dir, output_dir, breeds_list):
    """
    Organizes raw images into train/val/test splits
    """
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
    
    # Class mapping
    class_to_idx = {breed: idx for idx, breed in enumerate(breeds_list)}
    
    # Save class mapping
    with open(f"{output_dir}/classes.json", 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    all_images = []
    all_labels = []
    
    # Collect all images with labels
    for breed in breeds_list:
        breed_dir = os.path.join(raw_dir, breed)
        if os.path.exists(breed_dir):
            for img_file in os.listdir(breed_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(breed_dir, img_file))
                    all_labels.append(class_to_idx[breed])
    
    print(f"Total images found: {len(all_images)}")
    
    if len(all_images) == 0:
        print("ERROR: No images found! Please organize your raw data into breed folders.")
        print(f"Expected structure: {raw_dir}/breed_name/image.jpg")
        return
    
    # Split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls, test_size=0.5, random_state=42, stratify=temp_lbls
    )
    
    # Copy files to respective splits
    splits = {
        'train': (train_imgs, train_lbls),
        'val': (val_imgs, val_lbls),
        'test': (test_imgs, test_lbls)
    }
    
    for split_name, (images, labels) in splits.items():
        print(f"\nProcessing {split_name} split: {len(images)} images")
        
        for idx, (img_path, label) in enumerate(zip(images, labels)):
            # Copy image
            new_img_name = f"{split_name}_{idx:05d}.jpg"
            dest_img = f"{output_dir}/{split_name}/images/{new_img_name}"
            
            # Resize and save image
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((640, 640))  # Standard YOLO size
                img.save(dest_img, quality=95)
                
                # Create label file (class index for classification)
                label_file = f"{output_dir}/{split_name}/labels/{new_img_name.replace('.jpg', '.txt')}"
                with open(label_file, 'w') as f:
                    f.write(str(label))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print("\nDataset preparation complete!")
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Example usage
if __name__ == "__main__":
    # List your breed classes (Indian Bovine Breeds dataset - 3 cow breeds for MVP)
    BREEDS = [
        'gir',
        'sahiwal', 
        'red_sindhi'
    ]
    
    RAW_DIR = "data/raw/"  # Your raw dataset location
    OUTPUT_DIR = "data/processed/"
    
    organize_dataset(RAW_DIR, OUTPUT_DIR, BREEDS)
