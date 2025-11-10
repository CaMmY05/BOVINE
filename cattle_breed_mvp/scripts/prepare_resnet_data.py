import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def copy_files(src, dst):
    """Copy files from src to dst, creating directories if they don't exist."""
    os.makedirs(dst, exist_ok=True)
    for item in tqdm(os.listdir(src), desc=f"Copying {os.path.basename(src)}"):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)

def prepare_dataset(src_data_dir, output_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Prepare dataset by splitting into train/val/test sets.
    
    Args:
        src_data_dir: Source directory containing class subdirectories
        output_dir: Output directory to save the split dataset
        test_size: Proportion of data to use for test set
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    # Get list of classes
    classes = [d for d in os.listdir(src_data_dir) 
              if os.path.isdir(os.path.join(src_data_dir, d))]
    
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Process each class
    for cls in classes:
        cls_src = os.path.join(src_data_dir, cls)
        
        # Get list of image files
        images = [f for f in os.listdir(cls_src) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"Warning: No images found in {cls_src}")
            continue
            
        print(f"\nProcessing class: {cls} ({len(images)} images)")
        
        # Split into train+val and test
        train_val, test = train_test_split(
            images, 
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Split train into train and validation
        train, val = train_test_split(
            train_val,
            test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val
            random_state=random_state,
            shuffle=True
        )
        
        print(f"  Train: {len(train)} images")
        print(f"  Val: {len(val)} images")
        print(f"  Test: {len(test)} images")
        
        # Create class directories
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        # Copy files to respective directories
        for img in train:
            src = os.path.join(cls_src, img)
            dst = os.path.join(train_dir, cls, img)
            shutil.copy2(src, dst)
            
        for img in val:
            src = os.path.join(cls_src, img)
            dst = os.path.join(val_dir, cls, img)
            shutil.copy2(src, dst)
            
        for img in test:
            src = os.path.join(cls_src, img)
            dst = os.path.join(test_dir, cls, img)
            shutil.copy2(src, dst)
    
    print(f"\nDataset prepared successfully in {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for ResNet-18 training')
    parser.add_argument('--src_dir', type=str, required=True, 
                       help='Source directory containing class subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory to save the prepared dataset')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Proportion of data to use for test set (default: 0.15)')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Proportion of training data to use for validation (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    prepare_dataset(
        src_data_dir=args.src_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
