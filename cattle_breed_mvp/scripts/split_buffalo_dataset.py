import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(input_dir, output_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split dataset into train/val/test sets while maintaining class distribution.
    
    Args:
        input_dir: Path to the directory containing class folders
        output_dir: Path where to create train/val/test directories
        test_size: Proportion of data for test set
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Get list of classes (subdirectories in input_dir)
    classes = [d for d in os.listdir(input_dir) 
              if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(classes)} classes: {classes}")
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Create class directories in train/val/test
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
        
        # Get all image files for this class
        class_dir = os.path.join(input_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  Total images: {len(images)}")
        
        if len(images) == 0:
            print(f"  Warning: No images found in {class_dir}")
            continue
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            images, 
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=val_size/(1-test_size),  # Adjust for the initial split
            random_state=random_state
        )
        
        print(f"  Split: {len(train)} train, {len(val)} val, {len(test)} test")
        
        # Copy files to their respective directories
        def copy_files(files, split):
            for f in tqdm(files, desc=f"  Copying {split} files", leave=False):
                src = os.path.join(class_dir, f)
                dst = os.path.join(output_dir, split, class_name, f)
                shutil.copy2(src, dst)
        
        copy_files(train, 'train')
        copy_files(val, 'val')
        copy_files(test, 'test')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the directory containing class folders')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path where to create train/val/test directories')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for test set (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Proportion of training data for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Splitting dataset from {args.input_dir} to {args.output_dir}")
    print(f"Test size: {args.test_size}, Val size: {args.val_size}, Random seed: {args.seed}")
    
    split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    print("\nDataset splitting completed successfully!")
