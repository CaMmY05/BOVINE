import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Selected breeds
SELECTED_BREEDS = ['jaffarabadi', 'mehsana', 'murrah']

def copy_files(src, dst, class_name, file_list):
    """Copy files from src to dst for a specific class."""
    os.makedirs(dst, exist_ok=True)
    class_dst = os.path.join(dst, class_name)
    os.makedirs(class_dst, exist_ok=True)
    
    for img in tqdm(file_list, desc=f"Copying {class_name}"):
        src_path = os.path.join(src, class_name, img)
        dst_path = os.path.join(class_dst, img)
        shutil.copy2(src_path, dst_path)

def prepare_dataset(src_data_dir, output_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Prepare dataset with only 3 selected buffalo breeds.
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    print(f"Selected breeds: {', '.join(SELECTED_BREEDS)}")
    
    # Process each selected class
    for cls in SELECTED_BREEDS:
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
            test_size=val_size/(1-test_size),
            random_state=random_state,
            shuffle=True
        )
        
        print(f"  Train: {len(train)} images")
        print(f"  Val: {len(val)} images")
        print(f"  Test: {len(test)} images")
        
        # Copy files to respective directories
        copy_files(src_data_dir, train_dir, cls, train)
        copy_files(src_data_dir, val_dir, cls, val)
        copy_files(src_data_dir, test_dir, cls, test)
    
    print(f"\nDataset prepared successfully in {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare 3-breed buffalo dataset for training')
    parser.add_argument('--src_dir', type=str, required=True,
                       help='Source directory containing class subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for the prepared dataset')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Proportion of training data for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.src_dir,
        args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
