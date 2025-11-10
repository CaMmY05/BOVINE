import os
import shutil
from pathlib import Path

def create_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def copy_model(src, dst):
    """Copy model files to destination"""
    src = str(src)
    dst = str(dst)
    
    if not os.path.exists(src):
        print(f"Warning: Source file not found: {src}")
        return False
        
    if os.path.abspath(src) == os.path.abspath(dst):
        print(f"Skipping (same file): {src}")
        return True
        
    print(f"Copying {src} to {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False

def organize_models():
    base_dir = Path("models/classification")
    
    # Create target directories
    targets = [
        # Format: (source_path, target_path, is_directory)
        
        # EfficientNet-B0 V1 (Cow)
        ('breed_classifier_v1/final_model.pth', 'breed_classifier_v1/final_model.pth'),
        ('breed_classifier_v1/classes.json', 'breed_classifier_v1/classes.json'),
        
        # EfficientNet-B0 V1 (Buffalo)
        ('buffalo_classifier_v1/final_model.pth', 'buffalo_classifier_v1/final_model.pth'),
        ('buffalo_classifier_v1/classes.json', 'buffalo_classifier_v1/classes.json'),
        
        # EfficientNet-B0 V2 (Cow)
        ('cow_classifier_v2/final_model.pth', 'cow_classifier_v2/final_model.pth'),
        ('cow_classifier_v2/classes.json', 'cow_classifier_v2/classes.json'),
        
        # EfficientNet-B0 V3 (Cow)
        ('cow_classifier_v3/final_model.pth', 'cow_classifier_v3/final_model.pth'),
        ('cow_classifier_v3/classes.json', 'cow_classifier_v3/classes.json'),
        
        # ResNet18 (Cow)
        ('resnet18_cow_v1/best_model.pth', 'resnet18_cow_v1/checkpoints/best_model.pth'),
        ('resnet18_cow_v1/classes.json', 'resnet18_cow_v1/classes.json'),
        
        # ResNet18 (Buffalo)
        ('resnet18_buffalo_v1/best_model.pth', 'resnet18_buffalo_v1/checkpoints/best_model.pth'),
        ('resnet18_buffalo_v1/classes.json', 'resnet18_buffalo_v1/classes.json'),
        
        # ResNet34 as ResNet32 (Cow)
        ('resnet34_cow_v1/final_model.pth', 'resnet34_cow_v1/checkpoints/best_model.pth'),
        ('resnet34_cow_v1/class_to_idx.json', 'resnet34_cow_v1/classes.json'),
        
        # ResNet34 as ResNet32 (Buffalo)
        ('resnet34_buffalo_v1/final_model.pth', 'resnet34_buffalo_v1/checkpoints/best_model.pth'),
        ('resnet34_buffalo_v1/class_to_idx.json', 'resnet34_buffalo_v1/classes.json'),
    ]
    
    # Process each model
    for src, dst in targets:
        src_path = base_dir / src
        dst_path = base_dir / dst
        copy_model(src_path, dst_path)
    
    print("\nModel organization complete!")

if __name__ == "__main__":
    organize_models()
