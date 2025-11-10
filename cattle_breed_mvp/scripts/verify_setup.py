"""
Setup verification script for Cattle Breed Detection MVP
Run this to check if your environment is properly configured
"""

import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print("\n[1/8] Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\n[2/8] Checking required packages...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'ultralytics': 'Ultralytics (YOLO)',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
        'streamlit': 'Streamlit'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print("\n[3/8] Checking CUDA/GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA Version: {torch.version.cuda}")
            print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("  ⚠ CUDA not available - will use CPU (slower)")
            print("  Note: For faster training, install CUDA-enabled PyTorch")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_directory_structure():
    """Check if project directories exist"""
    print("\n[4/8] Checking directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models/detection',
        'models/classification',
        'results/evaluation',
        'results/predictions',
        'scripts',
        'test_images'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_yolo_model():
    """Check if YOLO model can be loaded"""
    print("\n[5/8] Checking YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("  ✓ YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error loading YOLO: {e}")
        print("  Note: Model will be downloaded on first use")
        return False

def check_data():
    """Check if data is available"""
    print("\n[6/8] Checking dataset...")
    
    raw_data_path = 'data/raw'
    if not os.path.exists(raw_data_path):
        print(f"  ✗ {raw_data_path} not found")
        return False
    
    breeds = [d for d in os.listdir(raw_data_path) 
              if os.path.isdir(os.path.join(raw_data_path, d))]
    
    if len(breeds) == 0:
        print("  ✗ No breed folders found in data/raw/")
        print("  Action: Add cattle images organized by breed")
        return False
    
    print(f"  ✓ Found {len(breeds)} breed folders:")
    total_images = 0
    for breed in breeds:
        breed_path = os.path.join(raw_data_path, breed)
        images = [f for f in os.listdir(breed_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        print(f"    - {breed}: {len(images)} images")
    
    if total_images == 0:
        print("  ✗ No images found")
        return False
    
    print(f"  ✓ Total images: {total_images}")
    
    if total_images < 50:
        print("  ⚠ Warning: Very few images. Recommend 50+ per breed for good results")
    
    return True

def check_processed_data():
    """Check if processed data exists"""
    print("\n[7/8] Checking processed data...")
    
    classes_file = 'data/processed/classes.json'
    if os.path.exists(classes_file):
        import json
        with open(classes_file, 'r') as f:
            classes = json.load(f)
        print(f"  ✓ classes.json found ({len(classes)} classes)")
        
        # Check splits
        for split in ['train', 'val', 'test']:
            split_dir = f'data/processed/{split}/images'
            if os.path.exists(split_dir):
                images = [f for f in os.listdir(split_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  ✓ {split}: {len(images)} images")
            else:
                print(f"  ✗ {split} split not found")
        
        return True
    else:
        print("  ✗ Processed data not found")
        print("  Action: Run 'python scripts/prepare_data.py'")
        return False

def check_trained_model():
    """Check if trained model exists"""
    print("\n[8/8] Checking trained model...")
    
    model_path = 'models/classification/breed_classifier_v1/best_model.pth'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ Trained model found ({size_mb:.2f} MB)")
        return True
    else:
        print("  ✗ Trained model not found")
        print("  Action: Run 'python scripts/train_classifier.py'")
        return False

def print_summary(results):
    """Print summary and recommendations"""
    print_header("SETUP VERIFICATION SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. If you haven't prepared data: python scripts/prepare_data.py")
        print("2. If you haven't trained: python scripts/train_classifier.py")
        print("3. Launch web app: streamlit run app.py")
    else:
        print("\n⚠ Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        
        if not results['packages']:
            print("- Install packages: pip install -r requirements.txt")
        
        if not results['directories']:
            print("- Project structure issue - ensure you're in the right directory")
        
        if not results['data']:
            print("- Add cattle images to data/raw/<breed_name>/")
            print("- Or run: python scripts/download_sample_data.py")
        
        if not results['processed_data']:
            print("- Run: python scripts/prepare_data.py")
        
        if not results['trained_model']:
            print("- Run: python scripts/train_classifier.py")

def main():
    print_header("CATTLE BREED DETECTION MVP - SETUP VERIFICATION")
    
    results = {
        'python': check_python_version(),
        'packages': check_packages(),
        'cuda': check_cuda(),
        'directories': check_directory_structure(),
        'yolo': check_yolo_model(),
        'data': check_data(),
        'processed_data': check_processed_data(),
        'trained_model': check_trained_model()
    }
    
    print_summary(results)
    
    print("\n" + "="*60)
    print("For detailed instructions, see:")
    print("  - README.md (comprehensive guide)")
    print("  - QUICKSTART.md (quick setup)")
    print("="*60)

if __name__ == "__main__":
    main()
