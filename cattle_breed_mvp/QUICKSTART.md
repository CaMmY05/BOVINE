# 🚀 Quick Start Guide - Cattle Breed Detection MVP

## ⚡ 5-Minute Setup

### Step 1: Activate Virtual Environment

If you haven't created it yet:
```bash
cd cattle_breed_mvp
python -m venv cattle_mvp_env
```

Activate it:
```bash
# Windows PowerShell
.\cattle_mvp_env\Scripts\Activate.ps1

# Windows CMD
cattle_mvp_env\Scripts\activate.bat

# Linux/Mac
source cattle_mvp_env/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**This will take 5-10 minutes depending on your internet speed.**

### Step 3: Verify Installation

```bash
python -c "import torch; import ultralytics; print('✓ All packages installed successfully!')"
```

Check GPU (if available):
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 📊 Option A: Use Pre-collected Data

If you have cattle images ready:

1. **Organize your images:**
```
data/raw/
├── gir/
│   └── (your gir cattle images)
├── sahiwal/
│   └── (your sahiwal cattle images)
└── ... (other breeds)
```

2. **Run data preparation:**
```bash
python scripts/prepare_data.py
```

3. **Extract ROIs (optional but recommended):**
```bash
python scripts/extract_roi.py
```

4. **Train the model:**
```bash
python scripts/train_classifier.py
```

5. **Test with web app:**
```bash
streamlit run app.py
```

## 📥 Option B: Download Sample Dataset

### Using Roboflow (Easiest)

1. Visit: https://universe.roboflow.com/
2. Search: "cattle detection" or "cow detection"
3. Download dataset in "Folder Structure" format
4. Extract to `data/raw/`
5. Continue with Step 2 from Option A

### Using Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API token (get from kaggle.com/settings)
# Place kaggle.json in:
# Windows: C:\Users\<YourUsername>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d vikramamin/cattle-breed-classification-dataset
unzip cattle-breed-classification-dataset.zip -d data/raw/
```

## 🎯 Testing Without Training

If you just want to see how the system works:

1. **Download YOLOv8 (automatic on first run):**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

2. **Test detection only:**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict('your_cattle_image.jpg', classes=[19])
results[0].show()
```

## 🐛 Common Issues & Solutions

### Issue 1: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: "CUDA out of memory"
**Solution:** Reduce batch size in `scripts/train_classifier.py`:
```python
batch_size=8  # Change from 32 to 8
```

### Issue 3: "No images found"
**Solution:** Check your data structure:
```bash
# Windows
dir data\raw
# Linux/Mac
ls -R data/raw
```

### Issue 4: Virtual environment not activating
**Solution (Windows):**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📝 Minimal Working Example

Create a test script `test_detection.py`:

```python
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('yolov8n.pt')

# Test on an image
image_path = 'test_images/cattle_1.jpg'  # Replace with your image
results = model.predict(image_path, classes=[19], conf=0.4)

# Display results
for r in results:
    img = r.plot()
    cv2.imshow('Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Detection complete!")
```

Run it:
```bash
python test_detection.py
```

## 🎓 Learning Path

### Day 1: Setup & Understanding
- ✅ Install dependencies
- ✅ Understand project structure
- ✅ Test YOLO detection

### Day 2: Data Preparation
- ✅ Collect/download cattle images
- ✅ Organize into breed folders
- ✅ Run data preparation script

### Day 3: Training
- ✅ Extract ROIs
- ✅ Train classification model
- ✅ Monitor training progress

### Day 4: Evaluation & Demo
- ✅ Evaluate model performance
- ✅ Test inference pipeline
- ✅ Launch web app demo

## 🎯 Success Criteria

Your MVP is working if:
- ✅ YOLO detects cattle in images
- ✅ Model trains without errors
- ✅ Web app launches and accepts uploads
- ✅ Predictions are displayed with confidence scores

## 📞 Need Help?

Check these files:
- `README.md` - Comprehensive documentation
- `scripts/` - All Python scripts with comments
- `requirements.txt` - All dependencies

## 🚀 Next Steps After MVP

1. **Improve accuracy:**
   - Collect more data (100+ images per breed)
   - Train for more epochs
   - Try different models (ResNet50, EfficientNet-B3)

2. **Add features:**
   - Video processing
   - Batch processing
   - API endpoint
   - Mobile app

3. **Deploy:**
   - Docker containerization
   - Cloud deployment (AWS, Azure, GCP)
   - Edge deployment (Raspberry Pi, Jetson)

---

**Remember:** This is an MVP! Focus on getting it working first, then iterate and improve. 🎉
