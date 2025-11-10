# 🐄 Cattle Breed Detection MVP with YOLO

A complete MVP (Minimum Viable Product) for cattle breed detection and classification using YOLOv8 for detection and deep learning for breed classification.

## 🎯 Features

- **YOLO-based Detection**: Automatic cattle detection in images using YOLOv8
- **Breed Classification**: Deep learning-based breed recognition using EfficientNet-B0
- **Multi-View Analysis**: Three-region analysis (Left, Front, Right) for enhanced accuracy
- **Web Interface**: Interactive Streamlit app for easy testing
- **Comprehensive Pipeline**: End-to-end solution from data preparation to inference

## 🏗️ Architecture

```
Input Image → YOLO Detection → Extract Animal ROI → 
Three-View Analysis (Optional) → Breed Classification → 
Ensemble Scoring → Final Prediction
```

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, RTX 4000 Ada or better)
- 8GB+ GPU memory
- 16GB+ RAM

## 🚀 Quick Start

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
conda create -n cattle_mvp python=3.10 -y
conda activate cattle_mvp
```

#### Option B: Using venv
```bash
python -m venv cattle_mvp_env
# Windows
cattle_mvp_env\Scripts\activate
# Linux/Mac
source cattle_mvp_env/bin/activate
```

### 2. Install Dependencies

```bash
cd cattle_breed_mvp
pip install -r requirements.txt
```

### 3. Verify GPU Setup (Optional but Recommended)

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

## 📊 Data Preparation

### Step 1: Organize Your Dataset

Create the following structure in `data/raw/`:

```
data/raw/
├── gir/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── sahiwal/
│   ├── img001.jpg
│   └── ...
├── red_sindhi/
├── murrah_buffalo/
└── mehsana_buffalo/
```

### Step 2: Download Sample Datasets (Optional)

**Option A: Roboflow**
- Visit: https://universe.roboflow.com/
- Search for "cattle detection" or "cow detection"
- Download in YOLO format

**Option B: Kaggle**
```bash
pip install kaggle
kaggle datasets download -d vikramamin/cattle-breed-classification-dataset
unzip cattle-breed-classification-dataset.zip -d data/raw/
```

**Option C: Bristol Dataset**
- Visit: https://data.bris.ac.uk/data/dataset/2inu67jru7a6821kkgehxg3cv2
- Download and extract to `data/raw/`

### Step 3: Prepare Dataset

```bash
python scripts/prepare_data.py
```

This will:
- Split data into train/val/test (70/15/15)
- Resize images to 640x640
- Create label files
- Generate `classes.json`

## 🎯 Training Pipeline

### Step 1: Extract ROIs (Optional but Recommended)

```bash
python scripts/extract_roi.py
```

This uses YOLOv8 to detect and crop cattle from images, improving classification accuracy.

### Step 2: Train Classification Model

```bash
python scripts/train_classifier.py
```

**Training Parameters:**
- Model: EfficientNet-B0
- Epochs: 30 (default)
- Batch Size: 32
- Optimizer: AdamW
- Learning Rate: 0.001

**Expected Training Time:**
- With RTX 4000 Ada: ~5-10 minutes for 30 epochs (depends on dataset size)
- CPU only: ~1-2 hours

### Step 3: Evaluate Model

```bash
python scripts/evaluate.py
```

This generates:
- Classification report
- Confusion matrix
- Top-K accuracy plots
- Error analysis

## 🔮 Inference

### Single Image Prediction

```bash
python scripts/inference.py
```

Edit the script to specify your test image path.

### Batch Prediction

```python
from scripts.inference import CattleBreedPredictor

predictor = CattleBreedPredictor()
predictor.predict_batch('test_images/', output_csv='results/predictions.csv')
```

## 🌐 Web Application

Launch the Streamlit demo:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload images for instant prediction
- Adjustable confidence threshold
- Three-view analysis visualization
- Top-3 breed predictions with confidence scores

## 📁 Project Structure

```
cattle_breed_mvp/
├── data/
│   ├── raw/                    # Raw images organized by breed
│   ├── processed/              # Processed train/val/test splits
│   │   ├── train/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── roi_images/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
├── models/
│   ├── detection/              # YOLO models
│   └── classification/         # Trained breed classifiers
│       └── breed_classifier_v1/
│           ├── best_model.pth
│           ├── final_model.pth
│           └── history.json
├── results/
│   ├── evaluation/             # Evaluation metrics and plots
│   ├── predictions/            # Prediction visualizations
│   └── classification/
├── scripts/
│   ├── prepare_data.py         # Data preparation
│   ├── extract_roi.py          # ROI extraction
│   ├── multi_view_analysis.py  # Three-view analysis
│   ├── dataset.py              # PyTorch dataset class
│   ├── train_classifier.py     # Training script
│   ├── inference.py            # Inference pipeline
│   └── evaluate.py             # Model evaluation
├── notebooks/                  # Jupyter notebooks for exploration
├── test_images/                # Test images
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎓 Supported Breeds

Default breeds (can be customized in `scripts/prepare_data.py`):
- Gir
- Sahiwal
- Red Sindhi
- Murrah Buffalo
- Mehsana Buffalo

## 🔧 Customization

### Add More Breeds

1. Add breed folders to `data/raw/`
2. Update `BREEDS` list in `scripts/prepare_data.py`
3. Re-run data preparation and training

### Use Three-View Analysis

In `scripts/train_classifier.py`, set:
```python
USE_THREE_VIEWS = True
```

This divides each image into left, front, and right regions for more detailed analysis.

### Change Model Architecture

In `scripts/train_classifier.py`, modify:
```python
classifier = BreedClassifier(
    num_classes=train_dataset.num_classes,
    model_name='resnet50',  # or 'efficientnet_b0'
    use_three_views=False
)
```

## 📊 Expected Performance

With proper training data (100+ images per breed):
- **Detection Accuracy**: 85-95% (using pre-trained YOLO)
- **Classification Accuracy**: 70-90% (depends on breed similarity and data quality)
- **Inference Speed**: 
  - GPU: ~50-100ms per image
  - CPU: ~500-1000ms per image

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in `train_classifier.py`:
```python
batch_size=16  # or 8
```

### No Cattle Detected
- Lower confidence threshold in inference
- Ensure images contain clear views of cattle
- Check if YOLO model downloaded correctly

### Low Accuracy
- Collect more training data (100+ images per breed minimum)
- Ensure data quality (clear, well-lit images)
- Train for more epochs
- Use data augmentation (already enabled)

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Usage Examples

### Example 1: Quick Test
```bash
# Activate environment
conda activate cattle_mvp

# Run inference on test image
python scripts/inference.py
```

### Example 2: Full Pipeline
```bash
# 1. Prepare data
python scripts/prepare_data.py

# 2. Extract ROIs
python scripts/extract_roi.py

# 3. Train model
python scripts/train_classifier.py

# 4. Evaluate
python scripts/evaluate.py

# 5. Run web app
streamlit run app.py
```

### Example 3: Custom Prediction
```python
from scripts.inference import CattleBreedPredictor

# Initialize predictor
predictor = CattleBreedPredictor()

# Predict single image
predictions = predictor.predict('my_cattle_image.jpg', visualize=True)

# Print results
for pred in predictions:
    top_breed = pred['breed_predictions'][0]
    print(f"Breed: {top_breed['breed']}, Confidence: {top_breed['score']:.2f}%")
```

## 🎯 MVP Demonstration Checklist

- [x] Data preparation pipeline
- [x] YOLO-based detection
- [x] ROI extraction
- [x] Multi-view analysis capability
- [x] Deep learning classification
- [x] Training pipeline
- [x] Evaluation metrics
- [x] Inference pipeline
- [x] Web interface
- [x] Visualization tools

## 📚 References

- YOLOv8: https://github.com/ultralytics/ultralytics
- EfficientNet: https://arxiv.org/abs/1905.11946
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/

## 🤝 Contributing

This is an MVP project. For improvements:
1. Add more breeds
2. Improve data augmentation
3. Implement ensemble methods
4. Add video processing capability
5. Deploy to cloud platforms

## 📄 License

This project is for educational and demonstration purposes.

## 👨‍💻 Author

Built for SIH (Smart India Hackathon) cattle breed recognition challenge.

## 🙏 Acknowledgments

- Pre-trained models from PyTorch and Ultralytics
- Public cattle datasets from Roboflow, Kaggle, and Bristol University
- Open-source community for tools and libraries

---

**Note**: This is an MVP (Minimum Viable Product) designed to demonstrate the feasibility of cattle breed detection. For production use, consider:
- Larger, more diverse datasets
- More extensive training
- Model optimization and quantization
- Robust error handling
- API development
- Cloud deployment
