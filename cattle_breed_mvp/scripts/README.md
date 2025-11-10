# Cattle Breed Classification - Model Management

This directory contains scripts for managing the cattle breed classification models, including training, evaluation, and tracking.

## Scripts

### `model_registry.py`

A central registry of all models with their configurations, dataset paths, and evaluation results.

### `run_model_pipeline.py`

A unified interface for training and evaluating models.

## Usage

### List Available Models

```bash
python scripts/model_registry.py
```

### Train a Model

```bash
# Train a specific model
python scripts/run_model_pipeline.py --model efficientnet_b0_v1 --train

# Train all models
python scripts/run_model_pipeline.py --train
```

### Evaluate a Model

```bash
# Evaluate a specific model
python scripts/run_model_pipeline.py --model efficientnet_b0_v1 --evaluate

# Evaluate all models
python scripts/run_model_pipeline.py --evaluate
```

### Generate a Report

```bash
# Generate a summary report of all models and their evaluations
python scripts/run_model_pipeline.py --report
```

## Model Registry

The model registry (`model_registry.py`) contains information about all models, including:

- Model architecture and version
- Supported animal types (cows, buffaloes, or both)
- Path to the dataset used for training
- Path where the model is stored
- Training script and evaluation script
- Hyperparameters (batch size, learning rate, etc.)

## Results

Evaluation results are stored in the `results/` directory with the following structure:

```
results/
├── efficientnet_b0_v1_evaluation.json
├── efficientnet_b0_v2_evaluation.json
├── efficientnet_b0_v3_evaluation.json
├── resnet18_v1_evaluation.json
├── resnet32_v1_evaluation.json
└── model_report.json
```

## Adding a New Model

To add a new model to the registry:

1. Add a new entry to the `MODEL_REGISTRY` dictionary in `model_registry.py`.
2. Create a training script (if needed) in the `scripts/` directory.
3. Create an evaluation script (if needed) in the `scripts/` directory.
4. Update the documentation in this README if necessary.

## Model Locations

- **EfficientNet-B0 V1 (Cows)**: `models/classification/cow_classifier_v2/`
- **EfficientNet-B0 V2 (Cows & Buffaloes)**: `models/classification/cow_classifier_v2/` and `models/classification/buffalo_classifier_v1/`
- **EfficientNet-B0 V3 (Cows)**: `models/classification/cow_classifier_v3/`
- **ResNet18 (Cows & Buffaloes)**: `models/classification/resnet18_cow_v1/` and `models/classification/resnet18_buffalo_v1/`
- **ResNet32 (Cows & Buffaloes)**: `models/classification/resnet34_cow_v1/` and `models/classification/resnet34_buffalo_v1/`

## Dataset Locations

- **Cows (v2)**: `data/processed_v2/cows/`
- **Buffaloes**: `data/processed_v2/buffaloes/`
- **Cows (v3)**: `data/processed_v3/cows/`

## Notes

- The ResNet32 model is actually using ResNet34 architecture due to compatibility.
- Some models share the same directory because they are different versions trained on the same architecture.
- The evaluation results include metrics like accuracy, precision, recall, and F1-score.
