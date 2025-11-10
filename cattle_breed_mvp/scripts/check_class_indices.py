import os
import json
from torchvision.datasets import ImageFolder

# Path to the processed data
data_dir = 'data/processed_v3/cows'

# Load the dataset
train_dataset = ImageFolder(os.path.join(data_dir, 'train'))

# Print class to index mapping
print("Class to index mapping:", train_dataset.class_to_idx)
print("Number of classes:", len(train_dataset.classes))

# Print first few samples to check labels
print("\nFirst 10 samples:")
for i, (_, label) in enumerate(train_dataset):
    if i >= 10:
        break
    print(f"Sample {i+1}: Label = {label}, Class = {train_dataset.classes[label]}")

# Check for any labels >= 5
print("\nChecking for invalid labels (>=5):")
invalid_labels = [i for i, (_, label) in enumerate(train_dataset) if label >= 5]
print(f"Found {len(invalid_labels)} samples with invalid labels (indices: {invalid_labels[:10]}{'...' if len(invalid_labels) > 10 else ''})")

# Print class distribution
print("\nClass distribution:")
class_counts = {}
for _, label in train_dataset:
    class_name = train_dataset.classes[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

for class_name, count in class_counts.items():
    print(f"{class_name}: {count} samples")
