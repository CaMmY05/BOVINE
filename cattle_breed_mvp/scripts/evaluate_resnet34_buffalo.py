"""
Evaluate ResNet-34 Buffalo Breed Classifier
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_model(model_path, num_classes, device):
    """Load ResNet-34 model with custom classifier"""
    model = models.resnet34(pretrained=False)
    num_ftrs = model.fc.in_features
    # Match the model's classifier structure
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        # Handle case where model was saved with DataParallel
        state_dict = checkpoint['state_dict']
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and return metrics"""
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, "results/resnet34_buffalo_confusion.png")
    
    # Calculate per-class accuracy
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    print("\nPer-Class Accuracy:")
    for i in range(len(class_names)):
        print(f"  {class_names[i]:<15}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_total[i]} images)")

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def main():
    # Configuration
    model_path = "models/classification/resnet34_buffalo_v1/final_model.pth"
    data_dir = "data/processed_v2/buffaloes/test"
    batch_size = 32
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class mapping from the model directory
    with open("models/classification/resnet34_buffalo_v1/class_to_idx.json", 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create a custom dataset that respects the model's class mapping
    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.samples = []
            
            # Map folder names to class indices
            for class_name in class_names:
                class_dir = self.root_dir / class_name
                if not class_dir.exists():
                    print(f"Warning: Class directory not found: {class_dir}")
                    continue
                    
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), class_to_idx[class_name]))
        
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    # Load dataset
    dataset = CustomDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Test dataset: {len(dataset)} images, {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path, len(class_names), device)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluate_model(model, dataloader, device, class_names)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
