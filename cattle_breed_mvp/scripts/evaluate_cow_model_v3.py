"""
Evaluate Cow Breed Classifier V3
Complete evaluation with metrics and visualizations for 5 breeds
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm

def load_model(model_path, num_classes=5, device='cuda'):
    """Load trained model"""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    model = model.to(device)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(labels, class_names, save_path):
    """Plot class distribution"""
    counts = [sum(labels == i) for i in range(len(class_names))]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=counts)
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    config = {
        "batch_size": 32,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("🐄 Evaluating V3 Cow Classifier (5 breeds)")
    print("="*60)
    
    # Paths
    data_dir = Path("data/processed_v3/cows")
    model_path = Path("models/classification/cow_classifier_v3/best_model.pth")
    results_dir = Path("results/cow_evaluation_v3")
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    with open(data_dir / "classes.json", "r") as f:
        class_names = json.load(f)
    
    print(f"🏷️ Classes: {class_names}")
    
    # Data transforms (should match training)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print("\n📂 Loading test dataset...")
    test_dataset = ImageFolder(data_dir / "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config["batch_size"],
                           shuffle=False, 
                           num_workers=config["num_workers"])
    
    print(f"  Found {len(test_dataset)} test images")
    
    # Load model
    print("\n🏗️ Loading trained model...")
    model = load_model(model_path, num_classes=len(class_names), device=config["device"])
    
    # Evaluate
    print("\n🧪 Running evaluation...")
    preds, labels, probs = evaluate_model(model, test_loader, config["device"], class_names)
    
    # Calculate metrics
    accuracy = (preds == labels).mean() * 100
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    
    print("\n📊 Evaluation Results:")
    print("="*60)
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print("\n📋 Classification Report:")
    print(report)
    
    # Save results
    with open(results_dir / "evaluation_report.txt", "w") as f:
        f.write("V3 Cow Classifier Evaluation\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    print("📊 Generating visualizations...")
    plot_confusion_matrix(cm, class_names, results_dir / "confusion_matrix.png")
    plot_class_distribution(labels, class_names, results_dir / "class_distribution.png")
    
    # Save raw predictions for further analysis
    np.savez(results_dir / "predictions.npz",
             predictions=preds,
             true_labels=labels,
             probabilities=probs,
             class_names=class_names)
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {results_dir}")
    print("\n🔍 You can find:")
    print(f"- Evaluation report: {results_dir}/evaluation_report.txt")
    print(f"- Confusion matrix: {results_dir}/confusion_matrix.png")
    print(f"- Class distribution: {results_dir}/class_distribution.png")
    print(f"- Raw predictions: {results_dir}/predictions.npz")

if __name__ == "__main__":
    main()
