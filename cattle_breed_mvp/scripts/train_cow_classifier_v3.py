"""
Train V3 Cow Classifier with 5 breeds
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def train_cow_classifier_v3():
    """Train EfficientNet-B0 on 5 cow breeds"""
    
    # Configuration
    config = {
        "model_name": "efficientnet_b0",
        "num_classes": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "num_epochs": 50,
        "early_stopping_patience": 10,
        "lr_patience": 5,
        "label_smoothing": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("🐄 Training V3 Cow Classifier (5 breeds)...")
    print("="*60)
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Paths
    data_dir = Path("data/processed_v3/cows")
    model_dir = Path("models/classification/cow_classifier_v3")
    results_dir = Path("results/cow_evaluation_v3")
    
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load classes
    with open(data_dir / "classes.json", "r") as f:
        classes = json.load(f)
    print(f"🏷️ Classes: {classes}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = ImageFolder(data_dir / "val", transform=val_transform)
    test_dataset = ImageFolder(data_dir / "test", transform=val_transform)
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images") 
    print(f"  Test: {len(test_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    # Create model
    print("\n🏗️ Creating model...")
    model = timm.create_model(config["model_name"], pretrained=True, num_classes=config["num_classes"])
    model = model.to(config["device"])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=config["lr_patience"]
    )
    
    # Calculate class weights for imbalance
    print("\n⚖️ Calculating class weights...")
    class_counts = np.zeros(config["num_classes"])
    for _, label in train_dataset:
        class_counts[label] += 1
    
    class_weights = 1.0 / (class_counts / class_counts.sum())
    class_weights = class_weights / class_weights.sum() * config["num_classes"]
    class_weights = torch.FloatTensor(class_weights).to(config["device"])
    
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "learning_rate": []
    }
    
    # Early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print("\n🚀 Starting training...")
    print("="*60)
    
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(config["device"]), target.to(config["device"])
            
            optimizer.zero_grad()
            output = model(data)
            
            # Apply class weights to loss
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for batch_idx, (data, target) in enumerate(val_pbar):
                data, target = data.to(config["device"]), target.to(config["device"])
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                # Update progress bar
                current_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config,
                'classes': classes
            }, model_dir / "best_model.pth")
            print(f"  🎉 New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= config["early_stopping_patience"]:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs!")
            print(f"  Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        print("-" * 40)
    
    # Load best model for evaluation
    print("\n📊 Loading best model for evaluation...")
    checkpoint = torch.load(model_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(config["device"]), target.to(config["device"])
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
    
    # Generate classification report
    print("\n📋 Classification Report:")
    report = classification_report(all_targets, all_predictions, target_names=classes, output_dict=True)
    print(classification_report(all_targets, all_predictions, target_names=classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label='Train Acc')
    plt.plot(history["val_acc"], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history["learning_rate"])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        "test_accuracy": test_acc,
        "val_accuracy": best_val_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "training_history": history,
        "config": config,
        "classes": classes,
        "total_train_images": len(train_dataset),
        "total_val_images": len(val_dataset),
        "total_test_images": len(test_dataset)
    }
    
    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: {model_dir / 'best_model.pth'}")
    print(f"📊 Results saved to: {results_dir}")
    print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return results

if __name__ == "__main__":
    results = train_cow_classifier_v3()
    print("\n🚀 Next Steps:")
    print("1. Run: python scripts/evaluate_cow_model_v3.py")
    print("2. Update app.py to include V3 model option")
