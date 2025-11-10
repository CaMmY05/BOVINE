import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image
import json
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

class CattleDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        
        # Base transformations
        base_transforms = [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Training augmentations
        train_transforms = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ] + base_transforms
        
        # Validation/Test transformations
        val_transforms = base_transforms
        
        self.transform = transforms.Compose(train_transforms if mode == 'train' else val_transforms)
        
        # Get class names from subdirectories
        self.classes = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, patience=7, class_weights=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Initialize early stopping
    epochs_no_improve = 0
    
    # Initialize learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    # Add warmup
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    # Combined scheduler
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_epochs]
    )
    
    # Initialize lists to store metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    # Apply class weights if provided
                    if phase == 'train' and class_weights is not None:
                        weights = class_weights[labels].to(device)
                        loss = (criterion(outputs, labels, reduction='none') * weights).mean()
                    else:
                        loss = criterion(outputs, labels)
                    
                    # Label smoothing
                    if phase == 'train':
                        loss = (1 - 0.1) * loss + 0.1 * torch.nn.functional.cross_entropy(outputs, torch.ones_like(outputs) / outputs.size(1))
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
                # Step the scheduler on validation loss
                scheduler.step(epoch_acc)
            
            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_val_acc': best_acc.item()
    }

def evaluate_model(model, dataloader, class_names, output_dir):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save metrics
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'class_metrics': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics['class_metrics'][class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1_score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    return metrics

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset."""
    class_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Inverse of class frequency
    weights = 1.0 / class_counts
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    return weights

if __name__ == '__main__':
    import argparse
    import torch.nn.functional as F
    
    parser = argparse.ArgumentParser(description='Train ResNet-18 for cattle breed classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data augmentation and normalization    # Data transforms are now handled in the dataset class
    
    # Create datasets with appropriate transforms
    train_dataset = CattleDataset(os.path.join(args.data_dir, 'train'), mode='train')
    val_dataset = CattleDataset(os.path.join(args.data_dir, 'val'), mode='val')
    test_dataset = CattleDataset(os.path.join(args.data_dir, 'test'), mode='test')
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    }
    
    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # Save class names
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"Dataset sizes: {', '.join([f'{x}: {len(dataloaders[x].dataset)}' for x in ['train', 'val', 'test']])}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {', '.join(class_names)}")
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")
    
    # Initialize the model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze initial layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(train_dataset.classes))
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Separate parameters for different learning rates
    params = []
    # Add feature extractor parameters with base learning rate
    params.append({
        'params': [p for name, p in model.named_parameters() 
                  if 'fc' not in name and p.requires_grad],
        'lr': args.lr
    })
    # Add classifier parameters with higher learning rate
    params.append({
        'params': [p for name, p in model.named_parameters() 
                  if 'fc' in name and p.requires_grad],
        'lr': args.lr * 10
    })
    
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        steps_per_epoch=len(dataloaders['train']),
        epochs=args.num_epochs,
        pct_start=0.1
    )
    
    # Train the model with class weights
    print("Starting training...")
    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        num_epochs=args.num_epochs,
        patience=args.patience,
        class_weights=class_weights
    )
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model, 
        dataloaders['test'], 
        class_names,
        output_dir
    )
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'training_history': {
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses'],
                'train_accs': history['train_accs'],
                'val_accs': history['val_accs'],
                'best_epoch': history['best_epoch'],
                'best_val_acc': history['best_val_acc']
            },
            'class_names': class_names,
            'num_classes': num_classes,
            'model_architecture': 'resnet18'
        }, f, indent=2)
    
    print("\nTest metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-score: {test_metrics['f1_score']:.4f}")
    
    print("\nTraining complete!")
    print(f"Model and results saved to: {output_dir}")
