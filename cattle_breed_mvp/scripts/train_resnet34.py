import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """Create and return data loaders for training and validation."""
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Just normalization for validation
    val_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets - handle both 'valid' and 'val' directory names
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    
    # Fallback to 'val' if 'valid' doesn't exist
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes

def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalanced datasets."""
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1
    
    # Add small epsilon to avoid division by zero
    class_weights = 1. / (torch.tensor(class_counts, dtype=torch.float) + 1e-6)
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    return class_weights

def train_epoch(model, loader, criterion, optimizer, device, scaler, max_grad_norm=1.0):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    return val_loss, val_acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save model checkpoint."""
    torch.save(state, filename)
    if is_best:
        best_model_path = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        torch.save(state, best_model_path)

def main():
    # Configuration
    import argparse
    parser = argparse.ArgumentParser(description='Train ResNet-34 for cattle breed classification')
    parser.add_argument('--data-dir', type=str, required=True, 
                        help='Path to the data directory (should contain train/val folders)')
    parser.add_argument('--model-type', type=str, choices=['cow', 'buffalo'], required=True,
                        help='Type of model to train (cow or buffalo)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    model_dir = f"models/classification/resnet34_{args.model_type}_v1"
    os.makedirs(model_dir, exist_ok=True)
    
    # Data loading
    print("Loading data...")
    train_loader, val_loader, classes = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Save class names
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    with open(os.path.join(model_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f)
    
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Initialize model
    print("Initializing model...")
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(classes))
    )
    model = model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different learning rates for feature extractor and classifier
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n and 'layer4' not in n], 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n], 'lr': args.lr},
        {'params': model.fc.parameters(), 'lr': args.lr * 10}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[group['lr'] for group in param_groups],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch}) with best accuracy {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print('-' * 10)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, 
            optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet34',
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'class_to_idx': class_to_idx,
        }, is_best, filename=os.path.join(model_dir, 'checkpoints', 'checkpoint.pth.tar'))
        
        # Early stopping if no improvement for 15 epochs
        if (epoch - start_epoch) > 15 and val_acc < best_acc * 0.95:
            print("Early stopping as validation accuracy hasn't improved for 15 epochs")
            break
    
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    # Save the final model
    torch.save({
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': 'resnet34'
    }, os.path.join(model_dir, 'final_model.pth'))
    print(f"Model saved to {os.path.join(model_dir, 'final_model.pth')}")

if __name__ == '__main__':
    main()
