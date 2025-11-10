import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import json
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CattleBreedDataset

class BreedClassifier:
    def __init__(self, num_classes, model_name='efficientnet_b0', use_three_views=False, class_weights=None):
        self.num_classes = num_classes
        self.model_name = model_name
        self.use_three_views = use_three_views
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Build model
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Loss and optimizer with improvements
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"Using class weights: {class_weights}")
        
        # Add label smoothing to reduce overconfidence
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
    def build_model(self):
        """
        Build classification model
        """
        if self.model_name == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            if self.use_three_views:
                # Modify first conv layer to accept 9 channels (3 views x 3 RGB)
                original_conv = model.features[0][0]
                model.features[0][0] = nn.Conv2d(
                    9, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
                
                # Initialize new weights
                with torch.no_grad():
                    model.features[0][0].weight[:, :3] = original_conv.weight
                    model.features[0][0].weight[:, 3:6] = original_conv.weight
                    model.features[0][0].weight[:, 6:9] = original_conv.weight
            
            # Replace classifier
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            
        elif self.model_name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        return model
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Reshape if using three views
            if self.use_three_views:
                batch_size = images.size(0)
                images = images.view(batch_size, -1, images.size(3), images.size(4))
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_three_views:
                    batch_size = images.size(0)
                    images = images.view(batch_size, -1, images.size(3), images.size(4))
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=50, save_dir='models/classification'):
        os.makedirs(save_dir, exist_ok=True)
        best_acc = 0.0
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': history
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        
        # Save history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
        return history


def main():
    # Data transforms - IMPROVED with more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),  # Increased from 15
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # More variation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    USE_THREE_VIEWS = False  # Set to True to use three-view approach
    
    train_dataset = CattleBreedDataset(
        'data/processed', 
        split='train', 
        transform=train_transform,
        use_three_views=USE_THREE_VIEWS
    )
    
    val_dataset = CattleBreedDataset(
        'data/processed',
        split='val',
        transform=val_transform,
        use_three_views=USE_THREE_VIEWS
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No data found! Please run prepare_data.py first.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate class weights to handle imbalance
    # Count samples per class
    class_counts = {}
    for _, label in train_dataset:
        label_idx = label.item() if torch.is_tensor(label) else label
        class_counts[label_idx] = class_counts.get(label_idx, 0) + 1
    
    # Calculate weights (inverse frequency)
    total_samples = len(train_dataset)
    class_weights = torch.zeros(train_dataset.num_classes)
    for class_idx in range(train_dataset.num_classes):
        count = class_counts.get(class_idx, 1)
        class_weights[class_idx] = total_samples / (train_dataset.num_classes * count)
    
    print(f"\nClass distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Initialize classifier with class weights
    classifier = BreedClassifier(
        num_classes=train_dataset.num_classes,
        model_name='efficientnet_b0',
        use_three_views=USE_THREE_VIEWS,
        class_weights=class_weights
    )
    
    # Train
    history = classifier.train(
        train_loader,
        val_loader,
        epochs=30,
        save_dir='models/classification/breed_classifier_v1'
    )

if __name__ == "__main__":
    main()
