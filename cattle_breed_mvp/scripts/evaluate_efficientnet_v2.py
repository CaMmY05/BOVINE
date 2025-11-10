"""
Evaluate EfficientNet-B0 v2 model for cow breed classification
"""

import os
import json
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import timm

class EfficientNetV2Evaluator:
    def __init__(self, model_path, data_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Class mapping
        self.classes = ['gir', 'red_sindhi', 'sahiwal']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Initialize model
        self.model = self._load_model()
        self.model.eval()
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load test data
        self.test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'test'),
            transform=self.transform
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def _load_model(self):
        """Load the trained EfficientNet-B0 model."""
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(self.classes))
        
        # Load the model weights
        model_path = os.path.join(self.model_path, 'best_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_path, 'final_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found in {self.model_path}")
        
        # Load the state dict
        model_state = torch.load(model_path, map_location=self.device)
        
        # Handle different model state formats
        if isinstance(model_state, dict):
            if 'model_state_dict' in model_state:
                state_dict = model_state['model_state_dict']
            elif 'state_dict' in model_state:
                state_dict = model_state['state_dict']
            else:
                state_dict = model_state
        else:
            state_dict = model_state
        
        # Load the state dict with strict=False to handle missing keys
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        return model
    
    def evaluate(self):
        """Run evaluation and return metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.classes,
            output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'report': report,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, save_dir):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, 
                    yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate EfficientNet-B0 v2 model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the trained model directory')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the data directory (should contain test/ folder)')
    
    args = parser.parse_args()
    
    print(f"Evaluating EfficientNet-B0 v2 model from {args.model_dir}")
    print(f"Using test data from {args.data_dir}")
    
    # Initialize evaluator
    evaluator = EfficientNetV2Evaluator(
        model_path=args.model_dir,
        data_dir=args.data_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Create output directory
    output_dir = os.path.join(args.model_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'classification_report': results['report']
        }, f, indent=2)
    
    # Plot and save confusion matrix
    evaluator.plot_confusion_matrix(results['confusion_matrix'], output_dir)
    
    # Print results
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        results['labels'], 
        results['predictions'],
        target_names=evaluator.classes
    ))

if __name__ == '__main__':
    main()
