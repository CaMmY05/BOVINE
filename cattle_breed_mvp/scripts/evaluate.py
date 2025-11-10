import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CattleBreedDataset
from train_classifier import BreedClassifier

class ModelEvaluator:
    def __init__(self, model_path, classes_path, data_root):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classes
        with open(classes_path, 'r') as f:
            self.class_to_idx = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Load model
        self.classifier = BreedClassifier(
            num_classes=self.num_classes,
            model_name='efficientnet_b0'
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.classifier.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.classifier.model.load_state_dict(checkpoint)
        
        self.classifier.model.eval()
        
        # Load test dataset
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_dataset = CattleBreedDataset(
            data_root,
            split='test',
            transform=test_transform
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
    
    def evaluate(self):
        """
        Evaluate model on test set
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        print("Evaluating model on test set...")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                
                outputs = self.classifier.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_predictions, all_labels, all_probs
    
    def generate_report(self, predictions, labels, output_dir='results/evaluation'):
        """
        Generate classification report and confusion matrix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Classification report
        class_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        
        report = classification_report(
            labels,
            predictions,
            target_names=class_names,
            digits=3
        )
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(report)
        
        # Save report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Calculate accuracy
        accuracy = 100.0 * (predictions == labels).sum() / len(labels)
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {output_dir}/confusion_matrix.png")
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        print("\nPer-Class Accuracy:")
        for i, acc in enumerate(per_class_acc):
            print(f"  {class_names[i]:20s}: {acc*100:.2f}%")
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': dict(zip(class_names, per_class_acc * 100)),
            'confusion_matrix': cm.tolist()
        }
    
    def plot_top_k_accuracy(self, predictions, labels, probs, k_values=[1, 3, 5], output_dir='results/evaluation'):
        """
        Calculate and plot top-k accuracy
        """
        results = {}
        
        for k in k_values:
            # Get top-k predictions
            top_k_preds = np.argsort(probs, axis=1)[:, -k:]
            
            # Check if true label is in top-k
            correct = sum([labels[i] in top_k_preds[i] for i in range(len(labels))])
            accuracy = 100.0 * correct / len(labels)
            
            results[f'top_{k}'] = accuracy
            print(f"Top-{k} Accuracy: {accuracy:.2f}%")
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(k_values)), [results[f'top_{k}'] for k in k_values])
        plt.xlabel('k')
        plt.ylabel('Accuracy (%)')
        plt.title('Top-K Accuracy')
        plt.xticks(range(len(k_values)), [f'Top-{k}' for k in k_values])
        plt.ylim([0, 100])
        
        for i, k in enumerate(k_values):
            plt.text(i, results[f'top_{k}'] + 1, f"{results[f'top_{k}']:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_k_accuracy.png'), dpi=300, bbox_inches='tight')
        
        return results
    
    def analyze_errors(self, predictions, labels, probs, output_dir='results/evaluation', top_n=10):
        """
        Analyze worst predictions
        """
        # Calculate confidence for each prediction
        pred_confidences = probs[np.arange(len(predictions)), predictions]
        
        # Find incorrect predictions
        incorrect_mask = predictions != labels
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            print("\n🎉 Perfect predictions! No errors found.")
            return
        
        # Get confidence for incorrect predictions
        incorrect_confidences = pred_confidences[incorrect_indices]
        
        # Sort by confidence (highest confidence errors are most interesting)
        sorted_indices = np.argsort(incorrect_confidences)[::-1][:top_n]
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} CONFIDENT MISCLASSIFICATIONS")
        print(f"{'='*60}")
        
        errors = []
        for rank, idx in enumerate(sorted_indices, 1):
            orig_idx = incorrect_indices[idx]
            true_label = self.idx_to_class[labels[orig_idx]]
            pred_label = self.idx_to_class[predictions[orig_idx]]
            confidence = incorrect_confidences[idx] * 100
            
            print(f"\n{rank}. Image index: {orig_idx}")
            print(f"   True: {true_label:20s} | Predicted: {pred_label:20s} | Confidence: {confidence:.2f}%")
            
            errors.append({
                'image_index': int(orig_idx),
                'true_breed': true_label,
                'predicted_breed': pred_label,
                'confidence': float(confidence)
            })
        
        # Save errors
        with open(os.path.join(output_dir, 'top_errors.json'), 'w') as f:
            json.dump(errors, f, indent=2)
        
        return errors


def main():
    evaluator = ModelEvaluator(
        model_path='models/classification/breed_classifier_v1/best_model.pth',
        classes_path='data/processed/classes.json',
        data_root='data/processed'
    )
    
    # Run evaluation
    predictions, labels, probs = evaluator.evaluate()
    
    # Generate reports
    metrics = evaluator.generate_report(predictions, labels)
    
    # Top-k accuracy
    topk_results = evaluator.plot_top_k_accuracy(predictions, labels, probs, k_values=[1, 3, 5])
    
    # Error analysis
    errors = evaluator.analyze_errors(predictions, labels, probs, top_n=10)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("Results saved to: results/evaluation/")

if __name__ == "__main__":
    main()
