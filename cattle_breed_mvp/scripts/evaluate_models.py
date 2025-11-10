import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    roc_auc_score
)
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class ModelEvaluator:
    def __init__(self, model_path, data_dir, model_type='resnet34'):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model directory
            data_dir: Path to the data directory (should contain test/ folder)
            model_type: Type of model architecture ('resnet34')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.data_dir = data_dir
        self.model_type = model_type
        
        # Load class to index mapping
        with open(os.path.join(model_path, 'class_to_idx.json'), 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        
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
        self.test_dataset = ImageFolder(
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
        """Load the trained model."""
        if self.model_type == 'resnet34':
            model = models.resnet34(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, len(self.classes))
            )
            
            # Load the entire model state dict directly
            model_path = os.path.join(self.model_path, 'final_model.pth')
            if os.path.exists(model_path):
                model_state = torch.load(model_path, map_location=self.device)
                if isinstance(model_state, dict):
                    if 'state_dict' in model_state:
                        # Handle case where model is saved as a checkpoint
                        model.load_state_dict(model_state['state_dict'])
                    else:
                        # Handle case where model is saved directly
                        model.load_state_dict(model_state)
                else:
                    # Handle case where model is saved directly (not in a dict)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model = model.to(self.device)
        model.eval()
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
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'accuracy': (all_preds == all_labels).mean(),
            'classification_report': classification_report(
                all_labels, all_preds, 
                target_names=self.classes,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'roc_auc': {}
        }
        
        # Calculate ROC AUC for each class
        for i, class_name in enumerate(self.classes):
            if len(np.unique(all_labels == i)) > 1:  # Check if class exists in test set
                metrics['roc_auc'][class_name] = roc_auc_score(
                    (all_labels == i).astype(int),
                    all_probs[:, i]
                )
        
        return metrics, all_probs, all_labels
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix."""
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=self.classes,
            y=self.classes,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=800,
            height=700
        )
        
        if save_path:
            fig.write_html(os.path.join(save_path, 'confusion_matrix.html'))
        
        return fig
    
    def plot_roc_curves(self, all_probs, all_labels, save_path=None):
        """Plot ROC curves for each class."""
        fig = go.Figure()
        
        for i, class_name in enumerate(self.classes):
            if len(np.unique(all_labels == i)) > 1:  # Check if class exists in test set
                fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f'{class_name} (AUC = {roc_auc:.2f})',
                    mode='lines'
                ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(dash='dash'),
            name='Random (AUC = 0.5)'
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(os.path.join(save_path, 'roc_curves.html'))
        
        return fig
    
    def create_dashboard(self, metrics, all_probs, all_labels, save_dir):
        """Create an interactive dashboard with evaluation metrics."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create confusion matrix
        cm_fig = self.plot_confusion_matrix(metrics['confusion_matrix'], save_dir)
        
        # Create ROC curves
        roc_fig = self.plot_roc_curves(all_probs, all_labels, save_dir)
        
        # Create classification report table
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_fig = go.Figure(data=[go.Table(
            header=dict(values=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[report_df.index, 
                             report_df['precision'].round(3), 
                             report_df['recall'].round(3),
                             report_df['f1-score'].round(3),
                             report_df['support'].astype(int)],
                     fill_color='lavender',
                     align='left'))
        ])
        
        report_fig.update_layout(
            title='Classification Report',
            width=800,
            height=400
        )
        
        # Save report figure
        report_fig.write_html(os.path.join(save_dir, 'classification_report.html'))
        
        # Create dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric-card {{ 
                    background: #f8f9fa; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .metric-value {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #3498db;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Evaluation Dashboard</h1>
                <p><strong>Model Path:</strong> {os.path.basename(self.model_path)}</p>
                <p><strong>Test Set:</strong> {os.path.basename(self.data_dir)}</p>
                
                <div class="metric-card">
                    <h2>Overall Accuracy</h2>
                    <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
                </div>
                
                <div class="metric-card">
                    <h2>Confusion Matrix</h2>
                    <div id="confusion-matrix"></div>
                </div>
                
                <div class="metric-card">
                    <h2>ROC Curves</h2>
                    <div id="roc-curves"></div>
                </div>
                
                <div class="metric-card">
                    <h2>Classification Report</h2>
                    <div id="classification-report"></div>
                </div>
            </div>
            
            <script>
                // Load and display the plots
                Promise.all([
                    fetch('confusion_matrix.html').then(r => r.text()),
                    fetch('roc_curves.html').then(r => r.text()),
                    fetch('classification_report.html').then(r => r.text())
                ]).then(([cm_html, roc_html, report_html]) => {{
                    // Extract div content from the HTML
                    const cm_div = document.createElement('div');
                    cm_div.innerHTML = cm_html;
                    document.getElementById('confusion-matrix').innerHTML = 
                        cm_div.querySelector('#\{plotly-html\}').innerHTML;
                    
                    const roc_div = document.createElement('div');
                    roc_div.innerHTML = roc_html;
                    document.getElementById('roc-curves').innerHTML = 
                        roc_div.querySelector('#\{plotly-html\}').innerHTML;
                    
                    const report_div = document.createElement('div');
                    report_div.innerHTML = report_html;
                    document.getElementById('classification-report').innerHTML = 
                        report_div.querySelector('#\{plotly-html\}').innerHTML;
                    
                    // Re-initialize Plotly
                    if (window.Plotly) {{
                        window.Plotly.newPlot(
                            document.getElementById('confusion-matrix'),
                            JSON.parse(JSON.parse(cm_div.querySelector('script[type="application/json"]').innerHTML)),
                            {{}},
                            {{responsive: true}}
                        );
                        
                        window.Plotly.newPlot(
                            document.getElementById('roc-curves'),
                            JSON.parse(JSON.parse(roc_div.querySelector('script[type="application/json"]').innerHTML)),
                            {{}},
                            {{responsive: true}}
                        );
                        
                        window.Plotly.newPlot(
                            document.getElementById('classification-report'),
                            JSON.parse(JSON.parse(report_div.querySelector('script[type="application/json"]').innerHTML)),
                            {{}},
                            {{responsive: true}}
                        );
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        with open(os.path.join(save_dir, 'dashboard.html'), 'w') as f:
            f.write(dashboard_html)
        
        print(f"Dashboard saved to {os.path.join(save_dir, 'dashboard.html')}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the trained model directory')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the data directory (should contain test/ folder)')
    parser.add_argument('--model-type', type=str, default='resnet34',
                        choices=['resnet34'],
                        help='Model architecture')
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model_type} model from {args.model_dir}")
    print(f"Using test data from {args.data_dir}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_dir,
        data_dir=args.data_dir,
        model_type=args.model_type
    )
    
    # Run evaluation
    metrics, all_probs, all_labels = evaluator.evaluate()
    
    # Create output directory
    output_dir = os.path.join(args.model_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'accuracy': metrics['accuracy'],
            'classification_report': metrics['classification_report'],
            'roc_auc': metrics['roc_auc']
        }, f, indent=2)
    
    # Create and save dashboard
    evaluator.create_dashboard(metrics, all_probs, all_labels, output_dir)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        np.argmax(all_probs, axis=1),
        target_names=evaluator.classes
    ))

if __name__ == '__main__':
    main()
