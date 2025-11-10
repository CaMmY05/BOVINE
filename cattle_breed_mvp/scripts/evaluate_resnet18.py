import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
from itertools import cycle
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
        
        # Get class names from subdirectories
        self.classes = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]
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
            
        return img, label, img_path

def load_model(model_path, num_classes, device):
    """Load the trained ResNet-18 model."""
    import torchvision.models as models
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth'), 
                                   map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate the model and return predictions and metrics."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    return {
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'paths': np.array(all_paths),
        'class_names': class_names
    }

def plot_confusion_matrix(results, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(results['labels'], results['preds'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        xticklabels=results['class_names'],
        yticklabels=results['class_names']
    )
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Create interactive plot
    fig = px.imshow(
        cm_norm,
        labels=dict(x="Predicted", y="True", color="Normalized Count"),
        x=results['class_names'],
        y=results['class_names'],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
    )
    
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title='Normalized Confusion Matrix',
        xaxis_title='Predicted label',
        yaxis_title='True label',
        width=800,
        height=700
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_dir, 'confusion_matrix.html'))
    plt.close()

def plot_roc_curve(results, output_dir):
    """Plot ROC curve for each class."""
    n_classes = len(results['class_names'])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            (results['labels'] == i).astype(int), 
            results['probs'][:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='{0} (AUC = {1:0.2f})'
                ''.format(results['class_names'][i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    
    # Create interactive plot
    fig = go.Figure()
    
    for i, color in zip(range(n_classes), colors):
        fig.add_trace(go.Scatter(
            x=fpr[i],
            y=tpr[i],
            name=f"{results['class_names'][i]} (AUC = {roc_auc[i]:.2f})",
            mode='lines',
            line=dict(color=color, width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random (AUC = 0.50)',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title='ROC Curves for Each Class',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=900,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_dir, 'roc_curves.html'))
    plt.close()

def plot_precision_recall_curve(results, output_dir):
    """Plot precision-recall curve for each class."""
    n_classes = len(results['class_names'])
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    # Calculate precision and recall for each class
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            (results['labels'] == i).astype(int),
            results['probs'][:, i]
        )
        average_precision[i] = average_precision_score(
            (results['labels'] == i).astype(int),
            results['probs'][:, i]
        )
    
    # Plot all precision-recall curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label='{0} (AP = {1:0.2f})'
                ''.format(results['class_names'][i], average_precision[i]))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves for Each Class')
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), 
               dpi=300, bbox_inches='tight')
    
    # Create interactive plot
    fig = go.Figure()
    
    for i, color in zip(range(n_classes), colors):
        fig.add_trace(go.Scatter(
            x=recall[i],
            y=precision[i],
            name=f"{results['class_names'][i]} (AP = {average_precision[i]:.2f})",
            mode='lines',
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title='Precision-Recall Curves for Each Class',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=900,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_dir, 'precision_recall_curves.html'))
    plt.close()

def plot_class_distribution(results, output_dir):
    """Plot class distribution in the test set."""
    unique, counts = np.unique(results['labels'], return_counts=True)
    class_counts = dict(zip([results['class_names'][i] for i in unique], counts))
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), 
               dpi=300, bbox_inches='tight')
    
    # Create interactive plot
    fig = px.bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        labels={'x': 'Class', 'y': 'Number of Samples'},
        title='Class Distribution in Test Set',
        text=list(class_counts.values())
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        width=900,
        height=600,
        margin=dict(l=50, r=50, t=80, b=150)
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_dir, 'class_distribution.html'))
    plt.close()

def generate_classification_report(results, output_dir):
    """Generate and save classification report."""
    report = classification_report(
        results['labels'],
        results['preds'],
        target_names=results['class_names'],
        output_dict=True
    )
    
    # Save as JSON
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Convert to DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()
    
    # Save as CSV
    df_report.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Create HTML table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[''] + list(df_report.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df_report.index] + [df_report[col] for col in df_report.columns],
            fill_color='lavender',
            align='left')
    )])
    
    fig.update_layout(
        title='Classification Report',
        width=900,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.write_html(os.path.join(output_dir, 'classification_report.html'))
    
    return report

def create_dashboard(results, output_dir):
    """Create an interactive dashboard with all visualizations."""
    # Create a dashboard with all plots
    fig = make_subplots(
        rows=3, 
        cols=2,
        subplot_titles=(
            'Confusion Matrix',
            'ROC Curves',
            'Precision-Recall Curves',
            'Class Distribution',
            'Classification Report'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "table", "colspan": 2}, None],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add confusion matrix
    cm = confusion_matrix(results['labels'], results['preds'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig.add_trace(
        go.Heatmap(
            z=cm_norm,
            x=results['class_names'],
            y=results['class_names'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Normalized Count')
        ),
        row=1, col=1
    )
    
    # Add ROC curves
    n_classes = len(results['class_names'])
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(
            (results['labels'] == i).astype(int), 
            results['probs'][:, i]
        )
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{results['class_names'][i]} (AUC = {roc_auc:.2f})",
                mode='lines',
                line=dict(color=color, width=2)
            ),
            row=1, col=2
        )
    
    # Add diagonal line for ROC
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random (AUC = 0.50)',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add precision-recall curves
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(
            (results['labels'] == i).astype(int),
            results['probs'][:, i]
        )
        ap = average_precision_score(
            (results['labels'] == i).astype(int),
            results['probs'][:, i]
        )
        
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                name=f"{results['class_names'][i]} (AP = {ap:.2f})",
                mode='lines',
                line=dict(color=color, width=2)
            ),
            row=2, col=1
        )
    
    # Add class distribution
    unique, counts = np.unique(results['labels'], return_counts=True)
    class_counts = [counts[np.where(unique == i)[0][0]] if i in unique else 0 
                   for i in range(len(results['class_names']))]
    
    fig.add_trace(
        go.Bar(
            x=results['class_names'],
            y=class_counts,
            name='Class Distribution',
            marker_color='#1f77b4',
            text=class_counts,
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Add classification report as table
    report = classification_report(
        results['labels'],
        results['preds'],
        target_names=results['class_names'],
        output_dict=True
    )
    
    df_report = pd.DataFrame(report).transpose()
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=[''] + list(df_report.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[df_report.index] + [df_report[col] for col in df_report.columns],
                fill_color='lavender',
                align='left'
            )
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"ResNet-18 Model Evaluation Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        showlegend=True,
        height=1500,
        width=1400,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='Predicted', row=1, col=1)
    fig.update_yaxes(title_text='True', row=1, col=1)
    fig.update_xaxes(title_text='False Positive Rate', row=1, col=2)
    fig.update_yaxes(title_text='True Positive Rate', row=1, col=2)
    fig.update_xaxes(title_text='Recall', row=2, col=1)
    fig.update_yaxes(title_text='Precision', row=2, col=1)
    fig.update_xaxes(title_text='Class', row=2, col=2)
    fig.update_yaxes(title_text='Number of Samples', row=2, col=2)
    
    # Save dashboard
    fig.write_html(os.path.join(output_dir, 'evaluation_dashboard.html'))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: model_dir/evaluation)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'evaluation')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load class names
    with open(os.path.join(args.model_dir, 'classes.json'), 'r') as f:
        class_names = json.load(f)
    
    # Create test dataset and dataloader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(args.data_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f'Test dataset size: {len(test_dataset)}')
    print(f'Number of classes: {len(class_names)}')
    print(f'Class names: {class_names}')
    
    # Load model
    print('Loading model...')
    model = load_model(args.model_dir, len(class_names), device)
    
    # Evaluate model
    print('Evaluating model...')
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Generate visualizations
    print('Generating visualizations...')
    plot_confusion_matrix(results, args.output_dir)
    plot_roc_curve(results, args.output_dir)
    plot_precision_recall_curve(results, args.output_dir)
    plot_class_distribution(results, args.output_dir)
    report = generate_classification_report(results, args.output_dir)
    
    # Create dashboard
    print('Creating dashboard...')
    create_dashboard(results, args.output_dir)
    
    print(f'\nEvaluation complete! Results saved to: {args.output_dir}')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(
        results['labels'],
        results['preds'],
        target_names=class_names
    ))

if __name__ == '__main__':
    main()
