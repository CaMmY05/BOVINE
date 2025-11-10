import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

def load_model_metrics(model_dirs):
    """Load metrics from model directories."""
    models = []
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        metrics_file = os.path.join(model_dir, 'metrics.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # Extract relevant metrics
            model_info = {
                'name': model_name,
                'type': 'cow' if 'cow' in model_name.lower() else 'buffalo',
                'architecture': metrics.get('model_architecture', 'unknown'),
                'num_classes': metrics.get('num_classes', 0),
                'test_accuracy': metrics.get('test_metrics', {}).get('accuracy', 0),
                'test_precision': metrics.get('test_metrics', {}).get('precision', 0),
                'test_recall': metrics.get('test_metrics', {}).get('recall', 0),
                'test_f1': metrics.get('test_metrics', {}).get('f1_score', 0),
                'best_val_accuracy': metrics.get('training_history', {}).get('best_val_acc', 0),
                'best_epoch': metrics.get('training_history', {}).get('best_epoch', 0),
                'class_metrics': metrics.get('test_metrics', {}).get('class_metrics', {})
            }
            
            models.append(model_info)
    
    return pd.DataFrame(models)

def create_comparison_plot(df, metric, title):
    """Create a bar plot comparing models by a specific metric."""
    fig = px.bar(
        df, 
        x='name', 
        y=metric,
        color='type',
        title=title,
        text_auto='.3f',
        labels={'name': 'Model', metric: metric.replace('_', ' ').title()}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat=".2%",
        height=500,
        legend_title="Animal Type"
    )
    
    return fig

def create_training_history_plot(model_dir):
    """Create training/validation accuracy and loss plots."""
    metrics_file = os.path.join(model_dir, 'metrics.json')
    
    if not os.path.exists(metrics_file):
        return None
        
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        
    history = metrics.get('training_history', {})
    
    if not history:
        return None
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))
    
    # Add accuracy traces
    fig.add_trace(
        go.Scatter(
            y=history.get('train_accs', []),
            name='Train Accuracy',
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=history.get('val_accs', []),
            name='Validation Accuracy',
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Add loss traces
    fig.add_trace(
        go.Scatter(
            y=history.get('train_losses', []),
            name='Train Loss',
            mode='lines+markers',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            y=history.get('val_losses', []),
            name='Validation Loss',
            mode='lines+markers',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Training History - {os.path.basename(model_dir)}",
        height=400,
        width=1200,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        xaxis2_title="Epoch",
        yaxis2_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_class_metrics_plot(model_dir):
    """Create a plot showing per-class metrics."""
    metrics_file = os.path.join(model_dir, 'metrics.json')
    
    if not os.path.exists(metrics_file):
        return None
        
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        
    class_metrics = metrics.get('test_metrics', {}).get('class_metrics', {})
    
    if not class_metrics:
        return None
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(class_metrics, orient='index')
    df = df.reset_index().rename(columns={'index': 'class'})
    
    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=('Precision', 'Recall', 'F1-Score')
    )
    
    # Add traces for each metric
    metrics = ['precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Bar(
                x=df['class'],
                y=df[metric],
                name=metric.replace('_', ' ').title(),
                text=df[metric].round(3),
                textposition='auto'
            ),
            row=1, col=i
        )
    
    # Update layout
    fig.update_layout(
        title=f"Per-Class Metrics - {os.path.basename(model_dir)}",
        height=500,
        width=1500,
        showlegend=False,
        yaxis_tickformat=".2%"
    )
    
    return fig

def generate_dashboard(model_dirs, output_file='performance_dashboard.html'):
    """Generate an HTML dashboard with model performance metrics."""
    # Load model metrics
    df = load_model_metrics(model_dirs)
    
    if df.empty:
        print("No model metrics found. Exiting...")
        return
    
    # Create dashboard
    fig = go.Figure()
    
    # Add title
    fig.add_annotation(
        text="<b>Cattle Breed Classification - Model Performance Dashboard</b>",
        x=0.5, y=1.1, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=24)
    )
    
    # Add timestamp
    fig.add_annotation(
        text=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        x=0.5, y=1.05, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    # Add empty layout (we'll add subplots)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white"
    )
    
    # Create comparison plots
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    metric_titles = {
        'test_accuracy': 'Test Accuracy by Model',
        'test_precision': 'Test Precision by Model',
        'test_recall': 'Test Recall by Model',
        'test_f1': 'Test F1-Score by Model'
    }
    
    # Add comparison plots
    for i, metric in enumerate(metrics, 1):
        plot = create_comparison_plot(df, metric, metric_titles[metric])
        fig.add_trace(plot.data[0], row=((i-1)//2)+2, col=((i-1)%2)+1)
        fig.add_trace(plot.data[1], row=((i-1)//2)+2, col=((i-1)%2)+1)
    
    # Update layout for comparison plots
    fig.update_layout(
        grid={
            'rows': 4,
            'columns': 2,
            'pattern': 'independent',
            'roworder': 'top to bottom'
        },
        height=1500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add training history plots for each model
    for i, model_dir in enumerate(model_dirs, 1):
        history_plot = create_training_history_plot(model_dir)
        class_plot = create_class_metrics_plot(model_dir)
        
        if history_plot:
            for trace in history_plot.data:
                fig.add_trace(trace, row=i+5, col=1)
                
        if class_plot:
            for trace in class_plot.data:
                fig.add_trace(trace, row=i+5, col=2)
    
    # Save dashboard
    fig.write_html(
        output_file,
        full_html=True,
        include_plotlyjs='cdn',
        config={'displayModeBar': True}
    )
    
    print(f"Dashboard saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance dashboard')
    parser.add_argument('--model_dirs', nargs='+', required=True,
                       help='List of model directories to include in the dashboard')
    parser.add_argument('--output', default='performance_dashboard.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    generate_dashboard(args.model_dirs, args.output)
