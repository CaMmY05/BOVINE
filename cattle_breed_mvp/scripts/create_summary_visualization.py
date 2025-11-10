import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_classification_report(report_path):
    """Load classification report from JSON file."""
    with open(report_path, 'r') as f:
        report = json.load(f)
    return report

def create_confusion_matrix_plot(cm_path):
    """Create a confusion matrix plot."""
    cm = pd.read_csv(cm_path, index_col=0)
    fig = go.Figure(data=go.Heatmap(
        z=cm.values,
        x=cm.columns,
        y=cm.index,
        colorscale='Blues',
        showscale=True,
        text=cm.values,
        texttemplate="%{text}",
        textfont={"size":12}
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,
        height=500
    )
    return fig

def create_class_distribution_plot(class_dist_path):
    """Create a class distribution bar plot."""
    class_dist = pd.read_csv(class_dist_path)
    fig = px.bar(
        class_dist, 
        x='class', 
        y='count',
        title='Class Distribution',
        labels={'count': 'Number of Samples', 'class': 'Class'},
        color='class',
        text='count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    return fig

def create_metrics_plot(report):
    """Create a bar plot of precision, recall, and F1-score."""
    metrics = ['precision', 'recall', 'f1-score']
    classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [report[cls][metric] for cls in classes]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=classes,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Performance Metrics by Class',
        barmode='group',
        xaxis_title='Class',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.1]),
        height=500
    )
    
    return fig

def create_summary_dashboard(model_dir):
    """Create a summary dashboard with all visualizations."""
    # Define paths
    eval_dir = os.path.join(model_dir, 'evaluation')
    report_path = os.path.join(eval_dir, 'classification_report.json')
    class_dist_path = os.path.join(eval_dir, 'class_distribution.csv')
    
    # Check if required files exist
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Classification report not found at {report_path}")
    if not os.path.exists(class_dist_path):
        # Try to create class distribution from confusion matrix if available
        if os.path.exists(os.path.join(eval_dir, 'confusion_matrix.png')):
            print("Warning: Using placeholder for class distribution as CSV not found")
            # Create a simple class distribution based on the report
            report = load_classification_report(report_path)
            classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
            class_dist = pd.DataFrame({
                'class': classes,
                'count': [int(report[cls]['support']) for cls in classes]
            })
            class_dist.to_csv(class_dist_path, index=False)
        else:
            raise FileNotFoundError(f"Class distribution data not found at {class_dist_path}")
    
    # Load data
    report = load_classification_report(report_path)
    
    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            'Confusion Matrix',
            'Class Distribution',
            'Performance Metrics by Class'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"colspan": 2, "type": "bar"}, None],
        ]
    )
    
    # Add confusion matrix if available, otherwise add a placeholder
    cm_path_png = os.path.join(eval_dir, 'confusion_matrix.png')
    if os.path.exists(cm_path_png):
        # Create a placeholder since we can't directly read the PNG
        # We'll add the image as an annotation
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref="paper",
            yref="paper",
            text="Confusion Matrix (see separate file)",
            showarrow=False,
            font=dict(size=14)
        )
    else:
        # Try to load as CSV if PNG not found
        try:
            cm = pd.read_csv(cm_path, index_col=0)
            fig.add_trace(
                go.Heatmap(
                    z=cm.values,
                    x=cm.columns,
                    y=cm.index,
                    colorscale='Blues',
                    showscale=True,
                    text=cm.values,
                    texttemplate="%{text}",
                    textfont={"size":12}
                ),
                row=1, col=1
            )
        except Exception as e:
            print(f"Could not load confusion matrix: {e}")
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Confusion Matrix not available",
                showarrow=False,
                font=dict(size=14)
            )
    
    # Add class distribution
    class_dist = pd.read_csv(class_dist_path)
    fig.add_trace(
        go.Bar(
            x=class_dist['class'],
            y=class_dist['count'],
            name='Samples',
            marker_color='#1f77b4',
            text=class_dist['count'],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Add metrics
    metrics = ['precision', 'recall', 'f1-score']
    classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        fig.add_trace(
            go.Bar(
                name=metric.capitalize(),
                x=classes,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition='auto',
                visible=(i == 0)  # Only show first metric by default
            ),
            row=2, col=1
        )
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": [True, True] + [i == j for j in range(3) for _ in classes]},
                             {"title": f"Performance Metrics: {metric.upper()}"}],
                        label=metric.capitalize(),
                        method="update"
                    )
                    for i, metric in enumerate(metrics)
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"Model Evaluation Summary - {os.path.basename(model_dir)}",
        showlegend=True,
        height=1000,
        width=1200,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Predicted', row=1, col=1)
    fig.update_yaxes(title_text='Actual', row=1, col=1)
    fig.update_xaxes(title_text='Class', row=1, col=2)
    fig.update_yaxes(title_text='Number of Samples', row=1, col=2)
    fig.update_xaxes(title_text='Class', row=2, col=1)
    fig.update_yaxes(title_text='Score', row=2, col=1, range=[0, 1.1])
    
    # Save the figure
    os.makedirs(eval_dir, exist_ok=True)
    fig.write_html(os.path.join(eval_dir, 'summary_dashboard.html'))
    
    print(f"Summary dashboard saved to: {os.path.join(eval_dir, 'summary_dashboard.html')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create summary visualization for model evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the model and evaluation results')
    
    args = parser.parse_args()
    create_summary_dashboard(args.model_dir)
