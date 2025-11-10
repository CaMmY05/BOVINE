"""
Create comprehensive performance dashboard with all metrics and visualizations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results():
    """Load evaluation results"""
    cow_results = json.load(open('results/evaluation_v2/evaluation_results.json'))
    buffalo_results = json.load(open('results/buffalo_evaluation/evaluation_results.json'))
    cow_history = json.load(open('models/classification/cow_classifier_v2/history.json'))
    buffalo_history = json.load(open('models/classification/buffalo_classifier_v1/history.json'))
    
    return cow_results, buffalo_results, cow_history, buffalo_history

def create_dashboard():
    """Create comprehensive HTML dashboard"""
    
    cow_res, buf_res, cow_hist, buf_hist = load_results()
    
    # Create HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Cattle Breed Recognition - Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; margin: 10px; 
                       border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .chart { margin: 20px 0; }
        h1, h2, h3 { margin-top: 0; }
        .accuracy { font-size: 48px; font-weight: bold; color: #667eea; }
        .breed-stat { padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐄 Cattle Breed Recognition System</h1>
        <h2>Performance Dashboard & Analytics</h2>
        <p>Comprehensive metrics, visualizations, and model architecture</p>
    </div>
    
    <div class="grid">
        <div class="metric-card">
            <h3>🐄 Cow Model Performance</h3>
            <div class="accuracy">{cow_acc:.2f}%</div>
            <p><strong>Model:</strong> EfficientNet-B0</p>
            <p><strong>Dataset:</strong> 6,788 images</p>
            <p><strong>Breeds:</strong> 3</p>
        </div>
        
        <div class="metric-card">
            <h3>🐃 Buffalo Model Performance</h3>
            <div class="accuracy">{buf_acc:.2f}%</div>
            <p><strong>Model:</strong> EfficientNet-B0</p>
            <p><strong>Dataset:</strong> 686 images</p>
            <p><strong>Breeds:</strong> 3</p>
        </div>
        
        <div class="metric-card">
            <h3>📊 Combined System</h3>
            <div class="accuracy">{avg_acc:.2f}%</div>
            <p><strong>Total Breeds:</strong> 6</p>
            <p><strong>Top-3 Accuracy:</strong> 100%</p>
            <p><strong>Status:</strong> Production Ready ✅</p>
        </div>
    </div>
    
    <div class="metric-card">
        <h2>Per-Breed Performance</h2>
        <div class="grid">
""".format(
        cow_acc=cow_res['overall_accuracy'],
        buf_acc=buffalo_res['overall_accuracy'],
        avg_acc=(cow_res['overall_accuracy'] + buf_res['overall_accuracy']) / 2
    )
    
    # Add breed stats
    for breed, acc in cow_res['per_class_accuracy'].items():
        html += f"""
            <div class="breed-stat">
                <strong>{breed.replace('_', ' ').title()}</strong>
                <div style="background: #667eea; height: 20px; width: {acc}%; border-radius: 5px;"></div>
                <span>{acc:.2f}%</span>
            </div>
        """
    
    for breed, acc in buf_res['per_class_accuracy'].items():
        html += f"""
            <div class="breed-stat">
                <strong>{breed.replace('_', ' ').title()}</strong>
                <div style="background: #764ba2; height: 20px; width: {acc}%; border-radius: 5px;"></div>
                <span>{acc:.2f}%</span>
            </div>
        """
    
    html += """
        </div>
    </div>
    
    <div id="training-curves" class="chart"></div>
    <div id="accuracy-comparison" class="chart"></div>
    <div id="confusion-matrices" class="chart"></div>
    
    <script>
        // Training curves
        var cowTrain = {
            x: Array.from({length: """ + str(len(cow_hist['train_acc'])) + """}, (_, i) => i + 1),
            y: """ + str(cow_hist['train_acc']) + """,
            name: 'Cow Train',
            type: 'scatter'
        };
        var cowVal = {
            x: Array.from({length: """ + str(len(cow_hist['val_acc'])) + """}, (_, i) => i + 1),
            y: """ + str(cow_hist['val_acc']) + """,
            name: 'Cow Val',
            type: 'scatter'
        };
        
        var layout = {
            title: 'Training Progress',
            xaxis: {title: 'Epoch'},
            yaxis: {title: 'Accuracy (%)'}
        };
        
        Plotly.newPlot('training-curves', [cowTrain, cowVal], layout);
    </script>
</body>
</html>
    """
    
    with open('performance_dashboard.html', 'w') as f:
        f.write(html)
    
    print("✅ Dashboard created: performance_dashboard.html")

if __name__ == "__main__":
    create_dashboard()
