"""Extract all metrics from evaluation results"""
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef

def parse_classification_report(report_str):
    """Parse sklearn classification report string"""
    lines = report_str.strip().split('\n')
    metrics = {}
    
    for line in lines[2:-4]:  # Skip header and summary lines
        parts = line.split()
        if len(parts) >= 5:
            breed = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            support = int(parts[4])
            metrics[breed] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100,
                'support': support
            }
    
    # Get macro and weighted averages
    for line in lines[-3:]:
        if 'macro avg' in line:
            parts = line.split()
            metrics['macro_avg'] = {
                'precision': float(parts[2]) * 100,
                'recall': float(parts[3]) * 100,
                'f1_score': float(parts[4]) * 100
            }
        elif 'weighted avg' in line:
            parts = line.split()
            metrics['weighted_avg'] = {
                'precision': float(parts[2]) * 100,
                'recall': float(parts[3]) * 100,
                'f1_score': float(parts[4]) * 100
            }
    
    return metrics

def calculate_additional_metrics(cm):
    """Calculate additional metrics from confusion matrix"""
    cm = np.array(cm)
    
    # True positives, false positives, false negatives for each class
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    
    # Specificity per class
    specificity = tn / (tn + fp)
    
    # NPV (Negative Predictive Value)
    npv = tn / (tn + fn)
    
    # Matthews Correlation Coefficient (overall)
    y_true = []
    y_pred = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            y_true.extend([i] * cm[i][j])
            y_pred.extend([j] * cm[i][j])
    
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return {
        'specificity': specificity.tolist(),
        'npv': npv.tolist(),
        'mcc': mcc,
        'cohen_kappa': kappa
    }

# Load results
cow_res = json.load(open('results/evaluation_v2/evaluation_results.json'))
buf_res = json.load(open('results/buffalo_evaluation/evaluation_results.json'))

# Parse reports
cow_metrics = parse_classification_report(cow_res['classification_report'])
buf_metrics = parse_classification_report(buf_res['classification_report'])

# Calculate additional metrics
cow_additional = calculate_additional_metrics(cow_res['confusion_matrix'])
buf_additional = calculate_additional_metrics(buf_res['confusion_matrix'])

# Save enhanced metrics
enhanced_cow = {
    **cow_res,
    'detailed_metrics': cow_metrics,
    'additional_metrics': cow_additional
}

enhanced_buf = {
    **buf_res,
    'detailed_metrics': buf_metrics,
    'additional_metrics': buf_additional
}

json.dump(enhanced_cow, open('results/evaluation_v2/enhanced_metrics.json', 'w'), indent=2)
json.dump(enhanced_buf, open('results/buffalo_evaluation/enhanced_metrics.json', 'w'), indent=2)

print("✅ Enhanced metrics extracted!")
print(f"\nCow Model:")
print(f"  Macro F1: {cow_metrics['macro_avg']['f1_score']:.2f}%")
print(f"  MCC: {cow_additional['mcc']:.4f}")
print(f"  Cohen's Kappa: {cow_additional['cohen_kappa']:.4f}")

print(f"\nBuffalo Model:")
print(f"  Macro F1: {buf_metrics['macro_avg']['f1_score']:.2f}%")
print(f"  MCC: {buf_additional['mcc']:.4f}")
print(f"  Cohen's Kappa: {buf_additional['cohen_kappa']:.4f}")
