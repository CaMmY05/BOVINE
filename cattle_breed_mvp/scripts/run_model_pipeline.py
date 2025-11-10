"""
Cattle Breed Classification - Model Pipeline

This script provides a unified interface for training and evaluating all models
in the cattle breed classification system.
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from model_registry import (
    MODEL_REGISTRY, 
    get_model_config, 
    list_models, 
    save_evaluation_results,
    RESULTS_DIR
)

def setup_environment():
    """Set up Python environment and paths."""
    # Add project root to Python path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = project_root
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available

def run_command(cmd: str, cwd: str = None) -> int:
    """Run a shell command and return the return code."""
    print(f"\nRunning: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd or os.getcwd(),
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDERR: {e.stderr}")
        return e.returncode

def train_model(model_id: str, config: dict) -> bool:
    """Train a model using its configuration."""
    print(f"\n{'='*50}")
    print(f"Training {model_id}")
    print(f"Model: {config.name} {config.version} ({config.architecture})")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.model_path}")
    print(f"{'='*50}\n")
    
    # Create model directory if it doesn't exist
    config.model_path.mkdir(parents=True, exist_ok=True)
    
    # Build training command
    train_cmd = (
        f"python scripts/{config.train_script} "
        f"--data-dir {config.dataset_path} "
        f"--model-dir {config.model_path} "
        f"--batch-size {config.batch_size} "
        f"--epochs {config.epochs} "
        f"--lr {config.learning_rate}"
    )
    
    # Run training
    return run_command(train_cmd) == 0

def evaluate_model(model_id: str, config: dict) -> dict:
    """Evaluate a trained model."""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_id}")
    print(f"Model: {config.name} {config.version} ({config.architecture})")
    print(f"Dataset: {config.dataset_path}")
    print(f"Model path: {config.model_path}")
    print(f"{'='*50}\n")
    
    # Build evaluation command
    eval_cmd = (
        f"python scripts/{config.evaluate_script} "
        f"--data-dir {config.dataset_path} "
        f"--model-path {config.model_path}/best_model.pth "
        f"--output-dir {RESULTS_DIR}"
    )
    
    # Run evaluation
    if run_command(eval_cmd) != 0:
        return {"status": "error", "message": "Evaluation failed"}
    
    # Load and return results
    results_file = RESULTS_DIR / f"{model_id}_evaluation.json"
    if not results_file.exists():
        return {"status": "error", "message": "Evaluation completed but results not found"}
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def generate_report():
    """Generate a summary report of all models and their evaluations."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "models": []
    }
    
    for model_id in MODEL_REGISTRY.keys():
        config = get_model_config(model_id)
        results = get_evaluation_results(model_id)
        
        model_info = {
            "id": model_id,
            "name": config.name,
            "version": config.version,
            "architecture": config.architecture,
            "supported_animals": config.supported_animals,
            "model_path": str(config.model_path),
            "dataset_path": str(config.dataset_path),
            "evaluation": results
        }
        report["models"].append(model_info)
    
    # Save report
    report_file = RESULTS_DIR / "model_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport generated at: {report_file}")
    return report

def main():
    parser = argparse.ArgumentParser(description='Cattle Breed Classification Pipeline')
    parser.add_argument('--model', type=str, help='Model ID to process (default: all)')
    parser.add_argument('--train', action='store_true', help='Train the model(s)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model(s)')
    parser.add_argument('--report', action='store_true', help='Generate a summary report')
    
    args = parser.parse_args()
    
    if not any([args.train, args.evaluate, args.report]):
        parser.print_help()
        return
    
    setup_environment()
    
    # Process models
    models_to_process = [args.model] if args.model else MODEL_REGISTRY.keys()
    
    for model_id in models_to_process:
        if model_id not in MODEL_REGISTRY:
            print(f"Error: Unknown model ID: {model_id}")
            continue
            
        config = get_model_config(model_id)
        
        if args.train:
            if not train_model(model_id, config):
                print(f"Training failed for {model_id}")
                continue
        
        if args.evaluate:
            results = evaluate_model(model_id, config)
            save_evaluation_results(model_id, results)
            print(f"Evaluation results for {model_id}:\n{json.dumps(results, indent=2)}")
    
    if args.report:
        report = generate_report()
        print("\nModel Evaluation Summary:")
        print("=" * 50)
        for model in report["models"]:
            print(f"\n{model['name']} {model['version']} ({model['architecture']})")
            print(f"- ID: {model['id']}")
            print(f"- Supported Animals: {', '.join(model['supported_animals'])}")
            print(f"- Model Path: {model['model_path']}")
            print(f"- Dataset: {model['dataset_path']}")
            
            if 'status' in model['evaluation']:
                if model['evaluation']['status'] == 'not_evaluated':
                    print("- Status: Not evaluated")
                else:
                    print(f"- Status: {model['evaluation']['status']}")
                    if 'metrics' in model['evaluation']:
                        print("- Metrics:")
                        for k, v in model['evaluation']['metrics'].items():
                            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
