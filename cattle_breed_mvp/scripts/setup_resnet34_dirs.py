import os

def create_directories():
    base_path = "models/classification"
    models = ["resnet34_cow_v1", "resnet34_buffalo_v1"]
    
    for model in models:
        # Create main model directory
        os.makedirs(os.path.join(base_path, model, "evaluation"), exist_ok=True)
        print(f"Created directory: {os.path.join(base_path, model, 'evaluation')}")
        
        # Create checkpoints directory
        os.makedirs(os.path.join(base_path, model, "checkpoints"), exist_ok=True)
        print(f"Created directory: {os.path.join(base_path, model, 'checkpoints')}")
        
        # Create logs directory
        os.makedirs(os.path.join(base_path, model, "logs"), exist_ok=True)
        print(f"Created directory: {os.path.join(base_path, model, 'logs')}")

if __name__ == "__main__":
    print("Setting up ResNet-34 model directories...")
    create_directories()
    print("Directory setup completed successfully.")
