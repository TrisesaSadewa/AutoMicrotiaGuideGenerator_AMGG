from ultralytics import YOLO
import yaml
import os

def train_model():
    # Load the configuration file
    with open('ear_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Set the path variable based on your configuration
    data_path = config.get('path', './ear_detection_yolo')

    print("Starting YOLOv8 training for 50 epochs...")
    print(f"Using configuration file: ear_config.yaml")
    
    # Check for CUDA availability and set device
    # Set device to '0' (for first GPU) if available, otherwise 'cpu'
    device_to_use = '0' if os.environ.get('CUDA_VISIBLE_DEVICES') else ('0' if torch.cuda.is_available() else 'cpu')
    
    # Load a pre-trained YOLOv8-n model (nano for speed)
    model = YOLO('yolov8n.pt') 

    # Start training
    results = model.train(
        data='ear_config.yaml',  # The data configuration file
        epochs=50,               # Number of epochs
        imgsz=640,               # Image size
        batch=16,                # Batch size (adjust based on your RTX 4050's VRAM)
        name='ear_detector_run', # Name for the run folder
        device=device_to_use     # *** CRITICAL: Use the detected device ***
    )
    
    print("\nTraining completed. Weights saved in runs/detect/...")

# If you run into issues, uncomment the line below to check PyTorch setup
# import torch
# print(f"Torch CUDA available after install: {torch.cuda.is_available()}")

# We must import torch here for the device check to work correctly
import torch 
if __name__ == "__main__":
    train_model()
