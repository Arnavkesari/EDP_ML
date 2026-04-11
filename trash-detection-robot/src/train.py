import argparse
import torch
from ultralytics import YOLO

def train_model(data_path='config/data.yaml', epochs=30, imgsz=640, model_type='yolov8s.pt', device='0'):
    """
    Train a YOLOv8 model for trash detection.
    """
    # Load a pre-trained model (recommended for training)
    print(f"Loading model {model_type}...")
    model = YOLO(model_type)

    if str(device).lower() != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU device was requested, but CUDA is not available in this environment. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )

    # Train the model
    print(f"Starting training with data={data_path}, epochs={epochs}, imgsz={imgsz}, device={device}...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project='runs/detect',
        name='trash_robot',
        exist_ok=True
    )
    
    print(f"Training completed. Models saved to runs/detect/trash_robot/weights/")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Trash Detection")
    parser.add_argument('--data', type=str, default='config/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model to start from (e.g., yolov8s.pt, yolov8n.pt)')
    parser.add_argument('--device', type=str, default='0', help='Training device: 0 for first GPU, cpu for CPU')
    
    args = parser.parse_args()
    
    train_model(data_path=args.data, epochs=args.epochs, imgsz=args.imgsz, model_type=args.model, device=args.device)
