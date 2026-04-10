import argparse
from ultralytics import YOLO

def train_model(data_path='config/data.yaml', epochs=50, imgsz=640, model_type='yolov8s.pt'):
    """
    Train a YOLOv8 model for trash detection.
    """
    # Load a pre-trained model (recommended for training)
    print(f"Loading model {model_type}...")
    model = YOLO(model_type)

    # Train the model
    print(f"Starting training with data={data_path}, epochs={epochs}, imgsz={imgsz}...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        project='runs/detect',
        name='trash_robot',
        exist_ok=True
    )
    
    print(f"Training completed. Models saved to runs/detect/trash_robot/weights/")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Trash Detection")
    parser.add_argument('--data', type=str, default='config/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model to start from (e.g., yolov8s.pt, yolov8n.pt)')
    
    args = parser.parse_args()
    
    train_model(data_path=args.data, epochs=args.epochs, imgsz=args.imgsz, model_type=args.model)
