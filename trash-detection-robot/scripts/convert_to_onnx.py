import argparse
from ultralytics import YOLO

def export_to_onnx(model_path, imgsz=320, dynamic=False):
    """
    Export PyTorch model to ONNX format optimized for CPU inference on Raspberry Pi
    """
    try:
        model = YOLO(model_path)
        print(f"Loaded {model_path} successfully. Starting ONNX export...")
        
        # Exporting to ONNX
        path = model.export(
            format='onnx', 
            imgsz=imgsz, 
            dynamic=dynamic,
            simplify=True # Simplifies the model graph for better ONNX runtime performance
        )
        print(f"Export successful! ONNX model saved at: {path}")
        return path
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX")
    parser.add_argument('--model', type=str, required=True, help='Path to the .pt file (e.g. runs/detect/trash_robot/weights/best.pt)')
    parser.add_argument('--imgsz', type=int, default=320, help='Inference image size (smaller = faster)')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic axes for batching and image size')
    
    args = parser.parse_args()
    export_to_onnx(args.model, imgsz=args.imgsz, dynamic=args.dynamic)
