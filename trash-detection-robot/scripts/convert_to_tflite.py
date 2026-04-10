import argparse
from ultralytics import YOLO

def export_to_tflite(model_path, imgsz=320, int8=False):
    """
    Export PyTorch model to TFLite format (FP16 or INT8)
    """
    try:
        model = YOLO(model_path)
        print(f"Loaded {model_path} successfully. Starting TFLite export...")
        
        # Exporting to TFLite
        path = model.export(
            format='tflite', 
            imgsz=imgsz,
            int8=int8 # Enables INT8 quantization if true, otherwise defaults to FP16 generally
        )
        print(f"Export successful! TFLite model saved at: {path}")
        return path
    except Exception as e:
        print(f"Error exporting to TFLite: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert YOLO model to TFLite")
    parser.add_argument('--model', type=str, required=True, help='Path to the .pt file')
    parser.add_argument('--imgsz', type=int, default=320, help='Inference image size')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 quantization (requires rep dataset ideally)')
    
    args = parser.parse_args()
    export_to_tflite(args.model, imgsz=args.imgsz, int8=args.int8)
