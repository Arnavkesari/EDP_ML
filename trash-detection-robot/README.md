# Trash Detection and Autonomous Pickup Robot

A complete end-to-end project built for a Raspberry Pi 5. Detects trash via a camera using YOLOv8, converts its pixel coordinate to world coordinates, and commands a robot arm / driving system to pick it up.

## Quickstart

### Prerequisites
1. **Hardware**:
   - Raspberry Pi 5 (8GB)
   - USB or CSI Camera
   - L298N Motor Driver (or similar) & 2x DC Motors
   - Servo motor for the gripper arm
2. **Software**:
   - Python 3.9+ 

### Installation
```bash
git clone <repository>
cd trash-detection-robot
pip install -r requirements.txt
```

### Model Preparation (Run on PC/GPU)
Due to computational demands, it's recommended to train on a PC and deploy to the RPi:
```bash
python src/train.py
python scripts/convert_to_onnx.py --model runs/detect/train/weights/best.pt
```

### Running on Raspberry Pi
Copy the project folder including the `best.onnx` file to your Raspberry Pi, wire your motors according to the pins defined in `src/robot_control.py` (Default: ENV variables or static fallbacks), and run:
```bash
python main.py
```
