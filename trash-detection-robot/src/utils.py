import logging
import time
import cv2

def setup_logger(name="trash_robot", level=logging.INFO):
    """Setup basic logging"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class FPSCounter:
    """Simple FPS counter for performance tracking"""
    def __init__(self):
        self.start_time = time.time()
        self.frames = 0
        self.fps = 0.0

    def update(self):
        self.frames += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.start_time = time.time()
        return self.fps

def draw_detections(frame, boxes, class_names, confidences, classes):
    """
    Draw bounding boxes and labels on the frame.
    """
    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names.get(int(cls), 'Unknown')}: {conf:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return frame
