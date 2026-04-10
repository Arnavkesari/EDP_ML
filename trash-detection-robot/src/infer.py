import time
import numpy as np
import cv2

class TrashDetector:
    
    def __init__(self, model_path=\"runs/detect/trash_robot/weights/best.pt\", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.is_onnx = model_path.endswith('.onnx')
        self.is_tflite = model_path.endswith('.tflite')
        self.class_names = {0: 'plastic', 1: 'metal', 2: 'paper', 3: 'glass'}
        
        print(f"Loading model: {model_path}")
        
        if self.is_onnx:
            import onnxruntime as ort
            # Use CPUExecutionProvider for RPi
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            
        elif self.is_tflite:
            # Fallback for TFLite
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        else:
            # PyTorch default using Ultralytics
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    def preprocess(self, img, size=(320, 320)):
        img_resized = cv2.resize(img, size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and channel-first config for ONNX
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        return img_batch, img_resized

    def postprocess_onnx(self, outputs, orig_w, orig_h, img_size=(320, 320)):
        boxes, confidences, class_ids = [], [], []
        # YOLOv8 ONNX output shape is generally [1, 4+num_classes, num_anchors]
        output = outputs[0][0]
        output = output.T # Transpose to [num_anchors, 4+num_classes]
        
        for row in output:
            box = row[0:4] # cx, cy, w, h
            scores = row[4:] 
            class_id = np.argmax(scores)
            conf = scores[class_id]
            
            if conf > self.conf_threshold:
                cx, cy, w, h = box
                x1 = int((cx - w/2) * (orig_w / img_size[0]))
                y1 = int((cy - h/2) * (orig_h / img_size[1]))
                x2 = int((cx + w/2) * (orig_w / img_size[0]))
                y2 = int((cy + h/2) * (orig_h / img_size[1]))
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(conf))
                class_ids.append(class_id)
                
        return boxes, confidences, class_ids

    def detect(self, img):
        if img is None:
            return [], [], []
            
        orig_h, orig_w = img.shape[:2]
            
        if self.is_onnx:
            img_tensor, _ = self.preprocess(img)
            outputs = self.session.run(None, {self.input_name: img_tensor})
            return self.postprocess_onnx(outputs, orig_w, orig_h)
            
        elif self.is_tflite:
            # TFLite typically expects channel last depending on export (NHWC)
            img_resized = cv2.resize(img, (320, 320))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], img_normalized)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(self.output_details[0]['index'])]
            # Assumes output shape matches ONNX for postprocessing step
            return self.postprocess_onnx(outputs, orig_w, orig_h)
            
        else:
            # PyTorch
            results = self.model(img, conf=self.conf_threshold, verbose=False)[0]
            boxes, confidences, class_ids = [], [], []
            if len(results.boxes):
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(box.conf[0]))
                    class_ids.append(int(box.cls[0]))
            return boxes, confidences, class_ids
