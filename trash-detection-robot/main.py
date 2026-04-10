import cv2
import argparse
import time
import os
from src.camera import CameraStream
from src.infer import TrashDetector
from src.coordinate_mapper import CoordinateMapper
from src.robot_control import RobotControl
from src.utils import FPSCounter, draw_detections, setup_logger

logger = setup_logger(\"main_pipeline\")

def main(args):
    logger.info(\"Initializing modules...\")
    
    # 1. Init Camera
    cam = CameraStream(width=args.width, height=args.height).start()
    
    # Wait for camera to warm up
    time.sleep(2.0)
    
    # 2. Init Detector
    if not os.path.exists(args.model):
        logger.warning(f\"Model {args.model} not found! Please run training/export first.\")
        # Initialize with dummy missing model path, it might crash or use pytorch default if configured
        
    detector = TrashDetector(model_path=args.model, conf_threshold=args.conf_thresh)
    
    # 3. Init Mapper
    mapper = CoordinateMapper(frame_width=args.width, frame_height=args.height)
    
    # 4. Init Robot
    robot = RobotControl()
    
    # 5. FPS Counter
    fps_counter = FPSCounter()
    
    try:
        logger.info(\"Starting main loop...\")
        while True:
            frame = cam.read()
            if frame is None:
                continue
                
            # Detect
            boxes, confidences, class_ids = detector.detect(frame)
            
            # Logic flow: Target the first detected box (or highest confidence)
            if boxes:
                # Naive: pick the first box. Better logic could sort by confidence.
                target_box = boxes[0]
                
                # Draw for visualization
                frame = draw_detections(frame, boxes, detector.class_names, confidences, class_ids)
                
                # Map to normalized space
                norm_x, norm_y = mapper.get_robot_mapped_coordinates(target_box)
                
                # Command Robot
                if not robot.is_holding:
                    ready_to_pick = robot.move_to(norm_x, norm_y)
                    if ready_to_pick:
                        robot.stop_motors()
                        robot.pick()
                        # Add release logic after picking (e.g. drop in bin over time)
                        time.sleep(3) # simulate dropping timeline
                        robot.release()
            else:
                # No trash found, maybe scan around?
                robot.stop_motors()

            # Record FPS
            fps = fps_counter.update()
            cv2.putText(frame, f\"FPS: {fps:.1f}\", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame (optional, disable for pure headless speed)
            if not args.headless:
                cv2.imshow(\"Trash Detector Robot\", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        logger.info(\"Interrupted by user.\")
    finally:
        logger.info(\"Cleaning up resources...\")
        cam.stop()
        robot.cleanup()
        cv2.destroyAllWindows()

if __name__ == \"__main__\":
    parser = argparse.ArgumentParser(description=\"Trash Detection Robot Main Pipeline\")
    parser.add_argument('--model', type=str, default='best.onnx', help='Path to ONNX/TFLite/PyTorch model')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--width', type=int, default=640, help='Camera capture width')
    parser.add_argument('--height', type=int, default=480, help='Camera capture height')
    parser.add_argument('--headless', action='store_true', help='Run without CV2 GUI displaying loop')
    
    args = parser.parse_args()
    main(args)
