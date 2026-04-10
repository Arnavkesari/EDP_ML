import cv2
import threading
from src.utils import setup_logger

logger = setup_logger("camera")

class CameraStream:
    """
    A threaded camera stream for optimized FPS on Raspberry Pi.
    Uses threading to read frames independently of the main processing loop.
    """
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read the first frame
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            logger.error("Failed to initialize camera stream.")
            
        self.stopped = False
        self.width = width
        self.height = height

    def start(self):
        """Start the thread to read frames"""
        logger.info("Starting camera stream thread.")
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Continuously read frames from the camera"""
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """Return the most recent frame"""
        return self.frame
        
    def read_resized(self, size=(320, 320)):
        """Return frame resized for model inference"""
        if self.frame is not None:
            return cv2.resize(self.frame, size)
        return None

    def stop(self):
        """Stop the camera stream"""
        logger.info("Stopping camera stream.")
        self.stopped = True
