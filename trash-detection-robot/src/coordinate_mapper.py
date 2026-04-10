class CoordinateMapper:
    """
    Maps pixel coordinates from object detection to robot-operable coordinates.
    """
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_center(self, box):
        """
        Compute center (cx, cy) from bounding box [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy

    def normalize(self, cx, cy):
        """
        Normalize coordinates to a range of [0, 1] for hardware agnostic representation.
        (0.5, 0.5) is the dead center of the frame.
        """
        norm_x = cx / self.frame_width
        norm_y = cy / self.frame_height
        # Clamp to [0, 1] just in case
        return min(max(norm_x, 0.0), 1.0), min(max(norm_y, 0.0), 1.0)
        
    def get_robot_mapped_coordinates(self, box):
        """
        Return normalized center coordinates suitable for robot.move_to logic.
        """
        cx, cy = self.get_center(box)
        return self.normalize(cx, cy)
