import time
import os
from src.utils import setup_logger

logger = setup_logger("robot_control")

# Dummy Mock setup if not on RPi
try:
    import RPi.GPIO as GPIO
    import pigpio
    ON_RPI = True
except (ImportError, RuntimeError):
    logger.warning("RPi.GPIO or pigpio not found. Running in mock mode.")
    ON_RPI = False

class RobotControl:
    """
    Controls the base chassis (DC motors) and the manipulator arm (Servo).
    Default configuration assumes an L298N motor driver.
    """
    def __init__(self, 
                 motor_left_pin1=17, motor_left_pin2=27, 
                 motor_right_pin1=22, motor_right_pin2=23, 
                 servo_pin=18):
        self.pins = {
            'L1': motor_left_pin1, 'L2': motor_left_pin2,
            'R1': motor_right_pin1, 'R2': motor_right_pin2,
            'SERVO': servo_pin
        }
        
        self.is_holding = False
        
        if ON_RPI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            # Setup motor pins
            for p in ['L1', 'L2', 'R1', 'R2']:
                GPIO.setup(self.pins[p], GPIO.OUT)
                GPIO.output(self.pins[p], GPIO.LOW)
            
            # Setup pigpio for accurate PWM for servo
            self.pi = pigpio.pi()
            if not self.pi.connected:
                logger.error("pigpio daemon not running! Try 'sudo pigpiod'. Servo will fail.")
            else:
                self.pi.set_mode(self.pins['SERVO'], pigpio.OUTPUT)
                
            self.release() # Start with open gripper

    def stop_motors(self):
        """Stop all DC motors."""
        if not ON_RPI: return
        for p in ['L1', 'L2', 'R1', 'R2']:
            GPIO.output(self.pins[p], GPIO.LOW)

    def move_forward(self, duration=0.1):
        """Move chassis forward slightly"""
        if ON_RPI:
            GPIO.output(self.pins['L1'], GPIO.HIGH)
            GPIO.output(self.pins['L2'], GPIO.LOW)
            GPIO.output(self.pins['R1'], GPIO.HIGH)
            GPIO.output(self.pins['R2'], GPIO.LOW)
            time.sleep(duration)
            self.stop_motors()
        else:
            logger.debug("MOCK: Moving forward")

    def turn_left(self, duration=0.1):
        if ON_RPI:
            GPIO.output(self.pins['L1'], GPIO.LOW)
            GPIO.output(self.pins['L2'], GPIO.HIGH)
            GPIO.output(self.pins['R1'], GPIO.HIGH)
            GPIO.output(self.pins['R2'], GPIO.LOW)
            time.sleep(duration)
            self.stop_motors()
        else:
            logger.debug("MOCK: Turning left")

    def turn_right(self, duration=0.1):
        if ON_RPI:
            GPIO.output(self.pins['L1'], GPIO.HIGH)
            GPIO.output(self.pins['L2'], GPIO.LOW)
            GPIO.output(self.pins['R1'], GPIO.LOW)
            GPIO.output(self.pins['R2'], GPIO.HIGH)
            time.sleep(duration)
            self.stop_motors()
        else:
            logger.debug("MOCK: Turning right")

    def move_to(self, norm_x, norm_y):
        """
        Use normalized coordinates to center the robot on the target.
        norm_x: 0 (left edge) to 1 (right edge). Target is ~0.5.
        norm_y: 0 (top edge) to 1 (bottom edge). Target is approaching 1 (close to robot).
        """
        # Simple proportional visual servoing proxy
        centered_x = abs(norm_x - 0.5) < 0.15 # 15% tolerance
        close_enough_y = norm_y > 0.8 # Near bottom of frame
        
        if not centered_x:
            if norm_x < 0.5:
                logger.info(f"Target left (x={norm_x:.2f}). Turning left.")
                self.turn_left()
            else:
                logger.info(f"Target right (x={norm_x:.2f}). Turning right.")
                self.turn_right()
            return False # Not yet at target
            
        if not close_enough_y:
            logger.info(f"Target ahead (y={norm_y:.2f}). Moving forward.")
            self.move_forward()
            return False # Not yet at target
            
        logger.info("Target centered and close enough!")
        return True # Ready to pick!

    def pick(self):
        """Close the gripper"""
        logger.info("Picking up object (closing gripper).")
        self.is_holding = True
        if ON_RPI and self.pi.connected:
            # PWM pulse width 500-2500, roughly 1500 is 90 deg.
            # Assuming 2000 is closed
            self.pi.set_servo_pulsewidth(self.pins['SERVO'], 2000)
            time.sleep(1)

    def release(self):
        """Open the gripper"""
        logger.info("Releasing object (opening gripper).")
        self.is_holding = False
        if ON_RPI and self.pi.connected:
            # Assuming 1000 is open
            self.pi.set_servo_pulsewidth(self.pins['SERVO'], 1000)
            time.sleep(1)

    def cleanup(self):
        """Release GPIO resources"""
        if ON_RPI:
            self.stop_motors()
            GPIO.cleanup()
            if self.pi.connected:
                self.pi.set_servo_pulsewidth(self.pins['SERVO'], 0) # Off
                self.pi.stop()
