import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import json
import time 

class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')

        self.sub_img = self.create_subscription(
            Image, '/esp32_img', self.image_callback, 1)

        self.bridge = CvBridge()
        
        self.board_size = (9, 7)
        self.square_size = 0.025

        self.last_capture_time = 0
        self.capture_interval = 5.0

        self.obj_points = [] # 3D points in real world space
        self.img_points = [] # 2D points in image plane

        self.image_size = None
        self.capture_count = 0
        self.min_captures = 25

        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.save_path = os.path.expanduser('~/camera_calibration')
        os.makedirs(self.save_path, exist_ok=True)

        self.get_logger().info("Camera Calibrator started")
        self.get_logger().info("Press 'c' to capture calibration image when the chessboard is detected")
        self.get_logger().info(f"Need at least {self.min_captures} captures for calibration")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)

        display_frame = frame.copy()

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(display_frame, self.board_size, corners_refined, ret)
            status = "Chessboard Detected - Press 'c' to Capture"
            color = (0, 255, 0)
        
        else:
            status = "Chessboard Not Detected"
            color = (0, 0, 255)

        cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"Captures: {self.capture_count}/{self.min_captures}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Camera Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if ret and (time.time() - self.last_capture_time > self.capture_interval):
            self.obj_points.append(self.objp)
            self.img_points.append(corners_refined)
            self.capture_count += 1
            self.get_logger().info(f"Captured {self.capture_count}/{self.min_captures}")
            self.last_capture_time = time.time()

            cv2.imwrite(os.path.join(self.save_path, f'calib_{self.capture_count}.jpg'), display_frame)

        elif key == ord('q'):
            self.get_logger().info("Exiting calibration")
            cv2.destroyAllWindows()
            rclpy.shutdown()
    
    def run_calibration(self):
        if self.capture_count < self.min_captures:
            self.get_logger().warn(f"Not enough captures for calibration. Need {self.min_captures}, got {self.capture_count}")
            return
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_size, None, None)

        if ret:
            calib_data = {
                'camera_matrix': mtx.tolist(),
                'dist_coeff': dist.tolist(),
                'rms_error': ret,
                'image_size': list(self.image_size),
                'num_captures': self.capture_count
            }
            with open(os.path.join(self.save_path, 'calibration_data.json'), 'w') as f:
                json.dump(calib_data, f, indent=2)
            
            np.savez(
                os.path.join(self.save_path, 'calibration_data.npz'),
                camera_matrix=mtx,
                dist_coeff=dist
            )
            self.get_logger().info("Calibration successful.")
        else:
            self.get_logger().error("Calibration failed")
        
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    calibrator = CameraCalibrator()
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        pass
    finally:
        calibrator.run_calibration()
        calibrator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


