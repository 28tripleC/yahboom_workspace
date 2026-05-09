import os
import time
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ArucoDistanceTest(Node):
    def __init__(self):
        super().__init__('aruco_distance_test')

        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(
            Image,
            '/esp32_img',
            self.image_callback,
            1
        )

        self.declare_parameter('marker_size', 0.06)
        self.declare_parameter('use_calibration', True)
        self.declare_parameter(
            'calib_file',
            os.path.expanduser('~/camera_calibration/calibration_data.npz')
        )

        self.marker_size = float(self.get_parameter('marker_size').value)
        self.use_calibration = bool(self.get_parameter('use_calibration').value)
        self.calib_file = self.get_parameter('calib_file').value

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        if self.use_calibration and os.path.exists(self.calib_file):
            data = np.load(self.calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeff']
            self.get_logger().info(f"Loaded calibration: {self.calib_file}")
        else:
            self.camera_matrix = np.array([
                [400, 0, 320],
                [0, 400, 240],
                [0, 0, 1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            self.get_logger().warn("Using default camera parameters")

        self.frame_count = 0
        self.last_print_time = time.time()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None:
            cv2.imshow("Aruco Distance Test", frame)
            cv2.waitKey(1)
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        obj_points = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0],
        ], dtype=np.float64)

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            corner = corners[i]

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corner[0],
                self.camera_matrix,
                self.dist_coeffs
            )

            if not success:
                continue

            distance = float(np.linalg.norm(tvec))
            z_distance = float(tvec[2][0])

            cx = int(corner[0][:, 0].mean())
            cy = int(corner[0][:, 1].mean())

            cv2.drawFrameAxes(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                0.05
            )

            cv2.putText(
                frame,
                f"ID:{marker_id} dist:{distance:.3f}m z:{z_distance:.3f}m",
                (cx - 80, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

            now = time.time()
            if now - self.last_print_time > 0.3:
                self.get_logger().info(
                    f"id={marker_id}, distance={distance:.4f}, z={z_distance:.4f}"
                )
                self.last_print_time = now

        cv2.imshow("Aruco Distance Test", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDistanceTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()