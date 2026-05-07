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

        # ---------- ROS parameters ----------
        self.declare_parameter('marker_size', 0.06)
        self.declare_parameter('use_calibration', True)
        self.declare_parameter(
            'calib_file',
            os.path.expanduser('~/camera_calibration/calibration_data.npz')
        )
        self.declare_parameter('target_id', 0)
        self.declare_parameter('min_marker_area', 300.0)
        self.declare_parameter('fixed_samples', 0)
        self.declare_parameter('history_size', 5)
        self.declare_parameter('show_image', True)

        self.marker_size = float(self.get_parameter('marker_size').value)
        self.use_calibration = bool(self.get_parameter('use_calibration').value)
        self.calib_file = self.get_parameter('calib_file').value
        self.target_id = int(self.get_parameter('target_id').value)
        self.min_marker_area = float(self.get_parameter('min_marker_area').value)
        self.fixed_samples = int(self.get_parameter('fixed_samples').value)
        self.history_size = int(self.get_parameter('history_size').value)
        self.show_image = bool(self.get_parameter('show_image').value)

        self.history_size = max(1, self.history_size)

        # ---------- ArUco ----------
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # ---------- Camera parameters ----------
        if self.use_calibration and os.path.exists(self.calib_file):
            data = np.load(self.calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeff']
            self.get_logger().info(f"Loaded calibration: {self.calib_file}")
        else:
            # Empirical intrinsics:
            # assume 640x480 image, principal point at image center,
            # and approximate focal length fx=fy=400.
            self.camera_matrix = np.array([
                [400, 0, 320],
                [0, 400, 240],
                [0, 0, 1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            self.get_logger().warn("Using empirical/default camera parameters")

        self.obj_points = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0],
        ], dtype=np.float64)

        # ---------- Statistics ----------
        self.frame_count = 0
        self.detected_count = 0

        self.single_distances = []
        self.single_z_values = []

        self.avg5_distances = []
        self.avg5_z_values = []

        self.processing_times = []
        self.tvec_history = []

        self.summary_printed = False

        self.get_logger().info(
            f"marker_size={self.marker_size:.3f}m, "
            f"target_id={self.target_id}, "
            f"fixed_samples={self.fixed_samples}, "
            f"history_size={self.history_size}, "
            f"use_calibration={self.use_calibration}"
        )

    def image_callback(self, msg):
        start_time = time.time()
        self.frame_count += 1

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        detected_this_frame = False
        best_id = None
        single_distance = None
        single_z = None
        avg5_distance = None
        avg5_z = None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])

                if marker_id != self.target_id:
                    continue

                corner = corners[i]
                area = cv2.contourArea(corner)
                if area < self.min_marker_area:
                    continue

                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    corner[0],
                    self.camera_matrix,
                    self.dist_coeffs
                )

                if not success:
                    continue

                detected_this_frame = True
                best_id = marker_id

                single_distance = float(np.linalg.norm(tvec))
                single_z = float(tvec[2][0])

                # ---------- 5-frame fusion ----------
                self.tvec_history.append(tvec)
                if len(self.tvec_history) > self.history_size:
                    self.tvec_history.pop(0)

                avg_tvec = np.mean(self.tvec_history, axis=0)
                avg5_distance = float(np.linalg.norm(avg_tvec))
                avg5_z = float(avg_tvec[2][0])

                self.detected_count += 1
                self.single_distances.append(single_distance)
                self.single_z_values.append(single_z)
                self.avg5_distances.append(avg5_distance)
                self.avg5_z_values.append(avg5_z)

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
                    f"ID:{marker_id} d:{single_distance:.3f} avg:{avg5_distance:.3f}",
                    (max(cx - 110, 0), max(cy - 20, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1
                )

                break

        elapsed_ms = (time.time() - start_time) * 1000.0
        self.processing_times.append(elapsed_ms)

        # ---------- Per-frame log ----------
        if detected_this_frame:
            self.get_logger().info(
                f"frame={self.frame_count}, detected=1, "
                f"id={best_id}, "
                f"distance={single_distance:.4f}, "
                f"z={single_z:.4f}, "
                f"avg5_distance={avg5_distance:.4f}, "
                f"avg5_z={avg5_z:.4f}, "
                f"processing_time_ms={elapsed_ms:.2f}"
            )
        else:
            self.get_logger().info(
                f"frame={self.frame_count}, detected=0, "
                f"processing_time_ms={elapsed_ms:.2f}"
            )

        # ---------- On-image statistics ----------
        if self.show_image:
            success_rate = (
                self.detected_count / self.frame_count * 100.0
                if self.frame_count > 0 else 0.0
            )

            cv2.putText(
                frame,
                f"Frames:{self.frame_count} Det:{self.detected_count} "
                f"Rate:{success_rate:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Proc:{elapsed_ms:.2f} ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("Aruco Distance / Fusion Test", frame)
            cv2.waitKey(1)

        # ---------- Fixed sample auto stop ----------
        if self.fixed_samples > 0 and self.frame_count >= self.fixed_samples:
            self.print_summary()
            self.summary_printed = True
            raise KeyboardInterrupt

    def _stats_text(self, name, values):
        if not values:
            return f"{name}: no_data"

        arr = np.array(values, dtype=np.float64)
        return (
            f"{name}_mean={arr.mean():.4f}, "
            f"{name}_std={arr.std():.4f}, "
            f"{name}_min={arr.min():.4f}, "
            f"{name}_max={arr.max():.4f}, "
            f"{name}_range={(arr.max() - arr.min()):.4f}"
        )

    def print_summary(self):
        self.get_logger().info("========== TEST SUMMARY ==========")
        self.get_logger().info(f"total_frames={self.frame_count}")
        self.get_logger().info(f"detected_frames={self.detected_count}")

        if self.frame_count > 0:
            rate = self.detected_count / self.frame_count * 100.0
            self.get_logger().info(f"detection_success_rate={rate:.2f}%")

        self.get_logger().info(self._stats_text("distance", self.single_distances))
        self.get_logger().info(self._stats_text("z", self.single_z_values))
        self.get_logger().info(self._stats_text("avg5_distance", self.avg5_distances))
        self.get_logger().info(self._stats_text("avg5_z", self.avg5_z_values))

        if self.processing_times:
            arr = np.array(self.processing_times, dtype=np.float64)
            self.get_logger().info(
                f"processing_mean_ms={arr.mean():.3f}, "
                f"processing_std_ms={arr.std():.3f}, "
                f"processing_min_ms={arr.min():.3f}, "
                f"processing_max_ms={arr.max():.3f}"
            )

        if self.single_distances and self.avg5_distances:
            single_std = np.array(self.single_distances).std()
            avg5_std = np.array(self.avg5_distances).std()

            if single_std > 1e-9:
                reduction = (single_std - avg5_std) / single_std * 100.0
                self.get_logger().info(
                    f"fusion_std_reduction={reduction:.2f}%"
                )

        self.get_logger().info("==================================")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDistanceTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if not node.summary_printed:
            node.print_summary()
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()