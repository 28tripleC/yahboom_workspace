import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import time
import json
import os
from datetime import datetime
import tf2_ros
import tf2_geometry_msgs

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        self.sub_img = self.create_subscription(
            Image, '/esp32_img', self.image_callback, 1)
        
        self.pub_img = self.create_publisher(Image, '/aruco_detected_img', 1)
        self.pub_markers = self.create_publisher(MarkerArray, '/inventory_markers', 1)

        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        calib_file = os.path.expanduser('~/camera_calibration/calibration_data.npz')
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeff']
            self.get_logger().info("Loaded camera calibration from file")
        else:
            self.get_logger().warning("Calibration file not found, using default parameters")
            self.camera_matrix = np.array([
                [400, 0, 320],
                [0, 400, 240],
                [0, 0, 1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((4,1), dtype=np.float64)

        self.marker_size = 0.06

        # 多帧融合
        self.detection_history = {}
        self.history_size = 5

        # 置信度过滤
        self.min_marker_area = 500

        # 物品ID与名称映射
        self.item_names = {}

        self.inventory = {}
        self.log_dir = os.path.expanduser('~/warehouse_log')
        os.makedirs(self.log_dir, exist_ok=True)

        self.mode = "register"
        self.baseline_path = os.path.join(self.log_dir, "baseline.json")
        self.baseline = {}

        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, 'r') as f:
                self.baseline = json.load(f)
            self.mode = "inspect"
            self.get_logger().info("Baseline loaded, switching to inspection mode")
            # Pre-mark all baseline items as Missing — detections upgrade to Normal/Displaced
            for marker_id, data in self.baseline.items():
                self.inventory[int(marker_id)] = {**data, "status": "Missing"}
        else:
            self.get_logger().info("No baseline found, starting in registration mode")

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Dynamic camera TF (this node owns camera_frame, not bringup)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.base_pitch = 0.3   # physical mount angle (radians)
        self.current_pitch = self.base_pitch
        self.create_timer(0.1, self._tf_timer_callback)  # publish TF at 10Hz

        # Servo publisher for vertical camera tilt
        self.pub_servo_y = self.create_publisher(Int32, 'servo_s2', 10)

        # Manual angle control for testing without patrol_node
        self.create_subscription(Int32, 'camera_angle', self._manual_angle_callback, 10)

        # Shelf row angles in degrees (positive = tilt up, negative = tilt down)
        self.declare_parameter('row_angles', [0, 20, 40])
        self.declare_parameter('scan_duration_per_row', 4.0)
        self.shelf_rows = self.get_parameter('row_angles').value
        self.scan_duration_per_row = self.get_parameter('scan_duration_per_row').value

        # Scan service — patrol_node calls this at each waypoint
        cb_group = ReentrantCallbackGroup()
        self.create_service(Trigger, 'scan_shelf', self.scan_shelf_callback,
                            callback_group=cb_group)

        # Only record detections during intentional scans, not while driving
        self.is_scanning = False

    def _tf_timer_callback(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_frame'
        t.transform.translation.x = 0.092
        t.transform.translation.y = 0.021
        t.transform.translation.z = 0.061
        half = self.current_pitch / 2.0
        t.transform.rotation.y = math.sin(half)
        t.transform.rotation.w = math.cos(half)
        self.tf_broadcaster.sendTransform(t)

    def set_camera_row(self, servo_angle_deg):
        msg = Int32()
        msg.data = int(servo_angle_deg)
        self.pub_servo_y.publish(msg)
        self.current_pitch = self.base_pitch + math.radians(servo_angle_deg)
        time.sleep(0.5)  # wait for servo to reach position

    def scan_shelf_callback(self, request, response):
        self.get_logger().info(f"Scanning {len(self.shelf_rows)} shelf rows...")
        self.is_scanning = True
        for i, servo_angle in enumerate(self.shelf_rows):
            self.get_logger().info(f"  Row {i+1}: servo={servo_angle}°")
            self.set_camera_row(servo_angle)
            time.sleep(self.scan_duration_per_row)
        self.set_camera_row(0)  # return to center
        self.is_scanning = False
        response.success = True
        response.message = f"Scanned {len(self.shelf_rows)} rows"
        self.get_logger().info("Shelf scan complete")
        return response

    def _manual_angle_callback(self, msg):
        angle = max(-90, min(90, msg.data))  # clamp to safe range
        self.get_logger().info(f"Manual camera angle: {angle}°")
        self.set_camera_row(angle)

    def transform_to_map(self, tvec, rvec):
        try:
            pose_camera = PoseStamped()
            pose_camera.header.frame_id = "camera_frame"
            pose_camera.pose.position.x = float(tvec[2][0]) # 相机Z轴对应物体的前后距离
            pose_camera.pose.position.y = float(-tvec[0][0]) # 相机Y轴对应物体的左右距离
            pose_camera.pose.position.z = float(-tvec[1][0]) # 相机X轴对应物体的上下距离
            pose_camera.pose.orientation.w = 1.0

            transform = self.tf_buffer.lookup_transform("map", "camera_frame", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            pose_map = tf2_geometry_msgs.do_transform_pose(pose_camera.pose, transform)
            return pose_map

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=5)

    def image_callback(self, msg):
        start_time = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                corner = corners[i]

                # 过滤小面积标记
                area = cv2.contourArea(corner)
                if area < self.min_marker_area:
                    continue

                obj_points = np.array([
                    [-self.marker_size/2, self.marker_size/2, 0],
                    [self.marker_size/2, self.marker_size/2, 0],
                    [self.marker_size/2, -self.marker_size/2, 0],
                    [-self.marker_size/2, -self.marker_size/2, 0]
                ], dtype=np.float64)

                success, rvec, tvec = cv2.solvePnP(obj_points, corner[0], self.camera_matrix, self.dist_coeffs)

                if not success:
                    continue
                
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

                if marker_id not in self.detection_history:
                    self.detection_history[marker_id] = []
                self.detection_history[marker_id].append((rvec, tvec))

                if len(self.detection_history[marker_id]) > self.history_size:
                    self.detection_history[marker_id].pop(0)
                
                avg_tvec = np.mean([t for _, t in self.detection_history[marker_id]], axis=0)
                distance = float(np.linalg.norm(avg_tvec))

                if self.mode == "inspect" and str(marker_id) in self.baseline:
                    item_name = self.baseline[str(marker_id)]['item_name']
                else:
                    item_name = f"Item_{marker_id}"

                cx = int(corner[0][:, 0].mean())
                cy = int(corner[0][:, 1].mean())
                cv2.putText(frame, f"ID:{marker_id} {item_name}",
                            (cx - 60, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Dist:{distance:.2f}m",
                            (cx - 60, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                pose_map = self.transform_to_map(tvec, rvec)
                map_x = None
                map_y = None
                map_z = None
                if pose_map:
                    map_x = pose_map.position.x
                    map_y = pose_map.position.y
                    map_z = pose_map.position.z
                    cv2.putText(frame, f"Map:({map_x:.2f},{map_y:.2f},z:{map_z:.2f})",
                                (cx - 60, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

                if self.is_scanning:
                    self.log_detection(marker_id, item_name, tvec, distance, map_x, map_y, map_z)
                self.get_logger().info(f"Detected {item_name} (ID: {marker_id}) at distance {distance:.2f}m")

        mode_text = f"Mode: {self.mode.upper()}"
        if self.mode == "register":
            mode_color = (0, 255, 255)
        else:
            mode_color = (0, 255, 0)
        cv2.putText(frame, mode_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        cv2.putText(frame, f"Items detected: {len(self.inventory)}", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_img.publish(img_msg)

        self.publish_markers()
        
        cv2.imshow("ArUco Detection", frame)
        cv2.waitKey(1)

        end_time = time.time()
        self.get_logger().info(f"Processing time: {(end_time - start_time)*1000:.2f} ms")

    def log_detection(self, marker_id, item_name, tvec, distance, map_x, map_y, map_z):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if distance > 1.5:
            status = "Out of range"
        elif self.mode == "inspect" and str(marker_id) in self.baseline:
            base = self.baseline[str(marker_id)]
            if map_x is not None and base.get('map_x') is not None:
                displacement = math.sqrt(
                    (map_x - base['map_x'])**2 +
                    (map_y - base['map_y'])**2 +
                    (map_z - base.get('map_z', map_z))**2
                )
                status = "Displaced" if displacement > 0.15 else "Normal"
            else:
                status = "Out of range"  # seen but TF failed, can't compute displacement
        else:
            status = "Normal"

        self.inventory[int(marker_id)] = {
            "item_name": item_name,
            "marker_id": int(marker_id),
            "cam_x": float(tvec[0][0]),
            "cam_y": float(tvec[1][0]),
            "cam_z": float(tvec[2][0]),
            "distance": float(distance),
            "map_x": map_x,
            "map_y": map_y,
            "map_z": map_z,
            "status": status,
            "last_seen": now
        }
    
    def publish_markers(self):
        marker_array = MarkerArray()
        for marker_id, data in self.inventory.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "inventory"
            marker.id = int(marker_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            if data["map_x"] is None:
                continue  # no valid position to display
            marker.pose.position.x = data["map_x"]
            marker.pose.position.y = data["map_y"]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            if data["status"] == "Normal":
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif data["status"] == "Displaced":
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            elif data["status"] == "Missing":
                marker.color.r = 0.5
                marker.color.g = 0.0
                marker.color.b = 0.5
            else:  # Out of range
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

            marker.color.a = 1.0
            marker.lifetime.sec = 30
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "inventory_labels"
            text_marker.id = int(marker_id) + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = data["map_x"]
            text_marker.pose.position.y = data["map_y"]
            text_marker.pose.position.z = 0.3
            text_marker.scale.z = 0.08
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"{data['item_name']} [{data['status']}]"
            text_marker.lifetime.sec = 30
            marker_array.markers.append(text_marker)

        self.pub_markers.publish(marker_array)
    
    def save_baseline(self):
        with open(self.baseline_path, 'w') as f:
            json.dump(self.inventory, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f"Baseline saved to {self.baseline_path}")

    def save_inventory_report(self):
        report_path = os.path.join(
            self.log_dir,
            f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(self.inventory, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f"Inventory report saved to {report_path}")

    def destroy_node(self):
        if self.mode == "register":
            self.save_baseline()
        self.save_inventory_report()
        cv2.destroyAllWindows()
        super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
