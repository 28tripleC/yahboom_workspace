import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import time
import json
import os
from datetime import datetime
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

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

        calib_file = os.path.expanduser('~/camera_calibration/calib_data.npz')
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

        self.marker_size = 0.08

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
        else:
            self.get_logger().info("No baseline found, starting in registration mode")

        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
                
                avg_tvec = np.mean(self.detection_history[marker_id], axis=0)
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
                map_x = 0.0
                map_y = 0.0
                if pose_map:
                    map_x = pose_map.position.x
                    map_y = pose_map.position.y
                    cv2.putText(frame, f"Map:({map_x:.2f},{map_y:.2f})",
                                (cx - 60, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

                self.log_detection(marker_id, item_name, tvec, distance, map_x, map_y)
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

    def log_detection(self, marker_id, item_name, tvec, distance, map_x, map_y):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if distance > 1.5:
            status = "Out of range"
        elif abs(tvec[1][0]) > 0.3:
            status = "Displaced"
        else:
            status = "Normal"

        self.inventory[int(marker_id)] = {
            "item_name": item_name,
            "marker_id": int(marker_id),
            "cam_x": float(tvec[0][0]),
            "cam_y": float(tvec[1][0]),
            "cam_z": float(tvec[2][0]),
            "distance": float(distance),
            "map_x": float(map_x),
            "map_y": float(map_y),
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
            else:
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
    
    def save_inventory_report(self):
        report_path = os.path.join(
            self.log_dir,
            f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(self.inventory, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f"Inventory report saved to {report_path}")

    def destroy_node(self):
        self.save_inventory_report()
        cv2.destroyAllWindows()
        super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
