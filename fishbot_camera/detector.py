#!/home/chairman/venv/bin/python3.10
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO
from geometry_msgs.msg import Point
from std_srvs.srv import SetBool


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('detector')

        self.model_path = "kfs.pt"
        self.target_labels = ['kfs', 'block']

        self.point_publisher_ = self.create_publisher(Point, 'pos_sub', 10)
        self.cli = self.create_client(SetBool, 'arm_ctr_srv')

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.get_logger().error(f"YOLO 模型加载失败: {e}")
            raise e

        # --- 状态变量 ---
        # 0 = 握手/等待阶段, 1 = 允许发送坐标阶段
        self.app_state = 0
        self.last_ping_time = 0.0
        self.is_waiting_response = False

        # --- 初始化 RealSense (带重试) ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.get_logger().info("正在寻找 RealSense 摄像头...")
        while True:
            try:
                self.profile = self.pipeline.start(self.config)
                self.get_logger().info(">>> 摄像头启动成功！")
                break
            except Exception as e:
                self.get_logger().warn(f"未检测到摄像头，1秒后重试... ({e})")
                time.sleep(1.0)

        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)

        # 定时器 30Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("系统就绪: YOLO识别运行中，等待服务端许可(状态0)...")

    # 机械臂坐标转换
    def convert_coords(self, original_xyz):
        return (original_xyz[0], original_xyz[2], -original_xyz[1])

    # --- 发送请求 ---
    def send_service_request(self, value):
        if not self.cli.service_is_ready():
            if int(time.time()) % 2 == 0:
                self.get_logger().warn("等待 'arm_ctr_srv' 服务上线...")
            return

        req = SetBool.Request()
        req.data = value
        future = self.cli.call_async(req)
        future.add_done_callback(lambda future: self.service_response_callback(future, value))
        self.is_waiting_response = True

    # --- 回调处理 ---
    def service_response_callback(self, future, sent_value):
        try:
            response = future.result()
            self.is_waiting_response = False

            if sent_value == False:  # 发送的是握手信号 0
                if response.success:
                    self.get_logger().info(f">>> 握手成功! 服务端已就绪，允许发送坐标 [进入状态1]")
                    self.app_state = 1
                else:
                    pass

            elif sent_value == True:  # 发送的是触发信号 1
                self.get_logger().info(">>> 坐标已发送，重置回 [状态0]")
                self.app_state = 0
                self.last_ping_time = time.time()

        except Exception as e:
            self.get_logger().error(f"服务调用异常: {e}")
            self.is_waiting_response = False

    def timer_callback(self):
        try:
            # 1. 获取图像
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: return

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # 获取深度数组

            # 检测坐标
            detected_target = None
            results = self.model.predict(source=color_image, save=False, verbose=False, conf=0.5)

            # 解析
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    # [Label 过滤]
                    if cls_name not in self.target_labels:
                        continue

                    # 获取坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # ROI 深度测距 (5x5 区域)
                    range_size = 2
                    y_min = max(0, cy - range_size)
                    y_max = min(720, cy + range_size + 1)
                    x_min = max(0, cx - range_size)
                    x_max = min(1280, cx + range_size + 1)

                    depth_roi = depth_image[y_min:y_max, x_min:x_max]
                    valid_depths = depth_roi[depth_roi > 0]

                    if len(valid_depths) > 0:
                        dist_m = np.median(valid_depths) / 1000.0

                        # 转换: 像素 -> 相机坐标
                        camera_xyz = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], dist_m)
                        # 转换: 相机坐标 -> 机械臂坐标
                        detected_target = self.convert_coords(camera_xyz)

                        # 标注
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                        label_text = f"{cls_name} {dist_m:.2f}m"
                        cv2.putText(color_image, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        break  # 找到第一个符合标签的目标即停止

                if detected_target is not None:
                    break

            # ==========================================
            #        状态机：决定发不发送数据
            # ==========================================
            current_time = time.time()

            # --- 状态 0: 等待握手 ---
            if self.app_state == 0:
                # 即使检测到了 detected_target，也不发送，只请求握手
                if (current_time - self.last_ping_time > 2.0) and (not self.is_waiting_response):
                    self.send_service_request(False)
                    self.last_ping_time = current_time

            # --- 状态 1: 允许发送 ---
            elif self.app_state == 1:
                # 只有在这里，且检测到了目标，才发布坐标
                if detected_target is not None and not self.is_waiting_response:
                    # 1. 构造 Point 消息
                    point_msg = Point()
                    point_msg.x = round(float(detected_target[0]), 3)
                    point_msg.y = round(float(detected_target[1]), 3)
                    point_msg.z = round(float(detected_target[2]), 3)

                    # 2. 发布 Topic
                    self.point_publisher_.publish(point_msg)

                    # 3. 发送 Service 信号 1 (通知已发送)
                    self.get_logger().info(
                        f">>> 发布坐标: X={point_msg.x}, Y={point_msg.y}, Z={point_msg.z} | Label={cls_name}")
                    self.send_service_request(True)

            # 显示画面
            cv2.imshow('YOLO RealSense', color_image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"处理循环出错: {e}")

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()