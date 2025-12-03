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
from typing import Optional, Tuple, List


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('detector')

        # --- 配置参数 ---
        self.model_path = "kfs.pt"
        self.target_labels = ['kfs', 'block']  # 目标白名单
        self.conf_thres = 0.5  # 置信度阈值
        self.roi_size = 2  # 深度测量半径 (2px -> 5x5区域)

        # --- ROS2 通讯 ---
        self.point_pub = self.create_publisher(Point, 'pos_sub', 10)
        self.cli = self.create_client(SetBool, 'arm_ctr_srv')

        # --- 状态机控制 ---
        # app_state: 0=等待握手, 1=允许发送
        self.app_state = 0
        self.last_ping_time = 0.0
        self.is_waiting_response = False

        # --- 初始化组件 ---
        self.init_yolo()
        self.init_realsense()

        # --- 启动循环 30Hz ---
        self.create_timer(1.0 / 30.0, self.run_loop)
        self.get_logger().info(">>> 系统就绪: 视觉检测与状态机运行中...")

    def init_yolo(self):
        """加载 YOLO 模型"""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.get_logger().error(f"YOLO 加载失败: {e}")
            raise e

    def init_realsense(self):
        """初始化 RealSense 相机 (带重试机制)"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.get_logger().info("正在连接摄像头...")
        while True:
            try:
                profile = self.pipeline.start(config)
                self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.align = rs.align(rs.stream.color)
                self.get_logger().info(">>> 摄像头连接成功")
                break
            except RuntimeError:
                self.get_logger().warn("未检测到摄像头，1秒后重试...")
                time.sleep(1.0)

    def to_arm_coords(self, camera_xyz: list) -> Tuple[float, float, float]:
        """
        [关键逻辑] 坐标系转换
        输入: 相机坐标系 (x:右, y:下, z:前)
        输出: 机械臂坐标系 (根据实际情况调整符号)
        """
        return (camera_xyz[0], camera_xyz[2], -camera_xyz[1])

    def get_roi_depth(self, depth_img, cx, cy) -> float:
        """获取 ROI 区域的中值深度 (米)"""
        y_min = max(0, cy - self.roi_size)
        y_max = min(depth_img.shape[0], cy + self.roi_size + 1)
        x_min = max(0, cx - self.roi_size)
        x_max = min(depth_img.shape[1], cx + self.roi_size + 1)

        depth_roi = depth_img[y_min:y_max, x_min:x_max]
        valid_pixels = depth_roi[depth_roi > 0]

        if len(valid_pixels) == 0: return 0.0
        return np.median(valid_pixels) / 1000.0

    def process_frame(self, color_img, depth_img) -> Tuple[Optional[Tuple], Optional[str]]:
        """执行推理与数据处理，返回 (坐标, 类别名)"""
        results = self.model.predict(source=color_img, save=False, verbose=False, conf=self.conf_thres)

        target_coords = None
        target_name = None

        for result in results:
            for box in result.boxes:
                cls_name = self.model.names[int(box.cls[0])]

                # 1. 标签过滤
                if cls_name not in self.target_labels: continue

                # 2. 提取几何信息
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 3. 深度解算
                dist_m = self.get_roi_depth(depth_img, cx, cy)
                if dist_m <= 0: continue

                # 4. 坐标解算 (2D -> 3D)
                cam_xyz = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], dist_m)
                target_coords = self.to_arm_coords(cam_xyz)
                target_name = cls_name

                # 5. 可视化绘制
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(color_img, f"{cls_name} {dist_m:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 找到第一个有效目标即返回
                return target_coords, target_name

        return None, None

    def manage_state_machine(self, detected_target: Optional[tuple], label_name: str):
        """
        [关键逻辑] 状态机管理
        状态0: 握手阶段，仅发送 False 请求许可
        状态1: 发送阶段，允许发布坐标并发送 True 确认
        """
        now = time.time()

        if self.app_state == 0:
            # 状态0: 每2秒尝试握手一次
            if (now - self.last_ping_time > 2.0) and not self.is_waiting_response:
                self.call_service(False)
                self.last_ping_time = now

        elif self.app_state == 1:
            # 状态1: 仅在检测到目标且空闲时发送
            if detected_target and not self.is_waiting_response:
                # 1. 发布 Topic 坐标
                msg = Point()
                msg.x, msg.y, msg.z = [round(float(v), 3) for v in detected_target]
                self.point_pub.publish(msg)

                # 2. 发送 Service 信号 (通知已抓取/发送)
                self.get_logger().info(f">>> 发送坐标: {msg.x}, {msg.y}, {msg.z} | Label: {label_name}")
                self.call_service(True)

    def call_service(self, data_val: bool):
        """异步调用服务"""
        if not self.cli.service_is_ready(): return

        req = SetBool.Request()
        req.data = data_val
        future = self.cli.call_async(req)
        future.add_done_callback(lambda f: self.on_service_response(f, data_val))
        self.is_waiting_response = True

    def on_service_response(self, future, sent_val):
        try:
            resp = future.result()
            self.is_waiting_response = False

            # 握手成功 (Sent False -> Got Success) -> 进入状态 1
            if not sent_val and resp.success:
                self.get_logger().info(">>> 握手成功: 进入发送模式 (State 1)")
                self.app_state = 1

            # 发送完成 (Sent True) -> 回到状态 0
            elif sent_val:
                self.get_logger().info(">>> 数据已确认: 重置状态 (State 0)")
                self.app_state = 0
                self.last_ping_time = time.time()  # 避免立即重发握手

        except Exception as e:
            self.get_logger().error(f"服务回调异常: {e}")
            self.is_waiting_response = False

    def run_loop(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            if not aligned.get_depth_frame() or not aligned.get_color_frame(): return

            # 转为 Numpy
            color_img = np.asanyarray(aligned.get_color_frame().get_data())
            depth_img = np.asanyarray(aligned.get_depth_frame().get_data())

            # 1. 视觉处理
            target_xyz, label_name = self.process_frame(color_img, depth_img)

            # 2. 状态机逻辑
            self.manage_state_machine(target_xyz, label_name)

            # 3. 显示
            cv2.imshow('YOLO Detector', color_img)
            if cv2.waitKey(1) == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"主循环异常: {e}")

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