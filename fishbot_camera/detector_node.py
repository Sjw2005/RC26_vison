#!/home/chairman/venv/bin/python3.10
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO
# 导入消息和服务
from geometry_msgs.msg import Point
from std_srvs.srv import SetBool

class RedSquareDetector(Node):
    def __init__(self):
        super().__init__('detector')
        
        # 1. 坐标发布者
        self.point_publisher_ = self.create_publisher(Point, 'pos_sub', 10)
        
        # 2. 服务客户端
        self.cli = self.create_client(SetBool, 'arm_ctr_srv')
        
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
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        self.get_logger().info("系统就绪: 图像识别运行中，等待服务端许可(状态0)...")

    def convert_coords(self, original_xyz):
        # 坐标系转换 (根据实际情况调整符号)
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
            
            if sent_value == False: # 发送的是握手信号 0
                if response.success:
                    self.get_logger().info(f">>> 握手成功! 服务端已就绪，允许发送坐标 [进入状态1]")
                    self.app_state = 1
                else:
                    # 服务端忙，保持状态0，继续识别但不发数据
                    pass 
            
            elif sent_value == True: # 发送的是触发信号 1
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

            # ==========================================
            #   核心修改：识别逻辑移到最外层，始终执行
            # ==========================================
            detected_target = None  # 存放 (x, y, z)
            detected_approx = None  # 存放轮廓用于画图

            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 120, 70]); upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])
            mask = cv2.inRange(hsv_image, lower_red1, upper_red1) + cv2.inRange(hsv_image, lower_red2, upper_red2)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    epsilon = 0.04 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        if 0.9 <= aspect_ratio <= 1.1:
                            M = cv2.moments(cnt)
                            if M['m00'] != 0:
                                cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
                                dist = depth_frame.get_distance(cx, cy)
                                if dist > 0:
                                    orig_xyz = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], dist)
                                    detected_target = self.convert_coords(orig_xyz)
                                    detected_approx = approx
                                    # 始终在画面上绘制（无论状态如何）
                                    cv2.drawContours(color_image, [approx], 0, (0, 255, 0), 2)
                                    cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)
                                    break # 找到一个就退出循环

            # ==========================================
            #        状态机：决定发不发送数据
            # ==========================================
            current_time = time.time()

            # --- 状态 0: 只是等待 ---
            # 识别到了目标 (detected_target非空) 也不发，因为服务端没允许
            if self.app_state == 0:
                if (current_time - self.last_ping_time > 2.0) and (not self.is_waiting_response):
                    # 可以在日志里看到识别状态，但不通过 Topic 发送
                    # msg = "检测到目标" if detected_target else "未检测到"
                    # self.get_logger().info(f"状态0: 发送握手(0)... 当前视觉状态: {msg}")
                    self.send_service_request(False) 
                    self.last_ping_time = current_time
            
            # --- 状态 1: 允许发送 ---
            elif self.app_state == 1:
                # 只有在这里，且识别到了目标，才真正发布坐标
                if detected_target is not None and not self.is_waiting_response:
                    # 1. 发布坐标 (Topic)
                    point_msg = Point()
                    point_msg.x = round(float(detected_target[0]), 3)
                    point_msg.y = round(float(detected_target[1]), 3)
                    point_msg.z = round(float(detected_target[2]), 3)
                    self.point_publisher_.publish(point_msg)
                    
                    # 2. 发送信号 1 (Service)
                    self.get_logger().info(f"{point_msg.x}, {point_msg.y},{point_msg.z}")
                    self.send_service_request(True)
            
            # 显示画面
            cv2.imshow('RealSense Client', color_image)
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
    node = RedSquareDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()