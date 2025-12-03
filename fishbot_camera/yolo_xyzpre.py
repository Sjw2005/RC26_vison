import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 模型路径
MODEL_PATH = r"kfs.pt"
model = YOLO(MODEL_PATH)

# RealSense 初始化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("正在启动 RealSense 相机...")
pipeline.start(config)
align = rs.align(rs.stream.color)  # 深度对齐到彩色

try:
    while True:
        # ---  获取图像帧 ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 获取相机内参
        depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics

        # 转为 Numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # ==================== [DEBUG 区域] 开始 ====================
        # 说明：如果你怀疑深度数据有问题，可以取消注释下面 3 行代码
        # 这会弹出一个额外的 'Depth Heatmap' 窗口，显示彩色的深度热力图
        # ---------------------------------------------------------
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('Depth Heatmap', depth_colormap)
        # ---------------------------------------------------------
        # ==================== [DEBUG 区域] 结束 ====================

        im_array = color_image.copy()

        # ---  YOLO 推理 ---
        results = model.predict(source=color_image, save=False, verbose=False, conf=0.2)

        for result in results:
            for box in result.boxes:
                # 获取基本坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ux, uy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cls_name = model.names[int(box.cls[0])]

                # ---  ROI 区域测距 (取中心 5x5 区域中位数) ---
                range_size = 2
                y_min = max(0, uy - range_size)
                y_max = min(480, uy + range_size + 1)
                x_min = max(0, ux - range_size)
                x_max = min(640, ux + range_size + 1)

                depth_roi = depth_image[y_min:y_max, x_min:x_max]
                valid_depths = depth_roi[depth_roi > 0]

                if len(valid_depths) == 0:
                    continue

                # 计算中位数距离 (mm -> m)
                dis_m = np.median(valid_depths) / 1000.0

                # ---  坐标转换 (像素 -> 相机坐标系) ---
                camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intri, (ux, uy), dis_m)

                # 转换为 mm
                rx = int(camera_xyz[0] * 1000)
                ry = int(camera_xyz[1] * 1000)
                rz = int(camera_xyz[2] * 1000)

                coord_str = f"(x:{rx}, y:{ry}, z:{rz})"

                print(f"目标: {cls_name} | 坐标: {coord_str}")

                cv2.rectangle(im_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(im_array, (ux, uy), 4, (0, 0, 255), -1)
                cv2.putText(im_array, cls_name, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(im_array, coord_str, (x1, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 100, 255), 2)

        cv2.imshow('YOLOv11 + RealSense', im_array)


finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("程序已结束")