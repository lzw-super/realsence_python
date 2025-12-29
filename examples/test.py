import pyrealsense2 as rs
import numpy as np
import cv2

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

# 启用流：彩色 (640x480, 30fps)，深度 (640x480, 30fps)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

try:
    while True:
        # 等待一组帧（深度+彩色）
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 转为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 深度图可视化（归一化到 0~255）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # 水平拼接图像
        images = np.hstack((color_image, depth_colormap))

        # 显示
        cv2.imshow('RealSense D455', images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()