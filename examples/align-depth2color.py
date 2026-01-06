## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 RealSense, Inc. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# 首先导入RealSense库
import pyrealsense2 as rs
# 导入NumPy库，用于数组操作
import numpy as np
# 导入OpenCV库，用于图像渲染和显示
import cv2

# 创建一个管道对象，用于管理数据流
pipeline = rs.pipeline()

# 创建配置对象，用于配置管道以流式传输不同分辨率的彩色和深度流
config = rs.config()

# 获取设备产品线，以便设置支持的分辨率
# pipeline_wrapper是管道的包装器，用于解析配置
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# 解析配置以获取管道配置文件
pipeline_profile = config.resolve(pipeline_wrapper)
# 从配置文件中获取设备对象
device = pipeline_profile.get_device()
# 获取设备产品线信息（如D400系列、L500系列等）
device_product_line = str(device.get_info(rs.camera_info.product_line))

# 检查设备是否支持RGB相机
found_rgb = False
# 遍历设备的所有传感器
for s in device.sensors:
    # 如果找到名为'RGB Camera'的传感器，则标记为找到
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
# 如果没有找到RGB相机，则退出程序
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 启用深度流：640x480分辨率，Z16格式（16位深度值），30帧每秒
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# 启用彩色流：640x480分辨率，BGR8格式（8位BGR彩色），30帧每秒
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始流式传输，返回管道配置文件
profile = pipeline.start(config)

# 获取深度传感器的深度比例（将深度值转换为米的系数）
# 例如：如果depth_scale=0.001，则深度值1000代表1米
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# 我们将移除距离超过clipping_distance_in_meters米的物体背景
clipping_distance_in_meters = 1  # 1米
# 将米转换为深度值（深度值 = 距离(米) / 深度比例）
clipping_distance = clipping_distance_in_meters / depth_scale

# 创建对齐对象
# rs.align允许我们将深度帧对齐到其他帧
# "align_to"参数指定我们要将深度帧对齐到的流类型
align_to = rs.stream.color
align = rs.align(align_to)

# 流式传输循环
try:
    while True:
        # 等待获取下一帧数据（包含彩色和深度帧）
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame()返回的是640x360的深度图像（未对齐）

        # 将深度帧对齐到彩色帧
        # 对齐后，深度帧的分辨率和视角将与彩色帧一致
        aligned_frames = align.process(frames)

        # 获取对齐后的帧
        # aligned_depth_frame是对齐后的深度图像，分辨率为640x480
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 获取彩色帧
        color_frame = aligned_frames.get_color_frame()

        # 验证两个帧是否有效
        if not aligned_depth_frame or not color_frame:
            continue

        # 将深度帧数据转换为NumPy数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # 将彩色帧数据转换为NumPy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 移除背景 - 将距离超过clipping_distance的像素设置为灰色
        grey_color = 153  # 灰色值
        # 将单通道深度图像扩展为三通道，以便与彩色图像进行比较
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
        # 使用np.where进行条件判断：
        # 如果深度值大于裁剪距离或小于等于0（无效深度），则设为灰色
        # 否则保留彩色图像的像素值
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # 渲染图像：
        #   左侧显示：对齐到彩色的深度图像（背景已移除）
        #   右侧显示：原始深度图像（彩色映射）
        # 将深度图像转换为8位无符号整数，并应用缩放因子alpha=0.03
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # 将两幅图像水平拼接在一起
        images = np.hstack((bg_removed, depth_colormap))

        # 创建窗口，允许调整大小
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # 显示拼接后的图像
        cv2.imshow('Align Example', images)
        # 等待1毫秒的键盘输入
        key = cv2.waitKey(1)
        # 按ESC键或'q'键关闭图像窗口并退出循环
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # 停止管道，释放资源
    pipeline.stop()
