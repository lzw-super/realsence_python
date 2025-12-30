## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2025 RealSense, Inc. All Rights Reserved.
#####################################################################################################
#                                                                                                  ##
#    使用软件设备将预捕获的深度图像对齐到彩色图像                                                   ##
#                                                                                                  ##
##  目的                                                                                           ##
##    本示例首先从RealSense相机捕获深度和彩色图像，然后演示如何使用软件设备对预捕获的图像进行深度到彩色的对齐 ##
##                                                                                                 ##
##  步骤：                                                                                         ##
##    1) 使用RealSense相机流式传输深度640x480@30fps和彩色1280x720@30fps                           ##
##    2) 捕获相机的深度和彩色内参以及外参                                                         ##
##    3) 捕获深度和彩色图像并保存为npy格式文件                                                     ##
##    4) 使用保存的内参、外参、深度和彩色图像构建软件设备                                          ##
##    5) 将预捕获的深度图像对齐到彩色图像                                                         ##
##                                                                                                 ##
#####################################################################################################

# 导入日志模块，用于错误记录
import logging
# 导入OpenCV库，用于图像显示
import cv2
# 导入RealSense库
import pyrealsense2 as rs
# 导入NumPy库，用于数组操作
import numpy as np
# 导入操作系统模块，用于文件路径操作
import os
# 导入时间模块，用于暂停功能
import time

fps = 30                  # 帧率
tv = 1000.0 / fps         # 帧间时间间隔（毫秒）

max_num_frames  = 100      # 要捕获到npy文件并使用软件设备处理的最大帧集数量

depth_file_name = "depth"  # 深度文件名前缀：depth_file_name + str(i) + ".npy"
color_file_name = "color"  # 彩色文件名前缀：color_file_name + str(i) + ".npy"

# 从相机获取的内参和外参
camera_depth_intrinsics          = rs.intrinsics()  # 相机深度内参（焦距、主点等）
camera_color_intrinsics          = rs.intrinsics()  # 相机彩色内参（焦距、主点等）
camera_depth_to_color_extrinsics = rs.extrinsics()  # 相机深度到彩色的外参（旋转矩阵、平移向量）

WIDTH = 1280
HEIGHT = 720
FPS = 30
######################## 第一部分开始 - 从实时设备捕获图像 #######################################
# 在连接的RealSense相机上流式传输深度和彩色，并将深度和彩色帧保存为npy格式的文件
try:
    # 创建上下文对象，该对象拥有所有连接的RealSense设备的句柄
    ctx = rs.context()
    # 查询所有连接的设备
    devs = list(ctx.query_devices())
    
    # 检查是否检测到设备
    if len(devs) > 0:
        print("Devices: {}".format(devs))
    else:
        print("No camera detected. Please connect a realsense camera and try again.")
        exit(0)
    
    # 创建管道对象
    pipeline = rs.pipeline()

    # 配置流
    config = rs.config()
    # config.enable_stream(rs.stream.depth)
    # config.enable_stream(rs.stream.color)
    # 启用深度流：1280x720分辨率，Z16格式，30帧每秒
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    # 启用彩色流：1280x720分辨率，BGR8格式，30帧每秒
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 开始流式传输，返回配置文件
    cfg = pipeline.start(config)
    
    # 获取深度比例（将深度值转换为米的系数）
    depth_sensor = cfg.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 获取内参
    # 获取深度流配置文件
    camera_depth_profile = cfg.get_stream(rs.stream.depth)
    # 转换为视频流配置文件并获取内参（包含宽度、高度、焦距、主点、畸变系数等）
    camera_depth_intrinsics = camera_depth_profile.as_video_stream_profile().get_intrinsics()
    
    # 获取彩色流配置文件
    camera_color_profile = cfg.get_stream(rs.stream.color)
    # 转换为视频流配置文件并获取内参
    camera_color_intrinsics = camera_color_profile.as_video_stream_profile().get_intrinsics()
    
    # 获取深度到彩色的外参（包含旋转矩阵和平移向量）
    camera_depth_to_color_extrinsics = camera_depth_profile.get_extrinsics_to(camera_color_profile)


    print("camera depth intrinsic:", camera_depth_intrinsics)
    print("camera color intrinsic:", camera_color_intrinsics)
    print("camera depth to color extrinsic:", camera_depth_to_color_extrinsics)

    print("streaming attached camera and save depth and color frames into files in npy format ...")

    i = 0
    # 循环捕获最多max_num_frames帧
    while i < max_num_frames:
        # 等待直到设备上有新的连贯帧集可用
        frames = pipeline.wait_for_frames()
        # 获取深度帧
        depth = frames.get_depth_frame()
        # 获取彩色帧
        color = frames.get_color_frame()

        # 验证两个帧是否有效
        if not depth or not color:
            continue
        
        # 将图像转换为NumPy数组
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())
        # 保存为npy格式的文件
        depth_file = depth_file_name + str(i) + ".npy"
        color_file = color_file_name + str(i) + ".npy"
        print("saving frame set ", i, depth_file, color_file)
        
        # 保存深度图像到npy文件
        with open(depth_file, 'wb') as f1:
            np.save(f1,depth_image)
        
        # 保存彩色图像到npy文件
        with open(color_file, 'wb') as f2:
            np.save(f2,color_image)

        # 下一帧
        i = i +1

except Exception as e:
    # 捕获并记录异常
    logging.error("An error occurred: %s", e, exc_info=True)
    exit(1)

######################## 第一部分结束 - 从实时设备捕获图像 #######################################



######################## 第二部分开始 - 在软件设备中对齐深度到彩色 #############################
# 使用软件设备将上述预捕获的图像进行深度到彩色的对齐

# 创建软件设备（虚拟设备，用于处理预捕获的数据）
sdev = rs.software_device()

# 创建软件深度传感器
depth_sensor: rs.software_sensor = sdev.add_sensor("Depth")

# 深度内参
depth_intrinsics = rs.intrinsics()

# 设置深度内参的宽度和高度
depth_intrinsics.width  = camera_depth_intrinsics.width
depth_intrinsics.height = camera_depth_intrinsics.height

# 设置深度内参的主点坐标（principal point）
depth_intrinsics.ppx = camera_depth_intrinsics.ppx
depth_intrinsics.ppy = camera_depth_intrinsics.ppy

# 设置深度内参的焦距（focal length）
depth_intrinsics.fx = camera_depth_intrinsics.fx
depth_intrinsics.fy = camera_depth_intrinsics.fy

# 设置深度内参的畸变系数（通常为[0.0, 0.0, 0.0, 0.0, 0.0]）
depth_intrinsics.coeffs = camera_depth_intrinsics.coeffs
# 设置深度内参的畸变模型（如Brown-Conrady模型）
depth_intrinsics.model = camera_depth_intrinsics.model

# 深度流配置
depth_stream = rs.video_stream()
# 设置流类型为深度
depth_stream.type = rs.stream.depth
# 设置流宽度
depth_stream.width = depth_intrinsics.width
# 设置流高度
depth_stream.height = depth_intrinsics.height
# 设置帧率
depth_stream.fps = fps
# 设置每像素字节数（Z16格式为2字节）
depth_stream.bpp = 2
# 设置格式为Z16
depth_stream.fmt = rs.format.z16
# 设置内参
depth_stream.intrinsics = depth_intrinsics
# 设置流索引
depth_stream.index = 0
# 设置唯一标识符
depth_stream.uid = 1

# 将深度流添加到深度传感器
depth_profile = depth_sensor.add_video_stream(depth_stream)

# 创建软件彩色传感器
color_sensor: rs.software_sensor = sdev.add_sensor("Color")

# 彩色内参
color_intrinsics = rs.intrinsics()
# 设置彩色内参的宽度和高度
color_intrinsics.width = camera_color_intrinsics.width
color_intrinsics.height = camera_color_intrinsics.height

# 设置彩色内参的主点坐标
color_intrinsics.ppx = camera_color_intrinsics.ppx
color_intrinsics.ppy = camera_color_intrinsics.ppy

# 设置彩色内参的焦距
color_intrinsics.fx = camera_color_intrinsics.fx
color_intrinsics.fy = camera_color_intrinsics.fy

# 设置彩色内参的畸变系数
color_intrinsics.coeffs = camera_color_intrinsics.coeffs
# 设置彩色内参的畸变模型
color_intrinsics.model = camera_color_intrinsics.model

# 彩色流配置
color_stream = rs.video_stream()
# 设置流类型为彩色
color_stream.type = rs.stream.color
# 设置流宽度
color_stream.width = color_intrinsics.width
# 设置流高度
color_stream.height = color_intrinsics.height
# 设置帧率
color_stream.fps = fps
# 设置每像素字节数（RGB8格式为3字节）
color_stream.bpp = 3
# 设置格式为RGB8
color_stream.fmt = rs.format.rgb8
# 设置内参
color_stream.intrinsics = color_intrinsics
# 设置流索引
color_stream.index = 0
# 设置唯一标识符
color_stream.uid = 2

# 将彩色流添加到彩色传感器
color_profile = color_sensor.add_video_stream(color_stream)

# 深度到彩色的外参（自动计算：depth_profile.get_extrinsics_to(other_profile)）
depth_to_color_extrinsics = rs.extrinsics()
# 设置旋转矩阵（从深度坐标系到彩色坐标系的旋转）
depth_to_color_extrinsics.rotation = camera_depth_to_color_extrinsics.rotation
# 设置平移向量（从深度坐标系到彩色坐标系的平移）
depth_to_color_extrinsics.translation = camera_depth_to_color_extrinsics.translation
# 注册深度到彩色的外参关系
depth_profile.register_extrinsics_to(color_profile, depth_to_color_extrinsics)

# 启动软件传感器
depth_sensor.open(depth_profile)
color_sensor.open(color_profile)

# 同步深度和彩色流的帧
camera_syncer = rs.syncer()
# 启动深度传感器，将帧发送到同步器
depth_sensor.start(camera_syncer)
# 启动彩色传感器，将帧发送到同步器
color_sensor.start(camera_syncer)

# 创建深度对齐对象
# rs.align允许我们将深度帧对齐到其他帧
# "align_to"参数指定我们要将深度帧对齐到的流类型
# 将深度帧对齐到彩色帧
align_to = rs.stream.color
align = rs.align(align_to)

# 创建深度着色器，用于深度图像的可视化
colorizer = rs.colorizer()

paused = False

# 循环处理预捕获的帧
for i in range(0, max_num_frames):
    print("\nframe set:", i)
    
    # 预捕获的深度和彩色图像文件（npy格式）
    df = depth_file_name + str(i) + ".npy"
    cf = color_file_name + str(i) + ".npy"

    # 检查文件是否存在
    if (not os.path.exists(cf)) or (not os.path.exists(df)): continue

    # 从预捕获的npy文件加载深度帧
    print('loading depth frame ', df)
    depth_npy = np.load(df, mmap_mode='r')

    # 创建软件深度帧
    depth_swframe = rs.software_video_frame()
    # 设置步长（每行字节数）
    depth_swframe.stride = depth_stream.width * depth_stream.bpp
    # 设置每像素字节数
    depth_swframe.bpp = depth_stream.bpp
    # 设置时间戳（基于帧号计算）
    depth_swframe.timestamp = i * tv
    # 设置像素数据
    depth_swframe.pixels = depth_npy
    # 设置时间戳域为硬件时钟
    depth_swframe.domain = rs.timestamp_domain.hardware_clock
    # 设置帧号
    depth_swframe.frame_number = i
    # 设置流配置文件
    depth_swframe.profile = depth_profile.as_video_stream_profile()
    # 设置深度单位
    depth_swframe.depth_units = depth_scale
    # 将软件帧发送到深度传感器
    depth_sensor.on_video_frame(depth_swframe)

    # 从预捕获的npy文件加载彩色帧
    print('loading color frame ', cf)
    color_npy = np.load(cf, mmap_mode='r')

    # 创建软件彩色帧
    color_swframe = rs.software_video_frame()
    # 设置步长
    color_swframe.stride = color_stream.width * color_stream.bpp
    # 设置每像素字节数
    color_swframe.bpp = color_stream.bpp
    # 设置时间戳
    color_swframe.timestamp = i * tv
    # 设置像素数据
    color_swframe.pixels = color_npy
    # 设置时间戳域
    color_swframe.domain = rs.timestamp_domain.hardware_clock
    # 设置帧号
    color_swframe.frame_number = i
    # 设置流配置文件
    color_swframe.profile = color_profile.as_video_stream_profile()
    # 将软件帧发送到彩色传感器
    color_sensor.on_video_frame(color_swframe)
    
    # 同步深度和彩色，接收为帧集
    frames = camera_syncer.wait_for_frames()
    print("frame set:", frames.size(), " ", frames)

    # 获取未对齐的深度帧
    unaligned_depth_frame = frames.get_depth_frame()
    if not unaligned_depth_frame: continue

    # 将深度帧对齐到彩色帧
    aligned_frames = align.process(frames)

    # 获取对齐后的深度帧
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # 获取彩色帧
    color_frame = aligned_frames.get_color_frame()

    # 验证帧是否有效
    if (not aligned_depth_frame) or (not color_frame): continue

    # 使用着色器将深度帧转换为彩色图像以便可视化
    aligned_depth_frame = colorizer.colorize(aligned_depth_frame)
    # 将对齐后的深度帧转换为NumPy数组
    npy_aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

    # 将彩色帧转换为NumPy数组
    npy_color_image = np.asanyarray(color_frame.get_data())

    # 渲染对齐后的图像：
    # 深度对齐到彩色
    # 左侧：对齐后的深度图像
    # 右侧：彩色图像
    images = np.hstack((npy_aligned_depth_image, npy_color_image))
    # 创建窗口，允许调整大小
    cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
    # 显示拼接后的图像
    cv2.imshow('Align Example', images)
    # 等待1毫秒的键盘输入
    key = cv2.waitKey(1)

    # 渲染原始未对齐的深度作为参考（注释掉的代码）
    # colorized_unaligned_depth_frame = colorizer.colorize(unaligned_depth_frame)
    # npy_unaligned_depth_image = np.asanyarray(colorized_unaligned_depth_frame.get_data())
    # cv2.imshow("Unaligned Depth", npy_unaligned_depth_image)
    
    # 按ENTER或SPACEBAR键暂停图像窗口5秒

    if key == 13 or key == 32: paused = not paused
        
    if paused:
        print("Paused for 5 seconds ...", i, ", press ENTER or SPACEBAR key anytime for additional pauses.")
        time.sleep(5)
        paused = not paused

# 第二部分结束 - 使用软件设备将预捕获的图像进行深度到彩色的对齐
######################## 第二部分结束 - 在软件设备中对齐深度到彩色 #############################
    
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()