## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 RealSense, Inc. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# 首先导入RealSense库
import pyrealsense2 as rs


# 声明点云对象，用于计算点云和纹理映射
# pointcloud是一个处理块，可以将深度帧转换为3D点云
pc = rs.pointcloud()
# 我们希望points对象是持久的，这样当帧丢失时可以显示最后一个点云
# points对象用于存储计算出的点云数据
points = rs.points()

# 声明RealSense管道，封装实际的设备和传感器
# pipeline用于管理数据流和设备配置
pipe = rs.pipeline()
# 创建配置对象
config = rs.config()
# 启用深度流（使用默认分辨率和格式）
config.enable_stream(rs.stream.depth)

# 使用选择的配置开始流式传输
pipe.start(config)

# 我们将使用着色器为我们的PLY生成纹理
# （或者，纹理可以从彩色或红外流获得）
# colorizer将深度数据转换为彩色图像，便于可视化
colorizer = rs.colorizer()

try:
    # 等待相机获取下一帧数据
    frames = pipe.wait_for_frames()
    # 使用着色器处理帧，为深度数据添加颜色纹理
    colorized = colorizer.process(frames)

    # 创建save_to_ply对象，用于将点云保存为PLY文件
    # "1.ply"是输出文件名
    ply = rs.save_to_ply("1.ply")

    # 设置选项为所需的值
    # 在这个示例中，我们将生成一个带有法线的文本PLY文件（网格默认已创建）
    # option_ply_binary: False表示使用ASCII文本格式（可读），True表示二进制格式（文件更小）
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    # option_ply_normals: True表示包含法线信息，用于表面渲染
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving to 1.ply...")
    # 将处理块应用于包含深度帧和纹理的帧集
    # 这会将深度数据转换为点云，并保存为PLY文件
    ply.process(colorized)
    print("Done")
finally:
    # 停止管道，释放资源
    pipe.stop()
