import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# 配置参数
WIDTH = 1280
HEIGHT = 720
FPS = 30
MAX_FRAMES = 400  # 最多获取的帧数
STRIDE = 5  # 帧间隔，每5帧保存一次

# 深度阈值配置
DEPTH_MIN = 0
DEPTH_MAX = 2
def create_output_folder():
    """创建带有时间戳的输出文件夹"""
    base_dir = os.path.join(os.path.dirname(__file__), 'out')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'capture_{timestamp}') 
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建 point 和 mask 子目录
    point_dir = os.path.join(results_dir, "point")
    mask_dir = os.path.join(results_dir, "mask")
    os.makedirs(point_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    return output_dir, results_dir

def save_camera_intrinsics_extrinsics(output_dir, profile):
    """获取并保存相机内外参"""
    # 获取深度和彩色流的配置
    depth_stream = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    color_stream = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    
    # 获取内参
    depth_intrinsics = depth_stream.get_intrinsics()
    color_intrinsics = color_stream.get_intrinsics()
    
    # 获取外参（深度到彩色的变换矩阵）
    extrinsics = depth_stream.get_extrinsics_to(color_stream)
    
    # 保存深度相机内参
    # depth_intrinsic_file = os.path.join(output_dir, 'depth_intrinsics.txt')
    # with open(depth_intrinsic_file, 'w') as f:
    #     f.write("Depth Camera Intrinsics:\n")
    #     f.write(f"Width: {depth_intrinsics.width}\n")
    #     f.write(f"Height: {depth_intrinsics.height}\n")
    #     f.write(f"PPX (Principal Point X): {depth_intrinsics.ppx}\n")
    #     f.write(f"PPY (Principal Point Y): {depth_intrinsics.ppy}\n")
    #     f.write(f"FX (Focal Length X): {depth_intrinsics.fx}\n")
    #     f.write(f"FY (Focal Length Y): {depth_intrinsics.fy}\n")
    #     f.write(f"Model: {depth_intrinsics.model}\n")
    #     f.write(f"Coeffs (Distortion Coefficients): {list(depth_intrinsics.coeffs)}\n")
    
    # 保存彩色相机内参  对齐后只需要RGB的内参
    color_intrinsic_file = os.path.join(output_dir, 'intrinsics.txt')
    with open(color_intrinsic_file, 'w') as f:
        f.write("Color Camera Intrinsics:\n")
        f.write(f"Width: {color_intrinsics.width}\n")
        f.write(f"Height: {color_intrinsics.height}\n")
        f.write(f"PPX (Principal Point X): {color_intrinsics.ppx}\n")
        f.write(f"PPY (Principal Point Y): {color_intrinsics.ppy}\n")
        f.write(f"FX (Focal Length X): {color_intrinsics.fx}\n")
        f.write(f"FY (Focal Length Y): {color_intrinsics.fy}\n")
        f.write(f"Model: {color_intrinsics.model}\n")
        f.write(f"Coeffs (Distortion Coefficients): {list(color_intrinsics.coeffs)}\n")
    
    # 保存外参（深度到彩色的变换）
    extrinsic_file = os.path.join(output_dir, 'extrinsics.txt')
    with open(extrinsic_file, 'w') as f:
        f.write("Extrinsics (Depth to Color):\n")
        f.write(f"Translation (X, Y, Z): {extrinsics.translation}\n")
        f.write(f"Rotation Matrix:\n")
        for i in range(3):
            f.write(f"  [{extrinsics.rotation[i*3]:.6f}, {extrinsics.rotation[i*3+1]:.6f}, {extrinsics.rotation[i*3+2]:.6f}]\n")
    
    print(f"相机参数已保存到: {output_dir}")
    # print(f"- 深度相机内参: depth_intrinsics.txt")
    print(f"- 彩色相机内参: intrinsics.txt")
    print(f"- 外参: extrinsics.txt")

def _init_depth_process():
    """初始化深度处理流程，设置各种滤波器以提高深度数据质量"""
    # 创建深度到视差的转换器（True表示转换到视差空间）
    depth_to_disparity = rs.disparity_transform(True)
    # 创建视差到深度的转换器（False表示转换回深度空间）
    disparity_to_depth = rs.disparity_transform(False)
    # 创建空间滤波器，用于平滑深度图像
    spatial = rs.spatial_filter()
    # 设置空间滤波器的幅度参数（控制滤波强度）
    spatial.set_option(rs.option.filter_magnitude, 5)
    # 设置空间滤波器的平滑alpha参数（控制平滑程度，0-1之间）
    spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
    # 设置空间滤波器的平滑delta参数（控制平滑的阈值）
    spatial.set_option(rs.option.filter_smooth_delta, 1)
    # 设置空间滤波器的孔洞填充参数（1表示填充小孔洞）
    spatial.set_option(rs.option.holes_fill, 1)
    # 创建时间滤波器，用于在时间维度上平滑深度数据
    temporal = rs.temporal_filter()
    # 设置时间滤波器的平滑alpha参数
    temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
    # 设置时间滤波器的平滑delta参数
    temporal.set_option(rs.option.filter_smooth_delta, 1)
    
    return depth_to_disparity, disparity_to_depth, spatial, temporal

def _process_depth(depth_frame, depth_to_disparity, disparity_to_depth, spatial, temporal):
    """对深度帧进行滤波处理，提高深度数据质量"""
    # 深度处理流程
    # 第一步：将深度转换为视差（视差空间更适合滤波）
    filtered_depth = depth_to_disparity.process(depth_frame)
    # 第二步：应用空间滤波器，平滑视差数据
    filtered_depth = spatial.process(filtered_depth)
    # 第三步：应用时间滤波器，在时间维度上平滑数据
    filtered_depth = temporal.process(filtered_depth)
    # 第四步：将视差转换回深度
    filtered_depth = disparity_to_depth.process(filtered_depth)
    # 返回处理后的深度帧
    return filtered_depth

def main():
    # 创建输出文件夹
    output_dir , results_dir = create_output_folder()
    print(f"输出文件夹: {output_dir}")
    print(f"结果文件夹: {results_dir}")
    print(f"分辨率: {WIDTH}x{HEIGHT}, 帧率: {FPS}")
    print(f"将获取 {MAX_FRAMES} 帧数据...")

    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用深度流和彩色流
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 启动管道
    profile = pipeline.start(config)

    # 获取并保存相机内外参
    save_camera_intrinsics_extrinsics(output_dir, profile)

    # 获取深度传感器的深度缩放因子
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = 1 / depth_scale

    # 初始化深度处理流程
    depth_to_disparity, disparity_to_depth, spatial, temporal = _init_depth_process()

    # 获取对齐流的配置，用于将深度图对齐到彩色图
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        frame_count = 0 
        save_count = 0
        # 定义可视化窗口的较小分辨率
        viz_width = 480
        viz_height = 240
        
        while save_count < MAX_FRAMES:
            # 等待帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧   对齐后采用RGB内参生成点云即可  
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 对深度帧进行滤波处理
            filtered_depth = _process_depth(depth_frame, depth_to_disparity, disparity_to_depth, spatial, temporal)

            # 转换为 NumPy 数组
            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 计算点云
            pointcloud = rs.pointcloud()
            pointcloud.map_to(color_frame)
            pointcloud = pointcloud.calculate(filtered_depth)
            points = (
                np.asanyarray(pointcloud.get_vertices())
                .view(np.float32)
                .reshape([HEIGHT, WIDTH, 3])
            )
            
            # 计算深度掩码
            mask = np.logical_and(
                (depth_image > DEPTH_MIN * clipping_distance),
                (depth_image < DEPTH_MAX * clipping_distance)
            )
            
            # 深度图可视化（归一化到 0~255）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # 将深度图映射到RGB图像上
            # 创建一个副本用于绘制深度信息
            depth_mapped_to_rgb = color_image.copy()
            
            # 创建深度阈值掩码 (例如，只显示距离在0.5米到3米之间的深度)
            depth_min = 500  # 0.5米 (以毫米为单位)
            depth_max = 3000  # 3米 (以毫米为单位)
            depth_mask = (depth_image * depth_scale * 1000 > depth_min) & (depth_image * depth_scale * 1000 < depth_max)
            
            # 在RGB图像上绘制深度轮廓
            depth_visualization = np.zeros_like(color_image)
            depth_visualization[depth_mask] = [0, 255, 0]  # 绿色表示有效深度区域
            depth_mapped_to_rgb = cv2.addWeighted(depth_mapped_to_rgb, 1, depth_visualization, 0.3, 0)

            # 将图像resize到较小的分辨率用于可视化
            resized_color = cv2.resize(color_image, (viz_width, viz_height))
            resized_depth = cv2.resize(depth_colormap, (viz_width, viz_height))
            resized_depth_mapped = cv2.resize(depth_mapped_to_rgb, (viz_width, viz_height))
            
            # 拼接图像布局 (2x2网格布局)
            top_row = np.hstack((resized_color, resized_depth))  # RGB和深度图水平拼接
            bottom_row = np.hstack((resized_depth_mapped, resized_depth_mapped))  # 为了匹配宽度，复制深度映射图
            # 或者使用一个全黑的图像来填充，使其宽度匹配top_row
            empty_space = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)  # 创建一个黑色图像
            bottom_row = np.hstack((resized_depth_mapped, empty_space))  # 用深度映射图和空白区域拼接
            
            # 垂直拼接形成2x2布局
            combined_display = np.vstack((top_row, bottom_row))
            
            # 显示拼接后的图像
            cv2.imshow('RGB & Depth Visualization (Combined Layout)', combined_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' key or Esc key (ASCII 27)
                break
            if frame_count % STRIDE == 0 and frame_count > 30: 
                # 保存深度图 (npy格式)
                depth_filename = os.path.join(results_dir, f'depth{save_count:04d}.npy')
                np.save(depth_filename, depth_image)

                # 保存 RGB 图
                color_filename = os.path.join(results_dir, f'frame{save_count:04d}.jpg')
                cv2.imwrite(color_filename, color_image)
                
                # 保存点云数据 (npy格式)
                point_filename = os.path.join(results_dir, 'point', f'{save_count:04d}.npy')
                np.save(point_filename, points)
                
                # 保存掩码数据 (npy格式)
                mask_filename = os.path.join(results_dir, 'mask', f'{save_count:04d}.npy')
                np.save(mask_filename, mask)

                # # 保存深度映射到RGB的图像
                # mapped_filename = os.path.join(output_dir, f'depth_mapped_rgb{save_count:04d}.jpg')
                # cv2.imwrite(mapped_filename, depth_mapped_to_rgb)

                save_count += 1
                print(f"已保存第 {save_count}/{MAX_FRAMES} 帧")
            frame_count += 1


        print(f"\n完成! 共保存 {save_count} 帧数据到: {results_dir}")

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()