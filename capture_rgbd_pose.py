import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import math 
import open3d as o3d
# ===================== 全局配置参数 =====================
WIDTH = 1280
HEIGHT = 720
FPS = 30
MAX_FRAMES = 100  # 最多获取的帧数
STRIDE = 1  # 帧间隔，每1帧保存一次（可以改为5以减少数据量）

def create_output_folder():
    """创建带有时间戳的输出文件夹"""
    base_dir = r'E:\Users\lenovo\Desktop\realsense\realsence_python\examples\out'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'capture_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
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
    
    # 保存彩色相机内参（对齐后只需要RGB的内参）
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
    print(f"- 彩色相机内参: intrinsics.txt")
    print(f"- 外参: extrinsics.txt")
def load_intrinsic(intrinsic_path):
    """读取相机内参文件"""
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
    
    params = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in ['PPX (Principal Point X)', 'PPY (Principal Point Y)', 'FX (Focal Length X)', 'FY (Focal Length Y)']:
                params[key] = float(value)
    
    fx = params['FX (Focal Length X)']
    fy = params['FY (Focal Length Y)']
    cx = params['PPX (Principal Point X)']
    cy = params['PPY (Principal Point Y)']
    return fx, fy, cx, cy

def compute_poses_from_rgbd(data_dir, output_file="poses_16dof.txt"):
    """
    从RGB-D图像计算相机位姿（16元数：4x4变换矩阵）
    
    Args:
        data_dir: 包含RGB-D图像和内参文件的目录
        output_file: 输出位姿文件名
    """
    results_dir = os.path.join(data_dir, "results")
    # 自动获取所有 frame 和 depth 文件（按数字排序）
    rgb_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(results_dir) if f.startswith("depth") and f.endswith(".png")])
    
    # 确保数量一致
    assert len(rgb_files) == len(depth_files), "RGB 和 Depth 图像数量不匹配！"
    num_frames = len(rgb_files)
    
    print(f"共找到 {num_frames} 帧 RGB-D 图像。")
    
    # 读取相机内参
    intrinsic_path = os.path.join(data_dir, "intrinsics.txt")
    if not os.path.exists(intrinsic_path):
        intrinsic_path = os.path.join(data_dir, "color_intrinsics.txt")
    
    fx, fy, cx, cy = load_intrinsic(intrinsic_path)
    print(f"相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # 获取图像尺寸（从第一张图读取）
    first_rgb = cv2.imread(os.path.join(results_dir, rgb_files[0]))
    height, width = first_rgb.shape[:2]
    
    # 创建Open3D相机内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    
    # 初始化轨迹
    trajectory = [np.eye(4)]  # 第一帧设为世界原点
    prev_rgbd = None
    
    # 辅助函数：提取帧序号
    def get_frame_id(filename):
        return int(''.join(filter(str.isdigit, filename)))
    
    # 确保按数字顺序排列
    rgb_files.sort(key=get_frame_id)
    depth_files.sort(key=get_frame_id)
    
    # 逐帧处理
    for i in range(num_frames):
        rgb_path = os.path.join(results_dir, rgb_files[i])
        depth_path = os.path.join(results_dir, depth_files[i])
        
        color = cv2.imread(rgb_path)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16-bit
        
        if color is None or depth_raw is None:
            print(f"警告：跳过第 {i} 帧（图像读取失败）")
            continue
            
        depth = depth_raw.astype(np.float32)  # 转为 float 供 Open3D 使用
        
        # 创建RGBD图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1000.0,      # mm → m
            depth_trunc=10.0,        # 忽略 10 米以外的深度（可调）
            convert_rgb_to_intensity=False
        )
        
        if i == 0:
            prev_rgbd = rgbd
            continue
        
        # 计算当前帧相对于上一帧的变换
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        
        success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd, rgbd, intrinsic,
            np.eye(4), jacobian,
            o3d.pipelines.odometry.OdometryOption()
        )
        
        if success:
            current_pose = trajectory[-1] @ trans
            trajectory.append(current_pose)
            prev_rgbd = rgbd
            print(f"Frame {i}: 成功跟踪")
        else:
            print(f"Frame {i}: 跟踪失败！复用上一帧位姿")
            trajectory.append(trajectory[-1])  # 保持上一位置
    
    # 保存为16元数格式（4x4变换矩阵）
    output_path = os.path.join(data_dir, output_file)
    
    with open(output_path, 'w') as f:
        for pose in trajectory:
            # 将 4x4 矩阵展平为 1D 数组（行优先）
            flattened = pose.flatten()  # shape: (16,)
            # 转为字符串，保留18位小数
            line = ' '.join([f"{x:.18f}" for x in flattened])
            f.write(line + '\n')
    
    print(f"✅ 已将 {len(trajectory)} 个位姿写入 {output_path}")
    print(f"   格式：16元数（4x4变换矩阵，行优先）")
    
    # 同时保存为TUM格式（时间戳 tx ty tz qx qy qz qw）
    tum_output_path = os.path.join(data_dir, "poses_tum.txt")
    with open(tum_output_path, 'w') as f:
        for i, pose in enumerate(trajectory):
            # 提取平移
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            
            # 提取旋转矩阵并转换为四元数
            R = pose[:3, :3]
            trace = np.trace(R)
            
            if trace > 0:
                qw = math.sqrt(trace + 1.0) / 2.0
                qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
                qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
                qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            # 时间戳（假设30 FPS）
            timestamp = i / 30.0
            
            f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"✅ 已将 {len(trajectory)} 个位姿写入 {tum_output_path}")
    print(f"   格式：TUM格式（时间戳 tx ty tz qx qy qz qw）")
    # 保存为16元数C2W格式（4x4变换矩阵，行优先）
    c2w_output_path = os.path.join(data_dir, "poses_c2w.txt")
    with open(c2w_output_path, 'w') as f:
        for pose_w2c in trajectory:
            # C2W 是 W2C 的逆矩阵
            pose_c2w = np.linalg.inv(pose_w2c)
            # 将 4x4 矩阵展平为 1D 数组（行优先）
            flattened = pose_c2w.flatten()  # shape: (16,)
            # 转为字符串，保留18位小数
            line = ' '.join([f"{x:.18f}" for x in flattened])
            f.write(line + '\n')
    
    print(f"✅ 已将 {len(trajectory)} 个位姿写入 {c2w_output_path}")
    print(f"   格式：16元数C2W（4x4变换矩阵，行优先）")
def main():
    # 创建输出文件夹
    output_dir, results_dir = create_output_folder()
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

    # 获取对齐流的配置，用于将深度图对齐到彩色图
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        frame_count = 0 
        save_count = 0
        # 定义可视化窗口的较小分辨率
        viz_width = 640
        viz_height = 360
        
        while save_count < MAX_FRAMES:
            # 等待帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 转换为 NumPy 数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 深度图可视化（归一化到 0~255）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # 将深度图映射到RGB图像上
            depth_mapped_to_rgb = color_image.copy()
            depth_min = 500  # 0.5米 (以毫米为单位)
            depth_max = 3000  # 3米 (以毫米为单位)
            depth_mask = (depth_image * depth_scale * 1000 > depth_min) & (depth_image * depth_scale * 1000 < depth_max)
            depth_visualization = np.zeros_like(color_image)
            depth_visualization[depth_mask] = [0, 255, 0]  # 绿色表示有效深度区域
            depth_mapped_to_rgb = cv2.addWeighted(depth_mapped_to_rgb, 1, depth_visualization, 0.3, 0)

            # 将图像resize到较小的分辨率用于可视化
            resized_color = cv2.resize(color_image, (viz_width, viz_height))
            resized_depth = cv2.resize(depth_colormap, (viz_width, viz_height))
            resized_depth_mapped = cv2.resize(depth_mapped_to_rgb, (viz_width, viz_height))
            
            # 拼接图像布局
            top_row = np.hstack((resized_color, resized_depth))
            empty_space = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
            
            # 在空白区域添加信息显示
            info_display = empty_space.copy()
            cv2.putText(info_display, "RGB-D Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_display, f"Frame: {save_count}/{MAX_FRAMES}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(info_display, f"FPS: {FPS}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(info_display, f"Resolution: {WIDTH}x{HEIGHT}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(info_display, "Press 'q' to quit", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            
            bottom_row = np.hstack((resized_depth_mapped, info_display))
            combined_display = np.vstack((top_row, bottom_row))
            
            # 显示拼接后的图像
            cv2.imshow('RGB-D Capture Visualization', combined_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' key or Esc key
                break
            
            # 按间隔保存帧
            if frame_count % STRIDE == 0 and frame_count > 30: 
                # 保存深度图 
                depth_filename = os.path.join(results_dir, f'depth{save_count:04d}.png')
                cv2.imwrite(depth_filename, depth_image)

                # 保存 RGB 图
                color_filename = os.path.join(results_dir, f'frame{save_count:04d}.jpg')
                cv2.imwrite(color_filename, color_image)

                save_count += 1
                print(f"已保存第 {save_count}/{MAX_FRAMES} 帧")
            
            frame_count += 1

        print(f"\n完成! 共保存 {save_count} 帧数据到: {results_dir}")
        print(f"- 图像文件: frameXXXX.jpg (RGB), depthXXXX.png (深度)")
        print(f"\n提示：使用 compute_pose_offline.py 来计算相机位姿")

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
    data_dir = output_dir
    compute_poses_from_rgbd(data_dir)

if __name__ == '__main__':
    main()
