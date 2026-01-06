import os
import cv2
import numpy as np
import open3d as o3d
import math

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
    # 自动获取所有 frame 和 depth 文件（按数字排序）
    rgb_files = sorted([f for f in os.listdir(data_dir) if f.startswith("frame") and f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(data_dir) if f.startswith("depth") and f.endswith(".png")])
    
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
    first_rgb = cv2.imread(os.path.join(data_dir, rgb_files[0]))
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
        rgb_path = os.path.join(data_dir, rgb_files[i])
        depth_path = os.path.join(data_dir, depth_files[i])
        
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

if __name__ == '__main__':
    # 设置数据目录
    data_dir = r'E:\Users\lenovo\Desktop\realsense\realsence_python\examples\out\capture_20251230_160728\results'
    
    # 计算位姿
    compute_poses_from_rgbd(data_dir)
