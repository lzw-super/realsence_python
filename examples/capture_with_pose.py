import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import threading
import time
import math

# ===================== 全局配置参数 =====================
# 图像采集参数
WIDTH = 1280
HEIGHT = 720
FPS = 30
MAX_FRAMES = 100  # 最多获取的帧数
STRIDE = 5  # 帧间隔，每5帧保存一次

# IMU位姿估计参数（四元数版）
alpha = 0.98  # 互补滤波器权重
first_imu_frame = True
last_gyro_ts = 0.0

# 全局位姿存储（四元数：w, x, y, z），线程安全锁
imu_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数初始值
quaternion_lock = threading.Lock()

# VO位姿存储（7元数：QW, QX, QY, QZ, TX, TY, TZ）
vo_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
vo_lock = threading.Lock()

# ===================== 四元数工具函数 =====================
def quaternion_multiply(q1, q2):
    """
    四元数乘法：q1 * q2
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def quaternion_normalize(q):
    """四元数归一化"""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

def euler_to_quaternion(roll, pitch, yaw):
    """
    欧拉角转四元数（Z-Y-X顺序，弧度）
    roll: 滚转 (x轴)
    pitch: 俯仰 (y轴)
    yaw: 偏航 (z轴)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    return np.array([w, x, y, z])

def quaternion_to_euler(q):
    """
    四元数转欧拉角（Z-Y-X顺序，弧度）
    返回: roll, pitch, yaw
    """
    w, x, y, z = q
    
    # 滚转 (x轴)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # 俯仰 (y轴)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # 偏航 (z轴)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def gyro_to_quaternion(gyro_data, dt):
    """
    陀螺仪角速度转四元数增量
    gyro_data: [x, y, z] 角速度 (rad/s)
    dt: 时间差 (s)
    """
    # 计算旋转向量（角速度 * 时间差）
    rx = gyro_data.x * dt
    ry = gyro_data.y * dt
    rz = gyro_data.z * dt
    
    # 旋转向量转四元数
    theta = np.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis_x = rx / theta
    axis_y = ry / theta
    axis_z = rz / theta
    
    half_theta = theta / 2
    w = math.cos(half_theta)
    x = axis_x * math.sin(half_theta)
    y = axis_y * math.sin(half_theta)
    z = axis_z * math.sin(half_theta)
    
    return np.array([w, x, y, z])

# ===================== IMU数据处理函数 =====================
def process_gyro(gyro_frame):
    """处理陀螺仪数据，更新四元数位姿"""
    global imu_quaternion, first_imu_frame, last_gyro_ts
    
    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
    current_ts = gyro_frame.get_timestamp()
    
    if first_imu_frame:
        last_gyro_ts = current_ts
        return
    
    # 计算时间差
    dt = (current_ts - last_gyro_ts) / 1000.0
    last_gyro_ts = current_ts
    
    # 计算陀螺仪对应的四元数增量
    delta_q = gyro_to_quaternion(gyro_data, dt)
    
    # 更新全局四元数（q = delta_q * q）
    with quaternion_lock:
        imu_quaternion = quaternion_multiply(delta_q, imu_quaternion)
        imu_quaternion = quaternion_normalize(imu_quaternion)

def process_accel(accel_frame):
    """处理加速度计数据，校准四元数位姿"""
    global imu_quaternion, first_imu_frame
    
    accel_data = accel_frame.as_motion_frame().get_motion_data()
    
    # 由加速度计计算重力方向的四元数（仅校准俯仰和滚转）
    ax = accel_data.x
    ay = accel_data.y
    az = accel_data.z
    
    # 计算俯仰和滚转
    pitch = math.atan2(ax, math.sqrt(ay*ay + az*az))
    roll = math.atan2(ay, az)
    yaw = 0.0  # 加速度计无法估计偏航角
    
    # 转换为四元数
    accel_q = euler_to_quaternion(roll, pitch, yaw)
    
    with quaternion_lock:
        if first_imu_frame:
            # 首帧初始化：保留初始偏航角，更新俯仰和滚转
            current_roll, current_pitch, current_yaw = quaternion_to_euler(imu_quaternion)
            imu_quaternion = euler_to_quaternion(roll, pitch, current_yaw)
            first_imu_frame = False
        else:
            # 互补滤波器融合：alpha*陀螺仪 + (1-alpha)*加速度计
            current_roll, current_pitch, current_yaw = quaternion_to_euler(imu_quaternion)
            fused_roll = alpha * current_roll + (1 - alpha) * roll
            fused_pitch = alpha * current_pitch + (1 - alpha) * pitch
            
            # 更新四元数
            imu_quaternion = euler_to_quaternion(fused_roll, fused_pitch, current_yaw)
            imu_quaternion = quaternion_normalize(imu_quaternion)

def imu_reader_thread():
    """独立线程读取IMU数据"""
    # 配置IMU管道
    imu_pipeline = rs.pipeline()
    imu_config = rs.config()
    imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
    
    try:
        imu_pipeline.start(imu_config)
        print("IMU线程已启动，开始读取位姿数据...")
        
        while True:
            frames = imu_pipeline.wait_for_frames()
            
            # 处理陀螺仪帧
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if gyro_frame:
                process_gyro(gyro_frame)
            
            # 处理加速度计帧
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                process_accel(accel_frame)
            
            time.sleep(0.001)
    
    except Exception as e:
        print(f"IMU线程出错: {e}")
    finally:
        imu_pipeline.stop()

# ===================== 原有图像采集函数 =====================
def create_output_folder():
    """创建带有时间戳的输出文件夹"""
    base_dir = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else os.getcwd(), 'out')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'capture_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

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
    depth_intrinsic_file = os.path.join(output_dir, 'depth_intrinsics.txt')
    with open(depth_intrinsic_file, 'w') as f:
        f.write("Depth Camera Intrinsics:\n")
        f.write(f"Width: {depth_intrinsics.width}\n")
        f.write(f"Height: {depth_intrinsics.height}\n")
        f.write(f"PPX (Principal Point X): {depth_intrinsics.ppx}\n")
        f.write(f"PPY (Principal Point Y): {depth_intrinsics.ppy}\n")
        f.write(f"FX (Focal Length X): {depth_intrinsics.fx}\n")
        f.write(f"FY (Focal Length Y): {depth_intrinsics.fy}\n")
        f.write(f"Model: {depth_intrinsics.model}\n")
        f.write(f"Coeffs (Distortion Coefficients): {list(depth_intrinsics.coeffs)}\n")
    
    # 保存彩色相机内参
    color_intrinsic_file = os.path.join(output_dir, 'color_intrinsics.txt')
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
    print(f"- 深度相机内参: depth_intrinsics.txt")
    print(f"- 彩色相机内参: color_intrinsics.txt")
    print(f"- 外参: extrinsics.txt")

def save_pose_data(output_dir, frame_idx, seven_dof_pose):
    """保存指定帧的位姿数据（7元数：QW QX QY QZ TX TY TZ）"""
    # 提取7元数
    qw, qx, qy, qz, tx, ty, tz = seven_dof_pose
    
    # 转换四元数为欧拉角（角度制，便于阅读）
    roll, pitch, yaw = quaternion_to_euler([qw, qx, qy, qz])
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    # 保存单帧位姿（7元数格式）
    pose_file = os.path.join(output_dir, f'pose_{frame_idx:04d}.txt')
    with open(pose_file, 'w') as f:
        f.write(f"Frame Index: {frame_idx}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write("7-DoF Pose (QW QX QY QZ TX TY TZ):\n")
        f.write(f"  QW: {qw:.8f}\n")
        f.write(f"  QX: {qx:.8f}\n")
        f.write(f"  QY: {qy:.8f}\n")
        f.write(f"  QZ: {qz:.8f}\n")
        f.write(f"  TX: {tx:.8f} m\n")
        f.write(f"  TY: {ty:.8f} m\n")
        f.write(f"  TZ: {tz:.8f} m\n")
        f.write("Euler Angles (degrees):\n")
        f.write(f"  Roll (滚转): {roll_deg:.4f}°\n")
        f.write(f"  Pitch (俯仰): {pitch_deg:.4f}°\n")
        f.write(f"  Yaw (偏航): {yaw_deg:.4f}°\n")
    
    # 追加到总位姿文件（7元数CSV格式）
    pose_summary_file = os.path.join(output_dir, 'pose_summary.txt')
    mode = 'a' if os.path.exists(pose_summary_file) else 'w'
    with open(pose_summary_file, mode) as f:
        if mode == 'w':
            f.write("Frame Index, QW, QX, QY, QZ, TX(m), TY(m), TZ(m), Roll(deg), Pitch(deg), Yaw(deg)\n")
        f.write(f"{frame_idx}, {qw:.8f}, {qx:.8f}, {qy:.8f}, {qz:.8f}, {tx:.8f}, {ty:.8f}, {tz:.8f}, {roll_deg:.4f}, {pitch_deg:.4f}, {yaw_deg:.4f}\n")

# ===================== 主函数 =====================
def main():
    # 启动IMU线程
    imu_thread = threading.Thread(target=imu_reader_thread, daemon=True)
    imu_thread.start()
    time.sleep(1)  # 等待IMU线程初始化
    
    # 创建输出文件夹
    output_dir = create_output_folder()
    print(f"输出文件夹: {output_dir}")
    print(f"分辨率: {WIDTH}x{HEIGHT}, 帧率: {FPS}")
    print(f"将获取 {MAX_FRAMES} 帧数据...")

    # 配置 RealSense 管道（图像流）
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用深度流和彩色流
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 启动管道
    profile = pipeline.start(config)

    # 初始化视觉里程计（VO）模块
    vo = rs.visual_odometry()
    vo_config = rs.vo_config()
    vo_config.enable_imu_fusion(True)  # 融合IMU数据提升精度
    vo_config.set_from_pipeline_profile(profile)  # 从管道配置中获取相机参数
    vo_config.set_max_distance(3.0)  # 设置最大有效深度距离（米）
    print("视觉里程计(VO)已初始化，启用IMU融合...")

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

            # 更新视觉里程计
            vo_status = vo.process(aligned_frames, vo_config)
            
            # 获取VO位姿（7元数）
            if vo_status == rs.rs2_visual_odometry_status.rs2_vos_valid:
                pose = vo.get_pose()
                with vo_lock:
                    # 更新全局VO位姿（7元数：QW, QX, QY, QZ, TX, TY, TZ）
                    vo_pose[0] = pose.rotation.w  # QW
                    vo_pose[1] = pose.rotation.x  # QX
                    vo_pose[2] = pose.rotation.y  # QY
                    vo_pose[3] = pose.rotation.z  # QZ
                    vo_pose[4] = pose.translation.x  # TX (米)
                    vo_pose[5] = pose.translation.y  # TY (米)
                    vo_pose[6] = pose.translation.z  # TZ (米)
            else:
                print(f"VO状态无效: {vo_status}")

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
            
            # 在空白区域添加位姿信息显示
            with quaternion_lock:
                current_q = imu_quaternion.copy()
                roll, pitch, yaw = quaternion_to_euler(current_q)
                roll_deg = math.degrees(roll)
                pitch_deg = math.degrees(pitch)
                yaw_deg = math.degrees(yaw)
            
            with vo_lock:
                current_vo_pose = vo_pose.copy()
                tx, ty, tz = current_vo_pose[4], current_vo_pose[5], current_vo_pose[6]
            
            # 绘制位姿信息（包含IMU旋转和VO平移）
            pose_display = empty_space.copy()
            cv2.putText(pose_display, "Camera Pose (VO+IMU)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(pose_display, f"Roll: {roll_deg:.2f}°", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(pose_display, f"Pitch: {pitch_deg:.2f}°", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(pose_display, f"Yaw: {yaw_deg:.2f}°", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(pose_display, f"TX: {tx:.3f}m", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(pose_display, f"TY: {ty:.3f}m", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(pose_display, f"TZ: {tz:.3f}m", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(pose_display, f"Frame: {save_count}/{MAX_FRAMES}", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            bottom_row = np.hstack((resized_depth_mapped, pose_display))
            combined_display = np.vstack((top_row, bottom_row))
            
            # 显示拼接后的图像
            cv2.imshow('RGB & Depth & VO+IMU Pose Visualization', combined_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' key or Esc key
                break
            
            # 按间隔保存帧和对应的位姿
            if frame_count % STRIDE == 0 and frame_count > 30: 
                # 保存深度图 
                depth_filename = os.path.join(output_dir, f'depth{save_count:04d}.png')
                cv2.imwrite(depth_filename, depth_image)

                # 保存 RGB 图
                color_filename = os.path.join(output_dir, f'frame{save_count:04d}.jpg')
                cv2.imwrite(color_filename, color_image)

                # 保存7元数位姿数据（优先使用VO位姿，无则用IMU旋转+0平移）
                with vo_lock:
                    if vo_status == rs.rs2_visual_odometry_status.rs2_vos_valid:
                        save_pose_data(output_dir, save_count, vo_pose.copy())
                    else:
                        # 备用方案：IMU旋转 + 0平移
                        backup_pose = np.array([current_q[0], current_q[1], current_q[2], current_q[3], 0.0, 0.0, 0.0])
                        save_pose_data(output_dir, save_count, backup_pose)

                save_count += 1
                print(f"已保存第 {save_count}/{MAX_FRAMES} 帧 (含7元数位姿数据)")
            
            frame_count += 1

        print(f"\n完成! 共保存 {save_count} 帧数据到: {output_dir}")
        print(f"- 图像文件: frameXXXX.jpg (RGB), depthXXXX.png (深度)")
        print(f"- 位姿文件: poseXXXX.txt (7元数位姿), pose_summary.txt (汇总)")
        print(f"- 位姿格式: QW QX QY QZ TX TY TZ (旋转四元数 + 平移米数)")

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()