# 导入Intel RealSense SDK的Python绑定库，用于控制RealSense相机
import pyrealsense2 as rs
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 导入PIL库的Image模块，用于图像处理
from PIL import Image
# 从工具模块导入相机配置参数：WH(宽高)、RECORDER_DIR(录制目录)、DEPTH_MIN/MAX(深度阈值)
from dovsg.utils.utils import WH, RECORDER_DIR, DEPTH_MIN, DEPTH_MAX
# 导入操作系统接口模块，用于文件和目录操作
import os
# 导入shutil模块，用于高级文件操作（如删除目录）
import shutil
# 导入OpenCV库，用于计算机视觉和图像处理
import cv2
# 导入time模块，用于时间相关操作
import time
# 导入threading模块，用于多线程编程
import threading
# 导入IPython的embed函数，用于交互式调试
from IPython import embed
# 导入json模块，用于JSON格式的数据读写
import json
# 导入tqdm进度条库，用于显示处理进度
from tqdm import tqdm
# 导入matplotlib的pyplot模块，用于数据可视化
import matplotlib.pyplot as plt
# 导入typing模块的Union类型，用于类型提示
from typing import Union

# 实际所需的图像尺寸，确保不捕获距离相机过近的深度和颜色数据
real_height = 600  # 移除底部200像素以去除图像底部的机械臂
real_width = 1200

# 定义RecorderImage类，用于录制RealSense相机的图像数据
class RecorderImage():
    # 初始化方法，设置相机参数和录制配置
    def __init__(self, recorder_dir=None, serial_number="239222303321", 
                 WH=WH, FPS=30, depth_threshold=[DEPTH_MIN, DEPTH_MAX]):
        
        # 如果提供了录制目录，则初始化录制相关参数
        if recorder_dir is not None:
            # 设置录制标志为True，表示开始录制
            self.record_flag = True
            # 保存录制目录路径
            self.recorder_dir = recorder_dir
            # 初始化变换字典，用于存储坐标变换信息
            self.transforms = {}

            # 如果录制目录已存在，询问用户是否覆盖
            if os.path.isdir(self.recorder_dir):
                if input("Overwrite file y/n?: ") == "y":
                    # 如果用户输入y，则删除整个目录及其内容
                    shutil.rmtree(self.recorder_dir)
                else:
                    # 如果用户不覆盖，直接返回
                    return
            # 创建depth子目录，用于保存深度图像
            os.makedirs(self.recorder_dir / "depth", exist_ok=True)
            # 创建rgb子目录，用于保存彩色图像
            os.makedirs(self.recorder_dir / "rgb", exist_ok=True)
            # 创建point子目录，用于保存点云数据
            os.makedirs(self.recorder_dir / "point", exist_ok=True)
            # 创建mask子目录，用于保存深度掩码
            os.makedirs(self.recorder_dir / "mask", exist_ok=True)
        
        # 保存图像宽高参数
        self.WH = WH
        # 保存相机序列号，用于指定要使用的相机
        self.serial_number = serial_number
        # 保存帧率参数
        self.FPS = FPS
        # 保存深度阈值范围，用于过滤有效深度数据
        self.depth_threshold = depth_threshold

        # 创建RealSense配置对象
        self.config = rs.config()
        # 指定腕部相机的序列号，确保连接到正确的相机
        self.config.enable_device(self.serial_number)
        # 创建RealSense管道对象，用于管理数据流
        self.pipeline = rs.pipeline()
        # 启用深度数据流：宽度、高度、格式(z16表示16位深度值)、帧率
        self.config.enable_stream(rs.stream.depth, self.WH[0], self.WH[1], rs.format.z16, self.FPS)
        # 启用彩色数据流：宽度、高度、格式(bgr8表示8位BGR彩色)、帧率
        self.config.enable_stream(rs.stream.color, self.WH[0], self.WH[1], rs.format.bgr8, self.FPS)
        # 启动管道并获取配置文件
        profile = self.pipeline.start(self.config)
        # 跳过前15帧，给自动曝光时间进行调整
        for x in range(15):
            self.pipeline.wait_for_frames()
        # 获取彩色数据流
        color_stream = profile.get_stream(rs.stream.color)
        # 获取彩色相机的内参（焦距、主点等）
        self.intrinsic = color_stream.as_video_stream_profile().get_intrinsics()

        # 将内参转换为可读的矩阵形式
        self.intrinsic_matrix, self.dist_coef = self._get_readable_intrinsic()
        # 打印内参矩阵
        print(self.intrinsic_matrix, self.dist_coef)
        # 获取深度传感器对象
        depth_sensor = profile.get_device().first_depth_sensor()
        # 获取深度比例因子，用于将深度值转换为米
        self.depth_scale = depth_sensor.get_depth_scale()
        # 打印深度比例因子
        print("Depth Scale is: " , self.depth_scale)
        # 计算裁剪距离（最大有效深度），单位为米
        self.clipping_distance = 1 / self.depth_scale
        # 打印裁剪距离
        print(self.clipping_distance)
        # 初始化深度处理流程
        self._init_depth_process()
        # 初始化帧索引为0
        self.frame_index = 0
        # 初始化数据字典
        self.data = {}

    # 初始化深度处理流程，设置各种滤波器以提高深度数据质量
    def _init_depth_process(self):
        # 创建深度到视差的转换器（True表示转换到视差空间）
        self.depth_to_disparity = rs.disparity_transform(True)
        # 创建视差到深度的转换器（False表示转换回深度空间）
        self.disparity_to_depth = rs.disparity_transform(False)
        # 创建空间滤波器，用于平滑深度图像
        self.spatial = rs.spatial_filter()
        # 设置空间滤波器的幅度参数（控制滤波强度）
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        # 设置空间滤波器的平滑alpha参数（控制平滑程度，0-1之间）
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        # 设置空间滤波器的平滑delta参数（控制平滑的阈值）
        self.spatial.set_option(rs.option.filter_smooth_delta, 1)
        # 设置空间滤波器的孔洞填充参数（1表示填充小孔洞）
        self.spatial.set_option(rs.option.holes_fill, 1)
        # 创建时间滤波器，用于在时间维度上平滑深度数据
        self.temporal = rs.temporal_filter()
        # 设置时间滤波器的平滑alpha参数
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        # 设置时间滤波器的平滑delta参数
        self.temporal.set_option(rs.option.filter_smooth_delta, 1)
        # 初始化对齐器，将深度数据对齐到彩色相机坐标系
        self.align = rs.align(rs.stream.color)

    # 将RealSense内参转换为标准的3x3内参矩阵和畸变系数
    def _get_readable_intrinsic(self):
        # 构建相机内参矩阵：fx和fy是焦距，ppx和ppy是主点坐标
        intrinsic_matrix = np.array(
            [
                [self.intrinsic.fx, 0, self.intrinsic.ppx],
                [0, self.intrinsic.fy, self.intrinsic.ppy],
                [0, 0, 1],
            ]
        )
        # 提取畸变系数（k1, k2, p1, p2, k3等）
        dist_coef = np.array(self.intrinsic.coeffs)
        # 返回内参矩阵和畸变系数
        return intrinsic_matrix, dist_coef


    # 将3D点（相机坐标系）投影到2D像素坐标
    def project_point_to_pixel(self, points):
        # 输入的点应该在相机坐标系中，形状为n*3（n个点，每个点3个坐标）
        points = np.array(points)
        pixels = []
        # # 使用RealSense的投影函数，但在循环中速度较慢；对于无效点会返回nan
        # for i in range(len(points)):
        #     pixels.append(rs.rs2_project_point_to_pixel(self.intrinsic, points))
        # pixels = np.array(pixels)

        # 使用OpenCV的投影函数（速度更快）
        # 这里的宽度和高度是反转的
        pixels = cv2.projectPoints(
            points,  # 要投影的3D点
            np.zeros(3),  # 旋转向量（全零表示无旋转）
            np.zeros(3),  # 平移向量（全零表示无平移）
            self.intrinsic_matrix,  # 相机内参矩阵
            self.dist_coef,  # 畸变系数
        )[0][:, 0, :]

        # 返回像素坐标，并反转x和y坐标（因为宽高是反转的）
        return pixels[:, ::-1]

    # 将2D像素坐标和深度值反投影到3D点（相机坐标系）
    def deproject_pixel_to_point(self, pixel_depth):
        # pixel_depth包含[i, j, depth[i, j]]，即像素坐标和对应的深度值
        points = []
        # 遍历每个像素深度对
        for i in range(len(pixel_depth)):
            # 这里的宽度和高度是反转的
            points.append(
                rs.rs2_deproject_pixel_to_point(
                    self.intrinsic,  # 相机内参
                    [pixel_depth[i, 1], pixel_depth[i, 0]],  # 像素坐标（注意交换了i和j）
                    pixel_depth[i, 2],  # 深度值
                )
            )
        # 返回3D点数组
        return np.array(points)

    # 对深度帧进行滤波处理，提高深度数据质量
    def _process_depth(self, depth_frame):
        # 深度处理流程
        # 第一步：将深度转换为视差（视差空间更适合滤波）
        filtered_depth = self.depth_to_disparity.process(depth_frame)
        # 第二步：应用空间滤波器，平滑视差数据
        filtered_depth = self.spatial.process(filtered_depth)
        # 第三步：应用时间滤波器，在时间维度上平滑数据
        filtered_depth = self.temporal.process(filtered_depth)
        # 第四步：将视差转换回深度
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        # 返回处理后的深度帧
        return filtered_depth

    # 获取当前帧的观测数据（点云、彩色图像、深度图像、掩码）
    def get_observations(self):
        # 获取图像宽度和高度
        width, height = self.WH
        # 初始化深度帧为None
        depth_frame = None
        # 跳过前5帧，确保获取稳定的帧
        for x in range(5):  # 每帧跳过5帧
            self.pipeline.wait_for_frames()
        # 循环直到获取到有效的深度帧
        while not depth_frame:
            # 从管道获取一帧数据
            frames = self.pipeline.wait_for_frames()
            # 将深度帧对齐到彩色相机坐标系
            aligned_frames = self.align.process(frames)    #对齐细节
            # 获取对齐后的深度帧
            depth_frame = aligned_frames.get_depth_frame()
            # 获取对齐后的彩色帧
            color_frame = aligned_frames.get_color_frame()

        # 对深度帧进行滤波处理
        filtered_depth = self._process_depth(depth_frame)
        # 创建点云对象
        pointcloud = rs.pointcloud()
        # 将点云映射到彩色帧（为点云添加颜色信息）
        pointcloud.map_to(color_frame)
        # 根据处理后的深度数据计算点云
        pointcloud = pointcloud.calculate(filtered_depth)
        # 获取3D点并重塑为图像形状
        points = (
            np.asanyarray(pointcloud.get_vertices())  # 获取顶点数据
            .view(np.float32)  # 转换为float32类型
            .reshape([height, width, 3])  # 重塑为[高度,宽度,3]的数组
        )
        # 将颜色从BGR转换为RGB（注释掉的代码）
        # colors = (np.asanyarray(color_frame.get_data()) / 255.0)[:, :, ::-1]
        # # 获取深度图像，使深度单位为米（注释掉的代码）
        # depths = np.asanyarray(filtered_depth.get_data()) / self.clipping_distance
        # 获取彩色图像数据（BGR格式）
        colors = np.asanyarray(color_frame.get_data())
        # 获取深度图像数据（原始深度值）
        depths = np.asanyarray(filtered_depth.get_data())
        # 获取深度阈值内的有效深度掩码
        mask = np.logical_and(
            (depths > self.depth_threshold[0] * self.clipping_distance),  # 深度大于最小阈值
            (depths < self.depth_threshold[1] * self.clipping_distance)  # 深度小于最大阈值
        )
        # 返回点云、彩色图像、深度图像和掩码
        return points, colors, depths, mask


    # 获取对齐后的帧并保存到文件
    def get_align_frame(self, frame_index):
        # 获取帧时不能直接裁剪，否则可能无法获取有效帧
        try:
            # 获取观测数据（点云、彩色图像、深度图像、掩码）
            points, colors, depths, mask = self.get_observations()
            # 构建彩色图像保存路径（6位数字编号，jpg格式）
            color_image_path = self.recorder_dir / "rgb" / f"{frame_index:06}.jpg"
            # 构建深度图像保存路径（npy格式）
            depth_path = self.recorder_dir / "depth" / f"{frame_index:06}.npy"
            # 构建点云保存路径（npy格式）
            point_path = self.recorder_dir / "point" / f"{frame_index:06}.npy"
            # 构建掩码保存路径（npy格式）
            mask_path = self.recorder_dir / "mask" / f"{frame_index:06}.npy"
            # 保存彩色图像，使用最高JPEG质量（100）
            cv2.imwrite(str(color_image_path), colors, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # 保存深度图像为numpy数组
            np.save(str(depth_path), depths)
            # 保存点云数据为numpy数组
            np.save(str(point_path), points)
            # 保存掩码数据为numpy数组
            np.save(str(mask_path), mask)
        except:
            # 如果出现异常，返回False表示保存失败
            return False
        # 返回True表示保存成功
        return True

    # 修改图像尺寸，裁剪底部区域并保存标定数据
    def change_size(self):
        # 创建calibration子目录，用于保存标定数据
        os.makedirs(self.recorder_dir / "calibration", exist_ok=True)
        # 将深度和彩色图像的形状改为600*1200（裁剪底部）
        depth_image_dir = self.recorder_dir / "depth"
        color_image_dir = self.recorder_dir / "rgb"
        point_dir = self.recorder_dir / "point"
        mask_dir = self.recorder_dir / "mask"
        # 获取所有深度图像路径并按文件名编号排序
        depth_paths = [f for f in sorted(depth_image_dir.iterdir(), key=lambda x: int(x.stem))]
        # 获取所有彩色图像路径并按文件名编号排序
        color_paths = [f for f in sorted(color_image_dir.iterdir(), key=lambda x: int(x.stem))]
        # 获取所有点云路径并按文件名编号排序
        point_paths = [f for f in sorted(point_dir.iterdir(), key=lambda x: int(x.stem))]
        # 获取所有掩码路径并按文件名编号排序
        mask_paths = [f for f in sorted(mask_dir.iterdir(), key=lambda x: int(x.stem))]

        # 保存数据总长度
        self.length = len(depth_paths)
        # 遍历所有帧，显示进度条
        for i in tqdm(range(self.length), desc="change size: "):
            # 注释掉的代码：使用PIL库进行裁剪（384*512）
            # Image.fromarray(np.asarray(Image.open(depth_paths[i]), dtype=np.uint16)[:384, 64:576]).save(depth_paths[i])
            # Image.fromarray(np.asarray(Image.open(color_paths[i]), dtype=np.uint8)[:384, 64:576, :]).save(color_paths[i])
            # 裁剪彩色图像的底部200像素（保留前600行）
            cv2.imwrite(str(color_paths[i]), cv2.imread(str(color_paths[i]))[:600, :, :])
            # 裁剪深度图像的底部200像素
            np.save(depth_paths[i], np.load(depth_paths[i])[:600, :])
            # 裁剪点云数据的底部200像素
            np.save(point_paths[i], np.load(point_paths[i])[:600, :])
            # 裁剪掩码数据的底部200像素
            np.save(mask_paths[i], np.load(mask_paths[i])[:600, :])
            # 保存标定矩阵到txt文件
            calibration_path = self.recorder_dir / "calibration" / f"{i:06}.txt"
            np.savetxt(str(calibration_path), self.intrinsic_matrix)

    # 设置元数据，保存到JSON文件
    def set_metadata(self):
        # 构建元数据字典，包含所有重要的相机参数和数据信息
        metadata = {
            "w": real_width,  # 实际图像宽度
            "h": real_height,  # 实际图像高度
            "dw": real_width,  # 深度图像宽度
            "dh": real_height,  # 深度图像高度
            "fps": self.FPS,  # 帧率
            "K": self.intrinsic_matrix.tolist(),  # 相机内参矩阵（转换为列表）
            "depth_scale": self.depth_scale,  # 深度比例因子
            "min_depth": DEPTH_MIN,  # 最小深度阈值
            "max_depth": DEPTH_MAX,  # 最大深度阈值
            "cameraType": 1,  # 相机类型（1表示RealSense）
            "dist_coef": self.dist_coef.tolist(),  # 畸变系数（转换为列表）
            "length": self.length  # 数据总长度（帧数）
        }
        # 将元数据保存为JSON文件，使用缩进格式化
        with open(self.recorder_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    # 开始录制，在独立线程中运行
    def start_record(self):
        # 循环录制，直到record_flag为False
        while self.record_flag:
            # 获取并保存当前帧
            flag = self.get_align_frame(self.frame_index)
            # 注释掉的代码：添加小延迟以降低CPU使用率
            # time.sleep(0.001)
            # time.sleep(0.1)
            # print(self.frame_index)
            # 如果保存成功，增加帧索引
            if flag: self.frame_index += 1
    
    # 停止录制
    def stop_record(self):
        # 设置录制标志为False，停止录制循环
        self.record_flag = False
        # 停止RealSense管道
        self.pipeline.stop()


# 主录制函数，用于交互式录制数据
def record():
    # 询问用户是否要录制数据
    if input("Do you want to record data? [y/n]: ") == "n":
        return 
    # 设置录制目录路径
    recorder_dir = RECORDER_DIR / "test"
    # 创建RecorderImage实例
    imagerecorder = RecorderImage(recorder_dir=recorder_dir)
    # 提示用户按任意键开始录制（绿色提示）
    input("\033[32mPress any key to Start.\033[0m")
    # 创建录制线程，运行start_record方法
    record_thread = threading.Thread(target=imagerecorder.start_record)
    # 启动录制线程
    record_thread.start()
    # 提示用户按任意键停止录制（红色提示）
    input("\033[31mRecording started. Press any key to stop.\033[0m")
    # 停止录制
    imagerecorder.stop_record()
    # 等待录制线程结束
    record_thread.join()
    # 修改图像尺寸（裁剪底部）
    imagerecorder.change_size()
    # 设置元数据
    imagerecorder.set_metadata()
    # 获取相机内参
    intrinsic = imagerecorder.intrinsic
    # 保存标定文件（包含焦距和主点坐标）
    with open(recorder_dir / "calib.txt", "w") as f:
        f.write(f'{intrinsic.fx} {intrinsic.fy} {intrinsic.ppx} {intrinsic.ppy}')
    # 注释掉的代码：可视化深度到彩色映射
    # imagerecorder.depth_to_color_vis()
    # 打印保存位置和数据长度
    print(f"All Images are save in {imagerecorder.recorder_dir}: depth / rgb, length is {len(os.listdir(imagerecorder.recorder_dir / 'depth'))}")

    # 删除录制器对象，释放资源
    del imagerecorder



# 主程序入口
if __name__ == "__main__":
    # 调用record函数开始录制流程
    record()
