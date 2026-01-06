# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 RealSense, Inc. All Rights Reserved.

"""
OpenCV和NumPy点云软件渲染器

此示例主要用于演示和教育目的。
它无法提供硬件加速所能达到的质量或性能。

使用方法:
------
鼠标:
    左键拖动：围绕枢轴点（粗小轴）旋转
    右键拖动：平移视图
    滚轮：缩放

键盘:
    [p]     暂停
    [r]     重置视图
    [d]     循环切换降采样值
    [z]     切换点缩放
    [c]     切换颜色源
    [s]     保存PNG (./out.png)
    [e]     导出点云到ply (./out.ply)
    [q\ESC] 退出
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

class AppState:
    """应用程序状态管理类，用于跟踪视图参数和用户交互状态"""

    def __init__(self, *args, **kwargs):
        """初始化应用程序状态"""
        self.WIN_NAME = 'RealSense'  # 窗口名称
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)  # 俯仰角和偏航角（弧度）
        self.translation = np.array([0, 0, -1], dtype=np.float32)  # 平移向量
        self.distance = 2  # 观察距离
        self.prev_mouse = 0, 0  # 上一次鼠标位置
        self.mouse_btns = [False, False, False]  # 鼠标按钮状态 [左, 右, 中]
        self.paused = False  # 暂停状态
        self.decimate = 1  # 降采样级别
        self.scale = True  # 是否缩放点云
        self.color = True  # 是否使用彩色

    def reset(self):
        """重置视图到初始状态"""
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        """计算旋转矩阵（俯仰和偏航的组合）"""
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))  # 绕X轴旋转（俯仰）
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))  # 绕Y轴旋转（偏航）
        return np.dot(Ry, Rx).astype(np.float32)  # 组合旋转矩阵

    @property
    def pivot(self):
        """计算枢轴点位置（旋转中心）"""
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()  # 创建全局应用状态对象

# 配置深度和彩色流
pipeline = rs.pipeline()  # 创建RealSense管道对象
config = rs.config()  # 创建配置对象

# 获取管道包装器和配置文件
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()  # 获取设备对象

# 检查设备是否支持RGB相机
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 启用深度流（Z16格式，30fps）和彩色流（BGR8格式，30fps）
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# 启动流
pipeline.start(config)

# 获取流配置文件和相机内参
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# 处理模块
pc = rs.pointcloud()  # 点云处理模块
decimate = rs.decimation_filter()  # 降采样滤波器
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)  # 设置降采样强度
colorizer = rs.colorizer()  # 深度图着色器


def mouse_cb(event, x, y, flags, param):
    """鼠标回调函数，处理鼠标交互事件"""

    # 左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    # 左键释放
    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    # 右键按下
    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    # 右键释放
    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    # 中键按下
    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    # 中键释放
    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    # 鼠标移动事件
    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]  # 计算鼠标移动增量

        # 左键拖动：旋转视图
        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2  # 调整偏航角
            state.pitch -= float(dy) / h * 2  # 调整俯仰角

        # 右键拖动：平移视图
        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        # 中键拖动：缩放视图
        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    # 鼠标滚轮事件
    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)  # 保存当前鼠标位置

# 创建窗口并设置鼠标回调
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """将3D向量数组投影到2D平面"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w  # 视图宽高比

    # 忽略除零错误（无效深度）
    with np.errstate(divide='ignore', invalid='ignore'):
        # 透视投影：除以z坐标并缩放到屏幕坐标
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # 近裁剪面：过滤掉太近的点
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """对向量数组应用视图变换（旋转和平移）"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """在3D空间中绘制从pt1到pt2的线段"""
    p0 = project(pt1.reshape(-1, 3))[0]  # 投影起点
    p1 = project(pt2.reshape(-1, 3))[0]  # 投影终点
    if np.isnan(p0).any() or np.isnan(p1).any():
        return  # 如果任一投影无效，则不绘制
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])  # 图像边界矩形
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)  # 裁剪到图像边界
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)  # 绘制抗锯齿线


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """在xz平面上绘制网格"""
    pos = np.array(pos)
    s = size / float(n)  # 网格单元大小
    s2 = 0.5 * size  # 网格半大小
    # 绘制平行于x轴的线
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    # 绘制平行于z轴的线
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """绘制3D坐标轴（RGB对应XYZ）"""
    # Z轴（红色）
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    # Y轴（绿色）
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    # X轴（蓝色）
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """绘制相机的视锥体"""
    orig = view([0, 0, 0])  # 相机原点
    w, h = intrinsics.width, intrinsics.height

    # 在不同距离绘制视锥体截面
    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)  # 像素到3D点
            line3d(out, orig, view(p), color)  # 从原点绘制射线
            return p

        # 获取四个角点
        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        # 绘制四条边
        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """绘制点云，可选择使用画家算法（从后往前排序）"""
    if painter:
        # 画家算法：按z坐标从后往前排序点

        # 获取按z坐标反向排序的索引（在视图空间中）
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    # 根据降采样级别缩放点云
    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj现在包含2D图像坐标
    j, i = proj.astype(np.uint32).T

    # 创建掩码以忽略超出边界的索引
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # 使用相同的索引对纹理坐标进行排序
        # 纹理坐标范围[0..1]，相对于左上角像素角点
        # 乘以尺寸并加0.5以居中
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # 裁剪纹理坐标到图像范围
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # 执行UV映射（纹理采样）
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)  # 输出图像缓冲区

# 主循环
while True:
    # 获取相机数据
    if not state.paused:
        # 等待一帧同步的深度和彩色帧
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()  # 获取深度帧
        color_frame = frames.get_color_frame()  # 获取彩色帧

        # 应用降采样滤波器
        depth_frame = decimate.process(depth_frame)

        # 获取新的内参（可能被降采样改变）
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # 将帧数据转换为NumPy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 生成深度图的彩色可视化
        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        # 根据状态选择颜色源
        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        # 计算点云
        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)  # 将彩色帧映射到点云

        # 将点云数据转换为NumPy数组
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz坐标
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv纹理坐标

    # 渲染
    now = time.time()

    out.fill(0)  # 清空输出图像

    # 绘制参考网格、视锥体和坐标轴
    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    # 绘制点云
    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        # 如果需要缩放，先绘制到临时缓冲区再缩放
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    # 如果正在交互，绘制枢轴点坐标轴
    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    # 计算帧时间
    dt = time.time() - now

    # 更新窗口标题，显示分辨率、FPS和状态
    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    # 显示图像
    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    # 键盘事件处理
    if key == ord("r"):
        state.reset()  # 重置视图

    if key == ord("p"):
        state.paused ^= True  # 切换暂停状态

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3  # 循环切换降采样级别
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True  # 切换点缩放

    if key == ord("c"):
        state.color ^= True  # 切换颜色源

    if key == ord("s"):
        cv2.imwrite('./out.png', out)  # 保存当前视图为PNG

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)  # 导出点云为PLY文件

    # 退出条件：ESC或q键，或窗口关闭
    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# 停止流
pipeline.stop()
