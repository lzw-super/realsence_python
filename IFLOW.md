# Intel RealSense Python 示例项目

## 项目概述

这是一个基于 Intel RealSense SDK 的 Python 示例代码集合，展示了如何使用 Python 包装器（pyrealsense2）与 Intel RealSense 深度相机（特别是 D400 系列）进行交互。该项目包含多个实用示例，涵盖了从基础的深度数据流获取到高级应用如多相机标定、以太网传输和自动校准等功能。

**主要技术栈：**
- Python 3.6+
- pyrealsense2 (Intel RealSense SDK Python 包装器)
- NumPy (数组处理)
- OpenCV (图像处理和可视化)

## 项目结构

```
realsence/
└── examples/
    ├── python-tutorial-1-depth.py        # 基础深度流获取教程
    ├── opencv_viewer_example.py          # 使用 OpenCV 渲染深度和彩色图像
    ├── align-depth2color.py              # 深度图与彩色图对齐
    ├── python-rs400-advanced-mode-example.py  # D400 高级模式配置
    ├── pybackend_example_1_general.py    # 后端接口控制设备
    ├── read_bag_example.py               # 读取 bag 文件
    ├── teset.py                          # 自定义测试文件（1280x720 分辨率）
    ├── depth_auto_calibration_example.py # D400 自动校准
    ├── numpy_to_frame.py                 # NumPy 数组转换为 pyrealsense 帧
    ├── align-with-software-device.py     # 使用软件设备进行流对齐
    ├── opencv_pointcloud_viewer.py       # OpenCV 点云可视化
    ├── pyglet_pointcloud_viewer.py       # PyGlet 点云可视化
    ├── export_ply_example.py             # 导出 PLY 点云文件
    ├── frame_queue_example.py            # 帧队列示例
    ├── depth_ucal_example.py             # 深度校准示例
    ├── d500_triggered_calibration.py     # D500 触发校准
    ├── box_dimensioner_multicam/         # 多相机尺寸测量
    └── ethernet_client_server/           # 以太网客户端/服务器
```

## 环境配置

### 依赖安装

```bash
pip install opencv-python numpy pyrealsense2
```

对于点云可视化示例，还需要：
```bash
pip install pyglet
```

### 系统要求
- Python 3.6 或更高版本
- Intel RealSense D400 系列深度相机
- USB 3.0 连接（推荐，某些功能需要）

## 构建和运行

### 运行基础示例

**获取深度流（ASCII 艺术显示）：**
```bash
python examples/python-tutorial-1-depth.py
```

**使用 OpenCV 显示深度和彩色图像：**
```bash
python examples/opencv_viewer_example.py
```

**深度图与彩色图对齐：**
```bash
python examples/align-depth2color.py
```

**自定义测试文件（1280x720 分辨率）：**
```bash
python examples/teset.py
```

### 运行高级示例

**自动校准（仅适用于 D400 系列）：**
```bash
python examples/depth_auto_calibration_example.py --exposure auto --onchip-speed medium
```

**多相机尺寸测量：**
```bash
python examples/box_dimensioner_multicam/box_dimensioner_multicam_demo.py
```

**以太网服务器端：**
```bash
python examples/ethernet_client_server/EtherSenseServer.py
```

**以太网客户端：**
```bash
python examples/ethernet_client_server/EtherSenseClient.py
```

## 开发约定

### 代码风格
- 示例代码遵循标准的 Python 编码规范（PEP 8）
- 使用 try-finally 块确保资源正确释放（pipeline.stop(), cv2.destroyAllWindows()）
- 添加适当的注释说明关键步骤

### 常用模式

**1. 管道初始化模式：**
```python
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
```

**2. 帧获取模式：**
```python
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
```

**3. NumPy 转换模式：**
```python
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())
```

**4. 深度图可视化模式：**
```python
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
```

### 设备检测
在需要彩色相机的示例中，代码会检测设备是否支持 RGB Camera：
```python
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
```

## 主要功能模块

### 1. 基础流获取
- `python-tutorial-1-depth.py`: 最简单的深度流获取示例
- `opencv_viewer_example.py`: 结合 OpenCV 的深度和彩色流显示

### 2. 图像处理
- `align-depth2color.py`: 深度图与彩色图对齐，用于背景移除
- `align-with-software-device.py`: 使用软件设备进行流对齐

### 3. 校准功能
- `depth_auto_calibration_example.py`: D400 系列自动校准（On-Chip、Focal Length、Tare Calibration）
- `d500_triggered_calibration.py`: D500 触发校准
- `depth_ucal_example.py`: 深度校准示例

### 4. 多相机应用
- `box_dimensioner_multicam/`: 使用多个相机计算物体尺寸（基于 Kabsch 算法）
- `ethernet_client_server/`: 通过以太网传输深度数据

### 5. 点云处理
- `opencv_pointcloud_viewer.py`: OpenCV 点云渲染器
- `pyglet_pointcloud_viewer.py`: PyGlet 点云渲染器
- `export_ply_example.py`: 导出 PLY 格式点云文件

### 6. 数据记录和回放
- `read_bag_example.py`: 读取 bag 文件
- `frame_queue_example.py`: 帧队列管理

## 注意事项

1. **设备要求**: 大多数示例需要支持深度和彩色的 RealSense 设备（如 D435、D455 等）
2. **USB 连接**: 某些功能（如自动校准）需要 USB 3.0 连接
3. **高级模式**: 自动校准示例需要启用相机的"Advanced Mode"
4. **分辨率**: 不同示例使用不同分辨率，根据需求调整
5. **资源释放**: 所有示例都确保在结束时正确释放资源（pipeline.stop()）

## 常见问题

### 如何检测连接的设备？
```python
ctx = rs.context()
devices = ctx.query_devices()
for device in devices:
    print(f"Device: {device.get_info(rs.camera_info.name)}")
```

### 如何获取深度值？
```python
distance = depth_frame.get_distance(x, y)
```

### 如何调整深度图可视化？
修改 `cv2.convertScaleAbs()` 中的 `alpha` 参数来调整深度图的对比度。

## 许可证
Apache 2.0