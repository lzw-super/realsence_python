## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2021 RealSense, Inc. All Rights Reserved.

#####################################################
##                   auto calibration              ##
#####################################################

# 导入命令行参数解析模块
import argparse
# 导入JSON处理模块
import json
# 导入系统模块
import sys
# 导入时间模块
import time

# 导入RealSense库
import pyrealsense2 as rs

__desc__ = """
此脚本演示自校准（UCAL）API的使用
"""

# 片上校准速度映射
occ_speed_map = {
    'very_fast': 0,  # 非常快
    'fast': 1,        # 快
    'medium': 2,      # 中等
    'slow': 3,        # 慢
    'wall': 4,        # 白墙模式
}
# Tare校准精度映射
tare_accuracy_map = {
    'very_high': 0,  # 非常高
    'high': 1,        # 高
    'medium': 2,      # 中等
    'low': 3,         # 低
}
# 扫描类型映射
scan_map = {
    'intrinsic': 0,   # 内参校准
    'extrinsic': 1,   # 外参校准
}
# 焦距调整侧映射
fl_adjust_map = {
    'right_only': 0,  # 仅右侧
    'both_sides': 1   # 两侧
}

# 创建RealSense上下文对象
ctx = rs.context()


def main(arguments=None):
    # 解析命令行参数
    args = parse_arguments(arguments)

    try:
        # 获取第一个连接的设备
        device = ctx.query_devices()[0]
    except IndexError:
        print('Device is not connected')
        sys.exit(1)

    # 验证前置条件：
    # 1. 此脚本仅适用于D400系列设备
    cam_name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "Unrecognized camera"
    if device.supports(rs.camera_info.product_line):
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        if device_product_line != 'D400':
            print(f'The example is intended for RealSense D400 Depth cameras, and is not', end =" ")
            print(f'applicable with {cam_name}')
            sys.exit(1)
    # 2. 此程序假设使用USB3连接类型
    #    如果是USB2连接，流配置文件需要重新调整
    if device.supports(rs.camera_info.usb_type_descriptor):
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        if not usb_type.startswith('3.'):
            print('The script is designed to run with USB3 connection type.')
            print('In order to enable it with USB2.1 mode the fps rates for the Focal Length and Ground Truth calculation stages should be re-adjusted')
            sys.exit(1)
    # 3. 高级模式应该被启用
    #    某些校准需要更改高级模式预设（取决于校准参数/类型）
    am_device = rs.rs400_advanced_mode(device)
    if not am_device or not am_device.is_enabled():
        print('Camera "Advanced Mode" must be enabled before calibrating.')
        sys.exit(1)
        # 要启用高级模式，使用 "am_device.toggle_advanced_mode(True)"。注意 - 这会导致相机重置（设置的选项将返回默认值）


    # 准备设备
    # 获取深度传感器
    depth_sensor = device.first_depth_sensor()
    # 关闭发射器（用于校准）
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    # 如果支持，关闭热补偿
    if depth_sensor.supports(rs.option.thermal_compensation):
        depth_sensor.set_option(rs.option.thermal_compensation, 0)
    # 设置曝光模式
    if args.exposure == 'auto':
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    else:
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        depth_sensor.set_option(rs.option.exposure, int(args.exposure))

    print("Starting UCAL...")
    try:
        # 推荐的程序顺序：片上校准 -> 焦距校准 -> Tare校准
        run_on_chip_calibration(args.onchip_speed, args.onchip_scan)
        run_focal_length_calibration((args.target_width, args.target_height), args.focal_adjustment)
        run_tare_calibration(args.tare_accuracy, args.tare_scan, args.tare_gt, (args.target_width, args.target_height))
    finally:
        # 恢复热补偿
        if depth_sensor.supports(rs.option.thermal_compensation):
            depth_sensor.set_option(rs.option.thermal_compensation, 1)
    print("UCAL finished successfully")


# 进度回调函数
def progress_callback(progress):
    print(f'\rProgress  {progress}% ... ', end ="\r")

# 运行片上校准
def run_on_chip_calibration(speed, scan):
    # 构建校准参数
    data = {
        'calib type': 0,  # 校准类型
        'speed': occ_speed_map[speed],  # 速度
        'scan parameter': scan_map[scan],  # 扫描参数
        'white_wall_mode': 1 if speed == 'wall' else 0,  # 白墙模式
    }

    # 将参数转换为JSON字符串
    args = json.dumps(data)

    # 配置深度流：256x144分辨率，Z16格式，90帧每秒
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
    # 创建管道
    pipe = rs.pipeline(ctx)
    pp = pipe.start(cfg)
    # 等待第一帧
    pipe.wait_for_frames()
    # 获取设备
    dev = pp.get_device()

    try:

        print('Starting On-Chip calibration...')
        print(f'\tSpeed:\t{speed}')
        print(f'\tScan:\t{scan}')
        # 将设备转换为自动校准设备
        adev = dev.as_auto_calibrated_device()
        # 运行片上校准，超时30秒
        table, health = adev.run_on_chip_calibration(args, progress_callback, 30000)
        print('On-Chip calibration finished')
        print(f'\tHealth: {health}')
        # 设置校准表
        adev.set_calibration_table(table)
        # 写入校准数据到相机
        adev.write_calibration()
    finally:
        # 停止管道
        pipe.stop()


# 运行焦距校准
def run_focal_length_calibration(target_size, adjust_side):
    number_of_images = 25  # 需要捕获的图像数量
    timeout_s = 30  # 超时时间（秒）

    # 配置红外流：1280x720分辨率，Y8格式，30帧每秒
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    # 创建帧队列，用于存储捕获的帧
    lq = rs.frame_queue(capacity=number_of_images, keep_frames=True)
    rq = rs.frame_queue(capacity=number_of_images, keep_frames=True)

    counter = 0
    flags = [False, False]

    # 帧回调函数
    def cb(frame):
        nonlocal counter, flags
        if counter > number_of_images:
            return
        # 遍历帧集中的每一帧
        for f in frame.as_frameset():
            p = f.get_profile()
            # 左红外相机（索引1）
            if p.stream_index() == 1:
                lq.enqueue(f)
                flags[0] = True
            # 右红外相机（索引2）
            if p.stream_index() == 2:
                rq.enqueue(f)
                flags[1] = True
            # 如果两个相机都捕获到帧，计数器加1
            if all(flags):
                counter += 1
        flags = [False, False]

    # 创建管道并启动，传入回调函数
    pipe = rs.pipeline(ctx)
    pp = pipe.start(cfg, cb)
    dev = pp.get_device()

    try:
        print('Starting Focal-Length calibration...')
        print(f'\tTarget Size:\t{target_size}')
        print(f'\tSide Adjustment:\t{adjust_side}')
        stime = time.time()
        # 等待捕获足够的帧
        while counter < number_of_images:
            time.sleep(0.5)
            if timeout_s < (time.time() - stime):
                raise RuntimeError(f"Failed to capture {number_of_images} frames in {timeout_s} seconds, got only {counter} frames")

        # 将设备转换为自动校准设备
        adev = dev.as_auto_calibrated_device()
        # 运行焦距校准
        table, ratio, angle = adev.run_focal_length_calibration(lq, rq, target_size[0], target_size[1],
                                                                fl_adjust_map[adjust_side],progress_callback)
        print('Focal-Length calibration finished')
        print(f'\tRatio:\t{ratio}')
        print(f'\tAngle:\t{angle}')
        # 设置校准表
        adev.set_calibration_table(table)
        # 写入校准数据到相机
        adev.write_calibration()
    finally:
        # 停止管道
        pipe.stop()


# 运行Tare校准
def run_tare_calibration(accuracy, scan, gt, target_size):
    # 构建校准参数
    data = {'scan parameter': scan_map[scan],
            'accuracy': tare_accuracy_map[accuracy],
            }
    args = json.dumps(data)

    print('Starting Tare calibration...')
    # 计算或设置目标距离（ground truth）
    if gt == 'auto':
        target_z = calculate_target_z(target_size)
    else:
        target_z = float(gt)

    # 配置深度流：256x144分辨率，Z16格式，90帧每秒
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
    pipe = rs.pipeline(ctx)
    pp = pipe.start(cfg)
    pipe.wait_for_frames()
    dev = pp.get_device()

    try:
        print(f'\tGround Truth:\t{target_z}')
        print(f'\tAccuracy:\t{accuracy}')
        print(f'\tScan:\t{scan}')
        # 将设备转换为自动校准设备
        adev = dev.as_auto_calibrated_device()
        # 运行Tare校准，超时30秒
        table, health = adev.run_tare_calibration(target_z, args, progress_callback, 30000)
        print('Tare calibration finished')
        # 设置校准表
        adev.set_calibration_table(table)
        # 写入校准数据到相机
        adev.write_calibration()

    finally:
        # 停止管道
        pipe.stop()


# 计算目标距离（ground truth）
def calculate_target_z(target_size):
    number_of_images = 50  # 所需的帧数为10+
    timeout_s = 30

    # 配置红外流：1280x720分辨率，Y8格式，30帧每秒
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)

    # 创建帧队列
    q = rs.frame_queue(capacity=number_of_images, keep_frames=True)
    # 帧队列q2、q3应该留空。为未来增强做准备。
    q2 = rs.frame_queue(capacity=number_of_images, keep_frames=True)
    q3 = rs.frame_queue(capacity=number_of_images, keep_frames=True)

    counter = 0

    # 帧回调函数
    def cb(frame):
        nonlocal counter
        if counter > number_of_images:
            return
        for f in frame.as_frameset():
            q.enqueue(f)
            counter += 1

    # 创建管道并启动，传入回调函数
    pipe = rs.pipeline(ctx)
    pp = pipe.start(cfg, cb)
    dev = pp.get_device()

    try:
        stime = time.time()
        # 等待捕获足够的帧
        while counter < number_of_images:
            time.sleep(0.5)
            if timeout_s < (time.time() - stime):
                raise RuntimeError(f"Failed to capture {number_of_images} frames in {timeout_s} seconds, got only {counter} frames")

        # 将设备转换为自动校准设备
        adev = dev.as_auto_calibrated_device()
        print('Calculating distance to target...')
        print(f'\tTarget Size:\t{target_size}')
        # 计算目标距离
        target_z = adev.calculate_target_z(q, q2, q3, target_size[0], target_size[1])
        print(f'Calculated distance to target is {target_z}')
    finally:
        # 停止管道
        pipe.stop()

    return target_z


# 解析命令行参数
def parse_arguments(args):
    parser = argparse.ArgumentParser(description=__desc__)

    # 曝光参数
    parser.add_argument('--exposure', default='auto', help="Exposure value or 'auto' to use auto exposure")
    # 目标尺寸参数
    parser.add_argument('--target-width', default=175, type=int, help='The target width')
    parser.add_argument('--target-height', default=100, type=int, help='The target height')

    # 片上校准参数
    parser.add_argument('--onchip-speed', default='medium', choices=occ_speed_map.keys(), help='On-Chip speed')
    parser.add_argument('--onchip-scan', choices=scan_map.keys(), default='intrinsic', help='On-Chip scan')

    # 焦距校准参数
    parser.add_argument('--focal-adjustment', choices=fl_adjust_map.keys(), default='right_only',
                        help='Focal-Length adjustment')

    # Tare校准参数
    parser.add_argument('--tare-gt', default='auto',
                        help="Target ground truth, set 'auto' to calculate using target size"
                             "or the distance to target in mm to use a custom value")
    parser.add_argument('--tare-accuracy', choices=tare_accuracy_map.keys(), default='medium', help='Tare accuracy')
    parser.add_argument('--tare-scan', choices=scan_map.keys(), default='intrinsic', help='Tare scan')

    return parser.parse_args(args)


if __name__ == '__main__':
    main()
