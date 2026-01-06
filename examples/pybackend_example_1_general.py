## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 RealSense, Inc. All Rights Reserved.

###############################################
## pybackend 示例 #1 - 概述 ##
###############################################

# 首先导入库
import pybackend2 as rs  # pybackend2 是 RealSense 的底层后端库，提供对 USB 设备的直接访问
import time

def on_frame(profile, f):
    """帧回调函数，当接收到新帧时被调用"""
    print ("Received %d bytes" % f.frame_size)  # 打印接收到的帧大小（字节数）

    # 访问图像像素数据
    p = f.pixels  # 获取像素数据指针
    print ("First 10 bytes are: ")  # 打印前10个字节的十六进制值
    for i in range(10):
        print (hex(p[i]))
    print

try:
    # 构建设备
    backend = rs.create_backend()  # 创建后端对象，用于管理 USB 设备
    infos = backend.query_uvc_devices()  # 查询所有连接的 UVC（USB Video Class）设备
    print("There are %d connected UVC devices" % len(infos))
    if len(infos) is 0: exit(1)  # 如果没有设备，退出程序
    info = infos[2]  # 获取第三个设备的信息（索引2）
    dev = backend.create_uvc_device(info)  # 创建 UVC 设备对象

    # 打印设备信息：VID（供应商ID）、PID（产品ID）、MI（接口号）、UID（唯一ID）
    print ("VID=%d, PID=%d, MI=%d, UID=%s" % (info.vid, info.pid, info.mi, info.unique_id))

    # 打开电源
    print ("Move device to D0 power state...")  # D0 是完全工作状态
    dev.set_power_state(rs.D0)  # 设置设备电源状态为 D0（完全工作）

    # 配置流
    print ("Print list of UVC profiles supported by the device...")  # 打印设备支持的所有 UVC 配置文件
    profiles = dev.get_profiles()  # 获取设备支持的所有配置文件（分辨率、帧率、格式等）
    for p in profiles:
        print (p)
        # 保存 IR VGA 设置供后续使用
        if p.width == 640 and p.height == 480 and p.fps == 30 and p.format == 1196574041:
            vga = p  # 保存 640x480@30fps 的 IR 配置文件
    first = profiles[0]  # 获取第一个配置文件

    print ("Negotiate Probe-Commit for first profile")  # 协商并提交第一个配置文件
    dev.probe_and_commit(first, on_frame)  # 探测并提交配置，设置帧回调函数

    # XU（扩展单元）控制 - RealSense 特有的控制接口
    # 定义深度扩展单元：GUID 是 RealSense 深度传感器的唯一标识符
    xu = rs.extension_unit(0, 3, 2, rs.guid("C9606CCB-594C-4D25-af47-ccc496435995"))
    dev.init_xu(xu)  # 初始化深度扩展单元
    ae = dev.get_xu(xu, 0xB, 1)  # 获取 XU 值。参数：XU 对象，控制编号，字节数
    print ("Auto Exposure option is:", ae)
    print ("Setting Auto Exposure option to new value")
    dev.set_xu(xu, 0x0B, [0x00])  # 设置 XU 值。参数：XU 对象，控制编号，字节列表
    ae = dev.get_xu(xu, 0xB, 1)
    print ("New Auto Exposure setting is:", ae)

    # PU（处理单元）控制 - 标准 UVC 控制接口
    gain = dev.get_pu(rs.option.gain)  # 获取增益值
    print ("Gain = %d" % gain)
    print ("Setting gain to new value")
    dev.set_pu(rs.option.gain, 32)  # 设置增益值为 32
    gain = dev.get_pu(rs.option.gain)
    print ("New gain = %d" % gain)

    # 开始流传输
    print ("Start listening for callbacks (for all pins)...")  # 开始监听回调（所有引脚）
    dev.start_callbacks()  # 启动回调机制

    print ("Start streaming (from all pins)...")  # 开始流传输（所有引脚）
    dev.stream_on()  # 开启数据流

    print ("Wait for 5 seconds while frames are expected:")  # 等待5秒以接收帧
    time.sleep(5)

    # 关闭设备
    print ("Stop listening for new callbacks...")  # 停止监听新回调
    dev.stop_callbacks()  # 停止回调机制

    print ("Close the specific pin...")  # 关闭特定引脚
    dev.close(first)  # 关闭第一个配置文件的引脚

    # 保存帧到磁盘
    def save_frame(profile, f):
        """保存帧为 PNG 文件的回调函数"""
        f.save_png("pybackend_example_1_general_depth_frame.png", 640, 480, f.frame_size / (640*480))
        # 参数：文件名，宽度，高度，步长（每行字节数）

    print ("Saving an IR VGA frame using profile:", vga)
    dev.probe_and_commit(vga, save_frame)  # 使用 VGA 配置文件并设置保存回调

    dev.set_xu(xu, 0x0B, [0x01])  # 重新开启自动曝光
    dev.start_callbacks()  # 启动回调
    dev.stream_on()  # 开启流
    time.sleep(1)  # 等待1秒以捕获帧
    dev.close(vga)  # 关闭 VGA 配置文件

    print ("Move device to D3")  # D3 是睡眠状态
    dev.set_power_state(rs.D3)  # 设置设备电源状态为 D3（睡眠）
    pass
except Exception as e:
    print (e)  # 打印异常信息
    pass