import pyrealsense2 as rs

def get_d455_serial_number():
    """
    获取连接的Intel RealSense D455相机序列号
    返回：str 相机序列号（若未检测到D455返回None）
    """
    # 创建上下文，枚举所有连接的设备
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("未检测到RealSense相机设备")
        return None
    
    # 遍历设备，筛选D455并获取序列号
    d455_serial = None
    for dev in ctx.devices:
        # 获取设备信息
        dev_info = dev.get_info(rs.camera_info.name)
        if "D455" in dev_info:
            d455_serial = dev.get_info(rs.camera_info.serial_number)
            break  # 若多个D455，可调整逻辑遍历所有
    
    if d455_serial is None:
        print("未检测到D455相机，当前连接的设备：")
        for dev in ctx.devices:
            print(f"设备名称：{dev.get_info(rs.camera_info.name)}, 序列号：{dev.get_info(rs.camera_info.serial_number)}")
    else:
        print(f"D455相机序列号：{d455_serial}")
    
    return d455_serial

# 调用示例
if __name__ == "__main__":
    serial_num = get_d455_serial_number()
    if serial_num:
        # 可将序列号写入配置文件/日志，或集成到你的数据集加载逻辑中
        print(f"已获取D455序列号：{serial_num}")