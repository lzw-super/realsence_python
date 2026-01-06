import os

def generate_timestamp_file(out_dirs, output_filename="associations.txt"):
    results_dir = os.path.join(out_dirs, "results")
    rgb_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(results_dir) if f.startswith("depth") and f.endswith(".png")])

    assert len(rgb_files) == len(depth_files), "RGB 和 Depth 图像数量不匹配！"
    num_frames = len(rgb_files)

    output_path = os.path.join(out_dirs, output_filename)

    with open(output_path, 'w') as f:
        f.write("# timestamp rbg_file_path depth_file_path\n")
        
        for i in range(num_frames):
            timestamp = i / 6
            rgb_file = rgb_files[i]
            depth_file = depth_files[i]
            
            rgb_path = os.path.join(results_dir, rgb_file)
            depth_path = os.path.join(results_dir, depth_file)
            
            f.write(f"{timestamp:.3f} {rgb_path} {depth_path}\n")

    print(f"✅ 已生成时间戳文件: {output_path}")
    print(f"   共 {num_frames} 帧数据")

if __name__ == '__main__':
    out_dirs = r'E:\Users\lenovo\Desktop\realsense\realsence_python\examples\out\capture_20251230_160728'
    generate_timestamp_file(out_dirs)
