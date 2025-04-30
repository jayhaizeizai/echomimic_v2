import numpy as np
import os
import glob

# 定义输入输出路径
input_dir = "assets/halfbody_demo/pose/01/"
output_dir = "tmp/pose/01_512_512"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有npy文件
npy_files = glob.glob(os.path.join(input_dir, "*.npy"))

# 处理每个文件
for input_path in npy_files:
    # 获取文件名（不含路径和扩展名）
    file_name = os.path.basename(input_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    # 定义输出文件路径 - NPY格式
    npy_output_path = os.path.join(output_dir, f"{file_name_without_ext}.npy")
    
    print(f"处理文件: {input_path} -> {npy_output_path}")
    
    # 加载并修改数据
    data = np.load(input_path, allow_pickle=True).item()
    
    # 修改分辨率为192x192
    if "draw_pose_params" in data:
        data["draw_pose_params"] = [512, 512, 0, 512, 0, 512]
    
    # 保存为NPY格式
    np.save(npy_output_path, data, allow_pickle=True)

print(f"已完成所有文件处理，共处理 {len(npy_files)} 个文件")
print(f"输出目录: {output_dir}")
print(f"- NPY文件已保存在上述目录下")