import base64
from pathlib import Path


def extract_base64_from_file(input_file, output_file):
    try:
        # 检查输入文件是否存在
        if not Path(input_file).exists():
            print(f"错误: 输入文件 '{input_file}' 不存在")
            return
        
        # 读取并清理内容
        with open(input_file, 'r') as f:
            content = f.read().strip()
        
        # 检查内容是否为空
        if not content:
            print("错误: 文件内容为空")
            return
        
        # 处理可能的 data URI 前缀
        if content.startswith('data:'):
            parts = content.split(',', 1)
            if len(parts) == 2:
                content = parts[1]
        
        # 修复 base64 字符串的 padding
        padding = len(content) % 4
        if padding:
            content += '=' * (4 - padding)
        
        # 尝试解码 base64 数据
        try:
            binary_data = base64.b64decode(content)
        except Exception as e:
            print(f"Base64 解码失败: {e}")
            return
        
        # 解码后数据有效性检查
        if len(binary_data) < 1024:
            print("警告: 解码后的数据过小，可能无效")
        
        # 确保输出目录存在
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入解码后的数据
        with open(output_file, 'wb') as f:
            f.write(binary_data)
        
        print(f"解码成功，文件已保存为: {output_file}")
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 使用示例
if __name__ == '__main__':
    input_file = 'full_output_base64.txt'
    output_file = 'decoded_output.bin'
    extract_base64_from_file(input_file, output_file) 