#!/usr/bin/env python3
"""
从 arxiv_433.jsonl 中提取 cite_counts，计算平均值并保存到 DBcite.json
"""
import json
import os
from pathlib import Path

def is_valid_number(value):
    """判断是否为有效的数字（正数或0）"""
    if value is None:
        return False
    try:
        num = float(value)
        # 只接受非负数
        return num >= 0 and not (num != num)  # 排除 NaN
    except (TypeError, ValueError):
        return False

def process_arxiv_data(input_file, output_file):
    """
    处理 arxiv_433.jsonl 文件，提取 cite_counts 并计算平均值
    
    Args:
        input_file: 输入的 jsonl 文件路径
        output_file: 输出的 json 文件路径
    """
    cite_counts_list = []
    valid_count = 0
    invalid_count = 0
    
    print(f"正在读取文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return
    
    # 读取并处理每一行
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 提取 cite_counts
                cite_count = None
                if 'meta_info' in data and isinstance(data['meta_info'], dict):
                    cite_count = data['meta_info'].get('cite_counts')
                
                # 记录所有值（包括无效的）
                cite_counts_list.append(cite_count)
                
                # 检查是否为有效数字
                if is_valid_number(cite_count):
                    valid_count += 1
                else:
                    invalid_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                cite_counts_list.append(None)
                invalid_count += 1
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理出错: {e}")
                cite_counts_list.append(None)
                invalid_count += 1
    
    # 计算有效数字的平均值
    valid_numbers = []
    for count in cite_counts_list:
        if is_valid_number(count):
            valid_numbers.append(float(count))
    
    if valid_numbers:
        average = sum(valid_numbers) / len(valid_numbers)
    else:
        average = None
        print("警告: 没有找到有效的 cite_counts 数字")
    
    # 准备输出数据
    result = {
        "cite_counts_list": cite_counts_list,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "average": average
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成!")
    print(f"总记录数: {len(cite_counts_list)}")
    print(f"有效数字: {valid_count}")
    print(f"无效数字: {invalid_count}")
    if average is not None:
        print(f"平均值: {average:.2f}")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    # 设置文件路径
    base_dir = Path(__file__).parent
    input_file = base_dir / "arxiv_dataset" / "arxiv_433.jsonl"
    output_file = base_dir.parent / "OUT" / "DBcite.json"
    
    process_arxiv_data(str(input_file), str(output_file))

