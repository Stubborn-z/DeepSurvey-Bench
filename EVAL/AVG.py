#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

def get_types_info():
    """从 eval_structure.md 中提取类型信息"""
    types_info = {
        'hsr': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'her': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'outline': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'citationrecall': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'citationprecision': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'paperold': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'paperour': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'rouge': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'bleu': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'recallak': ['a', 'a1', 'a2', 'f', 'f1', 'f2'],
        'recallpref': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'lourele': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
    }
    return types_info

def get_type_key(data):
    """获取数据行的类型键（排除 name）"""
    keys = [k for k in data.keys() if k != 'name']
    # 对于 paperour 类型，需要同时包含 paperour 和 reason
    if 'paperour' in keys:
        return ('paperour', 'reason')
    return tuple(sorted(keys))

def load_eval_file(file_path):
    """加载 eval.jsonl 文件，返回按类型和 name 组织的数据"""
    if not os.path.exists(file_path):
        return {}
    
    data_dict = {}  # {(type_key, name): value}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'name' not in data:
                        continue
                    
                    name = data.get('name')
                    type_key = get_type_key(data)
                    
                    # 获取值（排除 name）
                    value = {k: v for k, v in data.items() if k != 'name'}
                    if len(value) == 1:
                        value = list(value.values())[0]
                    else:
                        # 多个键的情况（如 paperour + reason）
                        # 对于 paperour 类型，只取 paperour 键的值（数字列表），忽略 reason（字符串列表）
                        if 'paperour' in value:
                            value = value['paperour']
                        else:
                            value = value
                    
                    data_dict[(type_key, name)] = value
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  警告: 读取文件 {file_path} 时出错: {e}")
        return {}
    
    return data_dict

def match_type_key(type_key, type_name):
    """检查 type_key 是否匹配 type_name"""
    if isinstance(type_key, tuple):
        type_str = ','.join(sorted(type_key))
    else:
        type_str = str(type_key)
    
    if type_name == 'paperour':
        return 'paperour' in type_str and 'reason' in type_str
    else:
        return type_name in type_str

def calculate_average(values, is_her=False):
    """计算平均值
    values: 值列表
    is_her: 如果是 her 类型，需要过滤掉 0.0 的值
    """
    if is_her:
        # 对于 her，过滤掉 0.0 的值
        filtered_values = [v for v in values if v != 0.0]
        if not filtered_values:
            return None
        return sum(filtered_values) / len(filtered_values)
    else:
        return sum(values) / len(values)

def calculate_list_average(list_values, is_her=False):
    """计算列表的平均值（对每一项分别计算）
    list_values: 列表的列表，例如 [[1,2,3], [4,5,6], [7,8,9]]
    """
    if not list_values:
        return None
    
    # 检查所有列表长度是否一致
    lengths = [len(lst) for lst in list_values]
    if len(set(lengths)) != 1:
        print(f"    警告: 列表长度不一致: {lengths}")
        return None
    
    # 对每一项分别计算平均值
    result = []
    for i in range(lengths[0]):
        column_values = [lst[i] for lst in list_values]
        if is_her:
            # 对于 her，过滤掉 0.0
            column_values = [v for v in column_values if v != 0.0]
            if not column_values:
                result.append(0.0)
                continue
        avg = sum(column_values) / len(column_values)
        result.append(avg)
    
    return result

def process_averages():
    """处理所有 eval.jsonl 文件，计算平均值"""
    # 获取脚本所在目录的父目录（EVAL）
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # EVAL 目录
    out_dir = base_dir / 'OUT'
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / 'AVG.jsonl'
    
    # 删除已存在的文件，重新生成
    if output_file.exists():
        output_file.unlink()
        print(f"已删除旧文件: {output_file}")
    
    # 获取类型信息
    types_info = get_types_info()
    
    # 加载所有 M=1 到 20 的文件
    all_data = {}  # {M: {(type_key, name): value}}
    existing_files = []
    
    print("=" * 60)
    print("加载文件...")
    print("=" * 60)
    
    for M in range(1, 21):
        eval_file = base_dir / f't{M}' / 'eval.jsonl'
        if eval_file.exists():
            data = load_eval_file(eval_file)
            all_data[M] = data
            existing_files.append(M)
            print(f"  t{M}/eval.jsonl: 加载成功 ({len(data)} 条记录)")
        else:
            print(f"  t{M}/eval.jsonl: 文件不存在")
    
    print(f"\n共加载 {len(existing_files)} 个文件: {existing_files}")
    print("=" * 60)
    
    # 对于每个类型和每个 name 取值，计算平均值
    results = []
    
    print("\n计算平均值...")
    print("=" * 60)
    
    for type_name, name_list in types_info.items():
        print(f"\n类型: {type_name}")
        print("-" * 60)
        
        for name in name_list:
            print(f"  处理 name={name}...")
            
            # 收集所有 M 中该类型和 name 的值
            values = []
            found_in = []
            missing_in = []
            
            for M in range(1, 21):
                if M not in all_data:
                    missing_in.append(M)
                    continue
                
                data = all_data[M]
                found = False
                
                for (type_key, n), value in data.items():
                    if n == name and match_type_key(type_key, type_name):
                        values.append((M, value))
                        found_in.append(M)
                        found = True
                        break
                
                if not found:
                    missing_in.append(M)
            
            # 检查是否在所有文件中都能找到
            if len(found_in) < 20:
                print(f"    ✗ 缺项: 仅在 {len(found_in)} 个文件中找到，缺失: {missing_in}")
                continue
            
            # 计算平均值
            print(f"    ✓ 在 20 个文件中都找到")
            
            # 判断值的类型
            if not values:
                print(f"    ✗ 错误: 未找到值")
                continue
            
            first_value = values[0][1]
            
            # 判断是否为列表
            if isinstance(first_value, list):
                # 列表类型（如 recallak, rouge, outline 等）
                list_values = [v[1] for v in values]
                is_her = (type_name == 'her')
                avg_result = calculate_list_average(list_values, is_her)
                
                if avg_result is None:
                    print(f"    ✗ 计算平均值失败")
                    continue
                
                # 显示计算过程
                print(f"    计算过程:")
                print(f"      找到 {len(values)} 个值")
                if is_her:
                    non_zero_count = sum(1 for v in list_values for item in v if item != 0.0)
                    print(f"      (her 类型，已过滤 0.0 值)")
                print(f"      平均值: {avg_result}")
                
                result_item = {
                    "name": name,
                    type_name: avg_result
                }
                results.append(result_item)
                
            elif isinstance(first_value, (int, float)):
                # 数字类型（如 hsr, her, bleu 等）
                num_values = [v[1] for v in values]
                is_her = (type_name == 'her')
                avg_result = calculate_average(num_values, is_her)
                
                if avg_result is None:
                    print(f"    ✗ 计算平均值失败")
                    continue
                
                # 显示计算过程
                print(f"    计算过程:")
                print(f"      找到 {len(values)} 个值: {num_values}")
                if is_her:
                    filtered = [v for v in num_values if v != 0.0]
                    print(f"      (her 类型，已过滤 0.0 值，从 {len(num_values)} 个值中取 {len(filtered)} 个)")
                    print(f"      过滤后的值: {filtered}")
                print(f"      平均值: {avg_result}")
                
                result_item = {
                    "name": name,
                    type_name: avg_result
                }
                results.append(result_item)
            
            elif isinstance(first_value, dict):
                # 字典类型（如 paperour + reason）
                print(f"    ⚠ 跳过字典类型（暂不支持）")
                continue
            else:
                # 字符串类型，跳过
                print(f"    ⚠ 跳过字符串类型")
                continue
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果...")
    print("=" * 60)
    
    if results:
        # 写入模式（重新生成）
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✓ 已生成 {len(results)} 条结果到 {output_file}")
    else:
        print("✗ 没有可保存的结果")

if __name__ == '__main__':
    process_averages()

