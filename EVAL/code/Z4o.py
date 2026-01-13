#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

def calculate_average(values):
    """计算平均值"""
    if not values:
        return None
    return sum(values) / len(values)

def calculate_list_average(list_values):
    """计算列表的平均值（对每一项分别计算）"""
    if not list_values:
        return None
    
    # 检查所有列表长度是否一致
    lengths = [len(lst) for lst in list_values]
    if len(set(lengths)) != 1:
        print(f"    警告: 列表长度不一致: {lengths}")
        # 使用最小长度
        min_length = min(lengths)
        list_values = [lst[:min_length] for lst in list_values]
    
    # 对每一项分别计算平均值
    result = []
    for i in range(len(list_values[0])):
        column_values = [lst[i] for lst in list_values]
        avg = sum(column_values) / len(column_values)
        result.append(avg)
    
    return result

def process_z4o_data():
    """处理所有 eval.jsonl 文件中 name 以 Z4o 结尾的数据"""
    # 获取脚本所在目录的父目录（EVAL）
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # EVAL 目录
    out_dir = base_dir / 'OUT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出目录
    avg_file = out_dir / 'AVG.jsonl'
    paperz4o_dir = out_dir / 'PAPERZ4o'
    paperz4o_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空 PAPERZ4o 目录中的旧 JSON 文件
    old_files = list(paperz4o_dir.glob('*.json'))
    if old_files:
        for old_file in old_files:
            old_file.unlink()
        print(f"已删除 {len(old_files)} 个旧 PAPERZ4o 文件")
    
    print("=" * 60)
    print("处理 name 以 Z4o 结尾的数据...")
    print("=" * 60)
    
    # 收集所有 Z4o 的数据
    # 结构: {name: {type_key: {M: value}}}
    all_z4o_data = defaultdict(lambda: defaultdict(dict))
    # 保存详细数据用于写入 PAPERZ4o
    detailed_data = defaultdict(dict)  # {(M, name): data}
    
    for M in range(1, 21):
        eval_file = base_dir / f't{M}' / 'eval.jsonl'
        
        if not eval_file.exists():
            print(f"  t{M}/eval.jsonl: 文件不存在")
            continue
        
        print(f"\n处理 t{M}/eval.jsonl...")
        found_count = 0
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        name = data.get('name', '')
                        
                        # 只处理以 Z4o 结尾的 name
                        if not name.endswith('Z4o'):
                            continue
                        
                        # 获取类型键（排除 name）
                        keys = [k for k in data.keys() if k != 'name']
                        if not keys:
                            continue
                        
                        # 只处理 outline, paperold, paperour
                        valid_keys = ['outline', 'paperold', 'paperour']
                        has_valid_key = False
                        for key in keys:
                            if key not in valid_keys:
                                continue
                            
                            value = data[key]
                            all_z4o_data[name][key][M] = value
                            found_count += 1
                            has_valid_key = True
                            
                            # 保存完整数据用于写入 PAPERZ4o（为每个类型创建文件：tM<name><type>.json）
                            # 例如：t1aZ4o_outline.json, t1aZ4o_paperold.json, t1aZ4o_paperour.json
                            file_key = (M, name, key)
                            detailed_data[file_key] = data
                        
                        print(f"    ✓ 找到: {name} - {', '.join([k for k in keys if k in valid_keys])}")
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"    ✗ 处理行时出错: {e}")
                        continue
        
        except Exception as e:
            print(f"  ✗ 读取文件时出错: {e}")
            continue
        
        if found_count > 0:
            print(f"  共找到 {found_count} 条 Z4o 数据")
    
    print("\n" + "=" * 60)
    print("保存 PAPERZ4o 文件...")
    print("=" * 60)
    
    # 保存详细数据到 PAPERZ4o
    # 优先保存 paperour，如果没有 paperour 则保存 outline 或 paperold
    # 按类型优先级排序：paperour > paperold > outline
    type_priority = {'paperour': 3, 'paperold': 2, 'outline': 1}
    
    # 按 (M, name) 分组，选择优先级最高的类型
    grouped_data = {}  # {(M, name): (priority, type_key, data)}
    for (M, name, type_key), data in detailed_data.items():
        key = (M, name)
        priority = type_priority.get(type_key, 0)
        if key not in grouped_data or priority > grouped_data[key][0]:
            grouped_data[key] = (priority, type_key, data)
    
    # 保存文件
    for (M, name), (priority, type_key, data) in grouped_data.items():
        filename = f't{M}{name}.json'
        output_file = paperz4o_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f"  ✓ 已保存: {filename} ({type_key})")
    
    print(f"\n共保存 {len(detailed_data)} 个 PAPERZ4o 文件")
    
    # 计算平均值
    print("\n" + "=" * 60)
    print("计算平均值...")
    print("=" * 60)
    
    results = []
    
    # 按 name 和 type_key 组织数据
    for name in sorted(all_z4o_data.keys()):
        name_data = all_z4o_data[name]
        
        for type_key in ['outline', 'paperold', 'paperour']:
            if type_key not in name_data:
                continue
            
            values_dict = name_data[type_key]  # {M: value}
            
            if not values_dict:
                continue
            
            # 收集所有值
            values = list(values_dict.values())
            found_in = list(values_dict.keys())
            
            print(f"\n{name} - {type_key}:")
            print(f"  在 {len(found_in)} 个文件中找到: {found_in}")
            
            # 判断值的类型并计算平均值
            first_value = values[0]
            
            if isinstance(first_value, list):
                # 列表类型
                list_values = values
                avg_result = calculate_list_average(list_values)
                
                if avg_result is None:
                    print(f"  ✗ 计算平均值失败")
                    continue
                
                print(f"  计算过程:")
                print(f"    找到 {len(values)} 个值")
                print(f"    平均值: {avg_result}")
                
                result_item = {
                    "name": name,
                    type_key: avg_result
                }
                results.append(result_item)
                
            elif isinstance(first_value, (int, float)):
                # 数字类型
                num_values = values
                avg_result = calculate_average(num_values)
                
                if avg_result is None:
                    print(f"  ✗ 计算平均值失败")
                    continue
                
                print(f"  计算过程:")
                print(f"    找到 {len(values)} 个值: {num_values}")
                print(f"    平均值: {avg_result}")
                
                result_item = {
                    "name": name,
                    type_key: avg_result
                }
                results.append(result_item)
            
            else:
                # 字符串或其他类型，跳过
                print(f"  ⚠ 跳过非数字类型")
                continue
    
    # 追加结果到 AVG.jsonl
    print("\n" + "=" * 60)
    print("保存平均值到 AVG.jsonl...")
    print("=" * 60)
    
    if results:
        with open(avg_file, 'a', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✓ 已追加 {len(results)} 条结果到 {avg_file}")
        for result in results:
            print(f"  {result}")
    else:
        print("✗ 没有可保存的结果")

if __name__ == '__main__':
    process_z4o_data()

