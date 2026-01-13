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
        return None
    
    # 对每一项分别计算平均值
    result = []
    for i in range(lengths[0]):
        column_values = [lst[i] for lst in list_values]
        avg = sum(column_values) / len(column_values)
        result.append(avg)
    
    return result

def process_g_data():
    """处理所有 eval.jsonl 文件中 name 为 G 的数据"""
    # 获取脚本所在目录的父目录（EVAL）
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # EVAL 目录
    out_dir = base_dir / 'OUT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出目录
    avg_file = out_dir / 'AVG.jsonl'
    papergold_dir = out_dir / 'PAPERgold'
    papergold_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空 PAPERgold 目录中的旧 JSON 文件
    old_files = list(papergold_dir.glob('*.json'))
    if old_files:
        for old_file in old_files:
            old_file.unlink()
        print(f"已删除 {len(old_files)} 个旧 PAPERgold 文件")
    
    print("=" * 60)
    print("处理 name='G' 的数据...")
    print("=" * 60)
    
    # 收集所有 G 的数据
    all_g_data = {}  # {M: {type_key: value}}
    paperour_data = {}  # {M: data} 用于保存到 PAPERgold
    
    for M in range(1, 21):
        eval_file = base_dir / f't{M}' / 'eval.jsonl'
        
        if not eval_file.exists():
            print(f"  t{M}/eval.jsonl: 文件不存在")
            continue
        
        print(f"\n处理 t{M}/eval.jsonl...")
        g_data = {}
        found_paperour = False
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('name') != 'G':
                            continue
                        
                        # 获取类型键（排除 name）
                        keys = [k for k in data.keys() if k != 'name']
                        if not keys:
                            continue
                        
                        # 对于 paperour + reason 的情况，使用 'paperour' 作为类型键
                        if 'paperour' in keys:
                            type_key = 'paperour'
                            value = data['paperour']
                            # 保存完整的 paperour 数据到 paperour_data
                            paperour_data[M] = data
                            found_paperour = True
                        else:
                            # 单个键的情况
                            type_key = keys[0]
                            value = data[type_key]
                        
                        g_data[type_key] = value
                        print(f"    ✓ 找到: {type_key} = {value if not isinstance(value, list) or len(str(value)) < 50 else str(value)[:50] + '...'}")
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"    ✗ 处理行时出错: {e}")
                        continue
        
        except Exception as e:
            print(f"  ✗ 读取文件时出错: {e}")
            continue
        
        if g_data:
            all_g_data[M] = g_data
            print(f"  共找到 {len(g_data)} 个类型的数据")
            if found_paperour:
                print(f"  ✓ 找到 paperour 数据，将保存到 t{M}G.json")
        else:
            print(f"  未找到 name='G' 的数据")
    
    print("\n" + "=" * 60)
    print("保存 PAPERgold 文件...")
    print("=" * 60)
    
    # 保存 paperour 数据到 PAPERgold
    for M, data in paperour_data.items():
        filename = f't{M}G.json'
        output_file = papergold_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f"  ✓ 已保存: {filename}")
    
    print(f"\n共保存 {len(paperour_data)} 个 PAPERgold 文件")
    
    # 计算平均值
    print("\n" + "=" * 60)
    print("计算平均值...")
    print("=" * 60)
    
    # 收集所有类型
    all_types = set()
    for g_data in all_g_data.values():
        all_types.update(g_data.keys())
    
    results = []
    
    # 排序类型键（将元组转换为字符串以便排序）
    sorted_types = sorted(all_types, key=lambda x: str(x))
    
    for type_key in sorted_types:
        print(f"\n类型: {type_key}")
        print("-" * 60)
        
        # 收集所有 M 中该类型的值
        values = []
        found_in = []
        
        for M in range(1, 21):
            if M not in all_g_data:
                continue
            
            g_data = all_g_data[M]
            if type_key in g_data:
                values.append((M, g_data[type_key]))
                found_in.append(M)
        
        if not values:
            print("  未找到数据")
            continue
        
        print(f"  在 {len(found_in)} 个文件中找到: {found_in}")
        
        # 判断值的类型并计算平均值
        first_value = values[0][1]
        
        if isinstance(first_value, list):
            # 列表类型
            list_values = [v[1] for v in values]
            avg_result = calculate_list_average(list_values)
            
            if avg_result is None:
                print(f"  ✗ 计算平均值失败")
                continue
            
            print(f"  计算过程:")
            print(f"    找到 {len(values)} 个值")
            print(f"    平均值: {avg_result}")
            
            result_item = {
                "name": "G",
                type_key: avg_result
            }
            results.append(result_item)
            
        elif isinstance(first_value, (int, float)):
            # 数字类型
            num_values = [v[1] for v in values]
            avg_result = calculate_average(num_values)
            
            if avg_result is None:
                print(f"  ✗ 计算平均值失败")
                continue
            
            print(f"  计算过程:")
            print(f"    找到 {len(values)} 个值: {num_values}")
            print(f"    平均值: {avg_result}")
            
            result_item = {
                "name": "G",
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
    process_g_data()

