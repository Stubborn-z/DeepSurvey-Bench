#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict

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

def process_zour0_data():
    """处理所有 evalZour0.jsonl 文件"""
    # 获取脚本所在目录的父目录（EVAL）
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # EVAL 目录
    out_dir = base_dir / 'OUT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出目录和文件
    avg_file = out_dir / 'AVGZour0.jsonl'
    paperzour0_dir = out_dir / 'PAPERZour0'
    paperzour0_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空 PAPERZour0 目录中的旧 JSON 文件
    old_files = list(paperzour0_dir.glob('*.json'))
    if old_files:
        for old_file in old_files:
            old_file.unlink()
        print(f"已删除 {len(old_files)} 个旧 PAPERZour0 文件")
    
    print("=" * 60)
    print("处理 evalZour0.jsonl 文件...")
    print("=" * 60)
    
    # 收集所有数据：{name: {M: paperour_value}}
    all_data = defaultdict(dict)  # {name: {M: paperour}}
    paperour_data = {}  # {(M, name): data} 用于保存到 PAPERZour0
    
    existing_files = []
    
    for M in range(1, 21):
        eval_file = base_dir / f't{M}' / 'evalZour0.jsonl'
        
        if not eval_file.exists():
            print(f"  t{M}/evalZour0.jsonl: 文件不存在")
            continue
        
        existing_files.append(M)
        print(f"\n处理 t{M}/evalZour0.jsonl...")
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        name = data.get('name')
                        if not name:
                            continue
                        
                        # 提取 paperour 数据
                        if 'paperour' in data:
                            paperour_value = data['paperour']
                            all_data[name][M] = paperour_value
                            paperour_data[(M, name)] = data
                            print(f"    ✓ 找到: name={name}, paperour={paperour_value}")
                        else:
                            print(f"    ⚠ 跳过: name={name} (无 paperour 字段)")
                        
                    except json.JSONDecodeError as e:
                        print(f"    ✗ JSON 解析错误: {e}")
                        continue
                    except Exception as e:
                        print(f"    ✗ 处理行时出错: {e}")
                        continue
        
        except Exception as e:
            print(f"  ✗ 读取文件时出错: {e}")
            continue
    
    print(f"\n共处理 {len(existing_files)} 个文件: {existing_files}")
    
    # 保存 PAPERZour0 文件
    print("\n" + "=" * 60)
    print("保存 PAPERZour0 文件...")
    print("=" * 60)
    
    for (M, name), data in paperour_data.items():
        filename = f't{M}{name}.json'
        output_file = paperzour0_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f"  ✓ 已保存: {filename}")
    
    print(f"\n共保存 {len(paperour_data)} 个 PAPERZour0 文件")
    
    # 计算平均值
    print("\n" + "=" * 60)
    print("计算平均值...")
    print("=" * 60)
    
    results = []
    
    # 按 name 排序
    sorted_names = sorted(all_data.keys())
    
    for name in sorted_names:
        print(f"\n处理 name={name}...")
        print("-" * 60)
        
        name_data = all_data[name]
        
        # 检查是否在所有文件中都能找到
        found_in = sorted(name_data.keys())
        missing_in = [M for M in range(1, 21) if M not in name_data]
        
        if len(found_in) < 20:
            print(f"  ✗ 缺项: 仅在 {len(found_in)} 个文件中找到，缺失: {missing_in}")
            continue
        
        print(f"  ✓ 在 20 个文件中都找到")
        
        # 收集所有 paperour 值
        paperour_values = [name_data[M] for M in sorted(found_in)]
        
        # 计算平均值
        avg_result = calculate_list_average(paperour_values)
        
        if avg_result is None:
            print(f"  ✗ 计算平均值失败")
            continue
        
        print(f"  计算过程:")
        print(f"    找到 {len(paperour_values)} 个值")
        print(f"    平均值: {avg_result}")
        
        result_item = {
            "name": name,
            "paperour": avg_result
        }
        results.append(result_item)
    
    # 追加结果到 AVGZour0.jsonl
    print("\n" + "=" * 60)
    print("保存平均值到 AVGZour0.jsonl...")
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
    process_zour0_data()

