#!/usr/bin/env python3
import json
import os
from pathlib import Path

def extract_paperour_data():
    """从所有 tM/eval.jsonl 文件中提取 paperour 相关的行"""
    # 获取脚本所在目录的父目录（EVAL）
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # EVAL 目录
    output_dir = base_dir / 'OUT' / 'PAPER'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空输出目录中的旧 JSON 文件
    old_files = list(output_dir.glob('*.json'))
    if old_files:
        for old_file in old_files:
            old_file.unlink()
        print(f"已删除 {len(old_files)} 个旧文件")
    
    print("=" * 60)
    print("提取 paperour 数据...")
    print("=" * 60)
    
    total_extracted = 0
    
    for M in range(1, 21):
        eval_file = base_dir / f't{M}' / 'eval.jsonl'
        
        if not eval_file.exists():
            print(f"  t{M}/eval.jsonl: 文件不存在")
            continue
        
        print(f"\n处理 t{M}/eval.jsonl...")
        count = 0
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # 检查是否包含 paperour
                        if 'paperour' not in data:
                            continue
                        
                        # 检查是否有 name 字段
                        if 'name' not in data:
                            continue
                        
                        name = data.get('name')
                        
                        # 生成文件名：tM + name，例如 t2a.json
                        filename = f't{M}{name}.json'
                        output_file = output_dir / filename
                        
                        # 保存为 JSON 文件（每行一个 JSON 对象）
                        # 由于 paperour 行可能包含 reason（字符串列表），我们需要保存完整的 JSON 对象
                        with open(output_file, 'w', encoding='utf-8') as out_f:
                            # 将整个 JSON 对象写入文件，每行一个 JSON
                            out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        
                        count += 1
                        total_extracted += 1
                        print(f"    ✓ 提取: {name} -> {filename}")
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"    ✗ 处理行时出错: {e}")
                        continue
        
        except Exception as e:
            print(f"  ✗ 读取文件时出错: {e}")
            continue
        
        print(f"  共提取 {count} 条 paperour 数据")
    
    print("\n" + "=" * 60)
    print(f"总计: 提取了 {total_extracted} 条 paperour 数据")
    print(f"保存位置: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    extract_paperour_data()

