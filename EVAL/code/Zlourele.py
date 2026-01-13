#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理所有t<a>目录下的a<b>.json, f<b>.json, x<b>.json文件
"""

import json
import sys
from pathlib import Path

# 导入lourele模块
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from lourele import (
    read_api_key, evaluate_method, append_to_eval_jsonl,
    OpenAI, BASE_URL
)


def check_result_exists(eval_jsonl_path, target_name):
    """
    检查eval.jsonl中是否已存在指定name的结果
    
    Args:
        eval_jsonl_path: eval.jsonl文件路径
        target_name: 要检查的name值（如 'a1', 'f2'）
    
    Returns:
        bool: 如果已存在则返回True
    """
    if not eval_jsonl_path.exists():
        return False
    
    try:
        with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if data.get('name') == target_name and 'lourele' in data:
                            return True
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"警告: 读取eval.jsonl失败: {e}")
        return False
    
    return False


def find_all_instances():
    """
    查找所有需要处理的实例
    
    Returns:
        list: [(t_dir, method, suffix), ...] 列表
    """
    instances = []
    
    # 遍历t1到t20目录
    for t_num in range(1, 21):
        t_dir = BASE_DIR / f"t{t_num}"
        if not t_dir.exists() or not t_dir.is_dir():
            continue
        
        # 查找该目录下的所有a<b>.json, f<b>.json, x<b>.json文件
        for method in ['a', 'f', 'x']:
            # 先查找无后缀的文件（如a.json）
            json_file = t_dir / f"{method}.json"
            if json_file.exists():
                instances.append((f"t{t_num}", method, ""))
            
            # 查找有后缀的文件（如a1.json, a2.json, ...）
            for suffix_num in range(1, 10):
                json_file = t_dir / f"{method}{suffix_num}.json"
                if json_file.exists():
                    instances.append((f"t{t_num}", method, str(suffix_num)))
    
    return instances


def main():
    """主函数"""
    # 读取API key
    api_key = read_api_key()
    if not api_key:
        print("错误: 无法读取API key")
        return
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    # 查找所有实例
    instances = find_all_instances()
    print(f"找到 {len(instances)} 个需要处理的实例")
    
    # 处理每个实例
    processed_count = 0
    skipped_count = 0
    
    for t_dir, method, suffix in instances:
        method_full_name = f"{method}{suffix}" if suffix else method
        eval_jsonl_path = BASE_DIR / t_dir / "eval.jsonl"
        
        # 检查是否已存在结果
        if check_result_exists(eval_jsonl_path, method_full_name):
            print(f"\n跳过已处理的实例: {t_dir}/{method_full_name}.json")
            skipped_count += 1
            continue
        
        # 处理实例
        print(f"\n处理实例: {t_dir}/{method_full_name}.json")
        try:
            result = evaluate_method(method, client, t_dir=t_dir, method_suffix=suffix)
            if result:
                append_to_eval_jsonl(result)
                processed_count += 1
                print(f"✓ 成功处理并保存: {method_full_name}")
        except Exception as e:
            print(f"✗ 处理失败 {method_full_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"处理: {processed_count} 个实例")
    print(f"跳过: {skipped_count} 个实例")
    print(f"总计: {len(instances)} 个实例")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

