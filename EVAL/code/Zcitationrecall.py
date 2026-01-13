#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import json
import re

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "code")
CITATIONRECALL_SCRIPT = os.path.join(CODE_DIR, "citationrecall.py")


def find_json_files(t_dir):
    """在指定目录下查找所有 a<b>.json, f<b>.json, x<b>.json 文件
    返回: list of tuples [(method_name, full_path), ...]
    method_name 格式：'a', 'a1', 'f2', 'x9' 等
    """
    json_files = []
    
    if not os.path.isdir(t_dir):
        return json_files
    
    # 匹配模式：a<b>.json, f<b>.json, x<b>.json
    # 其中 <b> 可以为空或1-9
    pattern = re.compile(r'^([afx])(\d*)\.json$')
    
    for filename in os.listdir(t_dir):
        if not filename.endswith('.json'):
            continue
        
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)  # a, f, 或 x
            suffix = match.group(2)  # 空字符串或1-9
            
            if not suffix or (suffix.isdigit() and 1 <= int(suffix) <= 9):
                method_name = prefix + suffix  # 'a', 'a1', 'f2' 等
                full_path = os.path.join(t_dir, filename)
                json_files.append((method_name, full_path))
    
    # 排序：先按前缀（a, f, x），再按数字
    json_files.sort(key=lambda x: (x[0][0], int(x[0][1:]) if len(x[0]) > 1 else 0))
    
    return json_files


def check_already_evaluated(eval_jsonl_path, method_name):
    """检查某个方法是否已经评估过
    返回: (has_recall, has_precision)
    """
    if not os.path.exists(eval_jsonl_path):
        return False, False
    
    has_recall = False
    has_precision = False
    
    try:
        with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get('name') == method_name:
                        if 'citationrecall' in item:
                            has_recall = True
                        if 'citationprecision' in item:
                            has_precision = True
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    return has_recall, has_precision


def evaluate_one_instance(t_dir, method_name):
    """评估单个实例
    返回: (success, recall, precision, skipped) 或 (False, None, None, False)
    """
    eval_jsonl_path = os.path.join(t_dir, "eval.jsonl")
    
    # 检查是否已评估
    has_recall, has_precision = check_already_evaluated(eval_jsonl_path, method_name)
    if has_recall and has_precision:
        # 如果已评估，尝试读取结果
        recall, precision = None, None
        if os.path.exists(eval_jsonl_path):
            with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if item.get('name') == method_name:
                            if 'citationrecall' in item and recall is None:
                                recall = item['citationrecall']
                            if 'citationprecision' in item and precision is None:
                                precision = item['citationprecision']
                            if recall is not None and precision is not None:
                                break
                    except json.JSONDecodeError:
                        continue
        
        if recall is not None and precision is not None:
            print(f"  {method_name}: Recall={recall:.4f}, Precision={precision:.4f} (已跳过，使用已有结果)")
        else:
            print(f"  {method_name}: 已跳过（已评估）")
        return True, recall, precision, True
    
    # 获取 t_dir 的目录名（如 't1', 't2'）
    t_dir_name = os.path.basename(t_dir)
    
    # 调用 citationrecall.py
    cmd = [
        sys.executable,
        CITATIONRECALL_SCRIPT,
        '--t_dir', t_dir_name,
        '--method', method_name
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=CODE_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 输出标准输出（用于调试）
        if result.stdout:
            # 只输出关键信息，避免太多输出
            for line in result.stdout.split('\n'):
                if 'Recall=' in line or 'Precision=' in line or '错误' in line or '警告' in line:
                    print(f"    {line}")
        
        if result.returncode == 0:
            # 从 eval.jsonl 读取最新结果
            recall, precision = None, None
            if os.path.exists(eval_jsonl_path):
                with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in reversed(list(f)):  # 从后往前查找
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if item.get('name') == method_name:
                                if 'citationrecall' in item and recall is None:
                                    recall = item['citationrecall']
                                if 'citationprecision' in item and precision is None:
                                    precision = item['citationprecision']
                                if recall is not None and precision is not None:
                                    break
                        except json.JSONDecodeError:
                            continue
            
            if recall is not None and precision is not None:
                print(f"  {method_name}: Recall={recall:.4f}, Precision={precision:.4f}")
            else:
                print(f"  {method_name}: 完成（但未找到结果）")
            
            return True, recall, precision, False
        else:
            print(f"  错误: {method_name} 评估失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"  错误输出: {result.stderr[:500]}")  # 只显示前500字符
            return False, None, None, False
            
    except subprocess.TimeoutExpired:
        print(f"  错误: {method_name} 评估超时")
        return False, None, None, False
    except Exception as e:
        print(f"  错误: {method_name} 评估异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, False


def main():
    """主函数"""
    print("="*80)
    print("批量评估 Citation Recall 和 Precision")
    print("="*80)
    
    # 遍历 t1 到 t20
    all_results = {}
    
    for t_num in range(1, 21):
        t_dir_name = f"t{t_num}"
        t_dir = os.path.join(BASE_DIR, t_dir_name)
        
        if not os.path.isdir(t_dir):
            continue
        
        print(f"\n处理目录: {t_dir_name}")
        print("-" * 80)
        
        # 查找所有符合条件的 JSON 文件
        json_files = find_json_files(t_dir)
        
        if not json_files:
            print(f"  未找到符合条件的 JSON 文件")
            continue
        
        print(f"  找到 {len(json_files)} 个文件")
        
        # 处理每个文件
        t_results = {}
        for method_name, json_path in json_files:
            success, recall, precision, skipped = evaluate_one_instance(t_dir, method_name)
            if success:
                t_results[method_name] = {
                    'recall': recall,
                    'precision': precision,
                    'skipped': skipped
                }
        
        all_results[t_dir_name] = t_results
    
    # 输出汇总
    print("\n" + "="*80)
    print("评估汇总")
    print("="*80)
    
    total_count = 0
    success_count = 0
    
    for t_dir_name, t_results in sorted(all_results.items()):
        if not t_results:
            continue
        
        print(f"\n{t_dir_name}:")
        for method_name, result in sorted(t_results.items()):
            total_count += 1
            if result['recall'] is not None and result['precision'] is not None:
                success_count += 1
                status = "（已跳过）" if result.get('skipped', False) else ""
                print(f"  {method_name}: Recall={result['recall']:.4f}, Precision={result['precision']:.4f}{status}")
            else:
                print(f"  {method_name}: 已跳过或评估失败")
    
    print(f"\n总计: {success_count}/{total_count} 个实例成功评估")
    print("\n完成！")


if __name__ == '__main__':
    main()
