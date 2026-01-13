#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import subprocess
import sys

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "code")
LLMEVAL_SCRIPT = os.path.join(CODE_DIR, "llmevaloutlineZ4o.py")


def check_result_exists(eval_file, name):
    """检查eval.jsonl中是否已存在指定name的结果（带Z4o后缀）"""
    if not os.path.exists(eval_file):
        return False
    
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("name") == name and "outline" in data:
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"警告: 读取 {eval_file} 时出错: {e}")
    
    return False


def get_name_from_filename(filename):
    """从文件名提取name（如fO.txt -> f, fO1.txt -> f1），会自动添加Z4o后缀"""
    match = re.match(r'([fax])O(\d*)\.txt', filename)
    if match:
        prefix = match.group(1)
        suffix = match.group(2) if match.group(2) else ""
        return prefix + suffix + "Z4o"
    return None


def evaluate_file(t_dir, filename):
    """评估单个文件，返回评估结果或None"""
    # 处理Outline.txt的特殊情况
    if filename == "Outline.txt":
        name = "GZ4o"
    else:
        name = get_name_from_filename(filename)
        if not name:
            print(f"警告: 无法解析文件名 {filename}，跳过")
            return None
    
    print(f"\n{'='*60}")
    print(f"处理实例: {t_dir}/{filename}")
    print(f"{'='*60}")
    
    # 构建eval.jsonl文件路径
    eval_file = os.path.join(BASE_DIR, t_dir, "eval.jsonl")
    
    # 调用 llmevaloutlineZ4o.py
    cmd = [
        sys.executable,
        LLMEVAL_SCRIPT,
        "--dir", t_dir,
        "--file", filename,
        "--quiet"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=CODE_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            # 读取评估结果
            if os.path.exists(eval_file):
                with open(eval_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("name") == name and "outline" in data:
                                scores = data["outline"]
                                print(f"评估结果: {t_dir}/{filename} -> name={name}, outline={scores}")
                                return scores
                        except json.JSONDecodeError:
                            continue
            print(f"完成: {t_dir}/{filename}")
        else:
            print(f"错误: 评估 {t_dir}/{filename} 失败")
            print(f"错误输出: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"错误: 评估 {t_dir}/{filename} 超时")
        return None
    except Exception as e:
        print(f"错误: 评估 {t_dir}/{filename} 时出错: {e}")
        return None
    
    return None


def main():
    print("开始遍历所有可能的实例（使用gpt-4o模型）...")
    print(f"基础目录: {BASE_DIR}")
    print(f"脚本位置: {LLMEVAL_SCRIPT}\n")
    
    # 遍历所有可能的目录 t1 到 t20
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for t_num in range(1, 21):
        t_dir = f"t{t_num}"
        t_path = os.path.join(BASE_DIR, t_dir)
        
        if not os.path.isdir(t_path):
            continue
        
        print(f"\n检查目录: {t_dir}")
        
        # 遍历所有可能的文件：fO<b>.txt, aO<b>.txt, xO<b>.txt, Outline.txt
        # b 为空或 1-9
        prefixes = ['f', 'a', 'x']
        suffixes = [''] + [str(i) for i in range(1, 10)]
        
        # 先处理fO<b>.txt, aO<b>.txt, xO<b>.txt
        for prefix in prefixes:
            for suffix in suffixes:
                filename = f"{prefix}O{suffix}.txt" if suffix else f"{prefix}O.txt"
                filepath = os.path.join(t_path, filename)
                
                if os.path.exists(filepath):
                    # name带Z4o后缀
                    name = get_name_from_filename(filename)
                    eval_file = os.path.join(t_path, "eval.jsonl")
                    
                    # 先检查是否已存在结果（带Z4o后缀）
                    if check_result_exists(eval_file, name):
                        print(f"跳过 {t_dir}/{filename} (结果已存在: name={name})")
                        total_skipped += 1
                        continue
                    
                    # 评估文件
                    result = evaluate_file(t_dir, filename)
                    if result is not None:
                        total_processed += 1
                    else:
                        total_errors += 1
        
        # 处理Outline.txt
        outline_filepath = os.path.join(t_path, "Outline.txt")
        if os.path.exists(outline_filepath):
            eval_file = os.path.join(t_path, "eval.jsonl")
            name = "GZ4o"
            
            # 先检查是否已存在结果（带Z4o后缀）
            if check_result_exists(eval_file, name):
                print(f"跳过 {t_dir}/Outline.txt (结果已存在: name={name})")
                total_skipped += 1
            else:
                # 评估文件
                result = evaluate_file(t_dir, "Outline.txt")
                if result is not None:
                    total_processed += 1
                else:
                    total_errors += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"成功处理: {total_processed} 个实例")
    print(f"跳过: {total_skipped} 个实例（结果已存在）")
    print(f"错误: {total_errors} 个实例")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

