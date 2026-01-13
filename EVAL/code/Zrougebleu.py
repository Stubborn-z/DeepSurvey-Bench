#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量评估ROUGE和BLEU指标的脚本
遍历所有t<a>/目录下的a<b>.json, f<b>.json, x<b>.json文件进行评估
<a>值为1到20，<b>值为空或1-9
"""

import os
import json
import subprocess
import sys


def check_result_exists(eval_jsonl_path, method_name):
    """
    检查eval.jsonl中是否已存在该方法的评估结果
    
    Args:
        eval_jsonl_path: eval.jsonl文件路径
        method_name: 方法名称（如'a', 'a2', 'f3'等）
    
    Returns:
        bool: 如果已存在完整的评估结果（rouge和bleu都有）返回True
    """
    if not os.path.exists(eval_jsonl_path):
        return False
    
    has_rouge = False
    has_bleu = False
    
    try:
        with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('name') == method_name:
                        if 'rouge' in data:
                            has_rouge = True
                        if 'bleu' in data:
                            has_bleu = True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取 {eval_jsonl_path} 时出错: {e}")
        return False
    
    return has_rouge and has_bleu


def evaluate_instance(t_dir, method_prefix, method_suffix, base_dir):
    """
    评估单个实例
    
    Args:
        t_dir: t目录名称（如't1'）
        method_prefix: 方法前缀（'a', 'f', 'x'）
        method_suffix: 方法后缀（''或'2', '3'等）
        base_dir: EVAL目录路径
    
    Returns:
        bool: 评估是否成功
    """
    # 构建方法名称
    method_name = method_prefix + method_suffix if method_suffix else method_prefix
    
    # 构建文件路径
    t_path = os.path.join(base_dir, t_dir)
    generated_file = os.path.join(t_path, f"{method_name}.json")
    paper_file = os.path.join(t_path, 'paper.json')
    eval_file = os.path.join(t_path, 'eval.jsonl')
    
    # 检查生成文件是否存在
    if not os.path.exists(generated_file):
        return False
    
    # 检查参考文件是否存在
    if not os.path.exists(paper_file):
        print(f"警告: {paper_file} 不存在，跳过 {t_dir}/{method_name}.json")
        return False
    
    # 检查是否已评估
    if check_result_exists(eval_file, method_name):
        print(f"跳过已评估实例: {t_dir}/{method_name}.json")
        return True
    
    # 调用rougebleu.py进行评估
    print(f"\n{'='*60}")
    print(f"评估实例: {t_dir}/{method_name}.json")
    print(f"{'='*60}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rougebleu_script = os.path.join(script_dir, 'rougebleu.py')
    
    try:
        result = subprocess.run(
            [sys.executable, rougebleu_script,
             '--generated', generated_file,
             '--reference', paper_file,
             '--output', eval_file,
             '--method', method_name],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            # 输出评估结果
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            # 读取并显示结果
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # 显示最后两行（rouge和bleu结果）
                        for line in lines[-2:]:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if data.get('name') == method_name:
                                        if 'rouge' in data:
                                            r1, r2, rl = data['rouge']
                                            print(f"  ROUGE: [{r1:.4f}, {r2:.4f}, {rl:.4f}]")
                                        elif 'bleu' in data:
                                            print(f"  BLEU: {data['bleu']:.4f}")
                                except json.JSONDecodeError:
                                    pass
                except Exception as e:
                    print(f"读取结果时出错: {e}")
            
            print(f"实例 {t_dir}/{method_name}.json 评估完成\n")
            return True
        else:
            print(f"评估失败: {t_dir}/{method_name}.json")
            print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"调用评估脚本时出错: {e}")
        return False


def main():
    # 获取EVAL目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # EVAL目录
    
    print("开始批量评估ROUGE和BLEU指标...")
    print(f"基础目录: {base_dir}")
    print("=" * 60)
    
    # 遍历所有可能的t目录（t1到t20）
    total_instances = 0
    processed_instances = 0
    skipped_instances = 0
    failed_instances = 0
    
    for t_num in range(1, 21):
        t_dir = f"t{t_num}"
        t_path = os.path.join(base_dir, t_dir)
        
        # 检查t目录是否存在
        if not os.path.isdir(t_path):
            continue
        
        print(f"\n处理目录: {t_dir}")
        print("-" * 60)
        
        # 遍历所有可能的方法文件
        for method_prefix in ['a', 'f', 'x']:
            # 处理空后缀和1-9的后缀
            suffixes = [''] + [str(i) for i in range(1, 10)]
            
            for suffix in suffixes:
                method_name = method_prefix + suffix if suffix else method_prefix
                generated_file = os.path.join(t_path, f"{method_name}.json")
                
                # 检查文件是否存在
                if os.path.exists(generated_file):
                    total_instances += 1
                    
                    # 检查是否已评估
                    eval_file = os.path.join(t_path, 'eval.jsonl')
                    if check_result_exists(eval_file, method_name):
                        skipped_instances += 1
                        print(f"跳过已评估: {t_dir}/{method_name}.json")
                        continue
                    
                    # 进行评估
                    if evaluate_instance(t_dir, method_prefix, suffix, base_dir):
                        processed_instances += 1
                    else:
                        failed_instances += 1
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("批量评估完成！")
    print("=" * 60)
    print(f"总实例数: {total_instances}")
    print(f"新处理: {processed_instances}")
    print(f"已跳过: {skipped_instances}")
    print(f"失败: {failed_instances}")
    print("=" * 60)


if __name__ == '__main__':
    main()

