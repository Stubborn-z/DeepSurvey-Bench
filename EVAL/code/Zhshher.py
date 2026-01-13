#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理所有t<a>目录下的xo<b>.json, fo<b>.txt, a<b>.json文件
调用hshher.py进行评估
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set, Tuple


def check_already_evaluated(eval_jsonl_path: Path, name: str) -> Tuple[bool, bool]:
    """
    检查eval.jsonl中是否已存在指定name的hsr和her结果
    
    返回:
        (has_hsr, has_her): 是否已有HSR结果，是否已有HER结果
    """
    if not eval_jsonl_path.exists():
        return False, False
    
    has_hsr = False
    has_her = False
    
    try:
        with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("name") == name:
                        if "hsr" in data:
                            has_hsr = True
                        if "her" in data:
                            has_her = True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"警告: 读取 {eval_jsonl_path} 时出错: {e}")
        return False, False
    
    return has_hsr, has_her


def process_instance(t_id: int, file_type: str, file_suffix: str) -> bool:
    """
    处理单个实例
    
    参数:
        t_id: t目录的编号（1-20）
        file_type: 文件类型 ('x', 'f', 'a')
        file_suffix: 文件编号后缀（空字符串或 '1'-'9'）
    
    返回:
        True: 成功处理
        False: 跳过或失败
    """
    base_dir = Path(__file__).parent.parent / f"t{t_id}"
    
    # 构建文件名
    suffix_str = file_suffix if file_suffix else ""
    if file_type == 'x':
        file_path = base_dir / f"xo{suffix_str}.json"
    elif file_type == 'f':
        file_path = base_dir / f"fo{suffix_str}.txt"
    elif file_type == 'a':
        file_path = base_dir / f"a{suffix_str}.json"
    else:
        return False
    
    # 检查文件是否存在
    if not file_path.exists():
        return False
    
    # 构建实例名
    name = f"{file_type}{file_suffix}" if file_suffix else file_type
    
    # 检查是否已评估
    eval_jsonl_path = base_dir / "eval.jsonl"
    has_hsr, has_her = check_already_evaluated(eval_jsonl_path, name)
    
    # 如果两者都已评估，跳过
    if has_hsr and has_her:
        print(f"跳过 t{t_id}/{name} (HSR和HER都已评估)")
        return False
    
    # 确定需要计算什么
    compute_hsr = not has_hsr
    compute_her = not has_her
    
    if has_hsr and not has_her:
        print(f"处理实例: t{t_id}/{name} (只计算HER)")
    elif not has_hsr and has_her:
        print(f"处理实例: t{t_id}/{name} (只计算HSR)")
    else:
        print(f"处理实例: t{t_id}/{name} (计算HSR和HER)")
    print(f"文件: {file_path.name}")
    
    try:
        # 构建命令
        script_path = Path(__file__).parent / "hshher.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--t_id", str(t_id),
            "--file_type", file_type,
            "--file_suffix", file_suffix
        ]
        
        # 添加计算标志
        if not compute_hsr:
            cmd.append("--no_compute_hsr")
        if not compute_her:
            cmd.append("--no_compute_her")
        
        # 执行命令
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            # 输出结果
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            # 读取评估结果
            hsr_value = None
            her_value = None
            if eval_jsonl_path.exists():
                with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 查找最后添加的结果（从后往前查找）
                    for line in reversed(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("name") == name:
                                if "hsr" in data and hsr_value is None:
                                    hsr_value = data['hsr']
                                if "her" in data and her_value is None:
                                    her_value = data['her']
                                if hsr_value is not None and her_value is not None:
                                    break
                        except json.JSONDecodeError:
                            continue
            
            # 显示评估结果
            if hsr_value is not None and her_value is not None:
                print(f"评估结果: {name} - HSR={hsr_value:.4f}, HER={her_value:.4f}")
            elif hsr_value is not None:
                print(f"评估结果: {name} - HSR={hsr_value:.4f}")
            elif her_value is not None:
                print(f"评估结果: {name} - HER={her_value:.4f}")
            
            return True
        else:
            print(f"错误: 处理失败")
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"错误: 处理时发生异常: {e}")
        return False


def main():
    """主函数：遍历所有可能的实例"""
    print("开始批量处理...")
    
    # 遍历所有t目录（1-20）
    for t_id in range(1, 21):
        base_dir = Path(__file__).parent.parent / f"t{t_id}"
        if not base_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"处理目录: t{t_id}")
        print(f"{'='*60}")
        
        # 遍历所有文件类型
        for file_type in ['x', 'f', 'a']:
            # 遍历所有可能的文件编号（空字符串和1-9）
            for file_suffix in [''] + [str(i) for i in range(1, 10)]:
                process_instance(t_id, file_type, file_suffix)
    
    print(f"\n{'='*60}")
    print("批量处理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

