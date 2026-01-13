#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import sys

def check_existing_result(eval_path, name):
    """检查eval.jsonl中是否已存在该实例的结果"""
    if not os.path.exists(eval_path):
        return False, False
    
    has_paperold = False
    has_paperour = False
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('name') == name:
                    if 'paperold' in data:
                        has_paperold = True
                    if 'paperour' in data:
                        has_paperour = True
                    if has_paperold and has_paperour:
                        return True, True
            except:
                continue
    
    return has_paperold, has_paperour

def find_json_files(base_dir, dir_name):
    """在指定目录下查找所有 a<b>.json, f<b>.json, x<b>.json 文件"""
    dir_path = os.path.join(base_dir, dir_name)
    if not os.path.exists(dir_path):
        return []
    
    files = []
    prefixes = ['a', 'f', 'x']
    
    # b为空的情况（a.json, f.json, x.json）
    for prefix in prefixes:
        file_path = os.path.join(dir_path, f'{prefix}.json')
        if os.path.exists(file_path):
            files.append(prefix)
    
    # b为1-9的情况（a1.json, a2.json, ..., f1.json, ...）
    for prefix in prefixes:
        for b in range(1, 10):
            file_path = os.path.join(dir_path, f'{prefix}{b}.json')
            if os.path.exists(file_path):
                files.append(f'{prefix}{b}')
    
    return files

def main():
    # 获取脚本所在目录
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(code_dir)
    
    # 遍历 t1 到 t20
    for t_num in range(1, 21):
        dir_name = f't{t_num}'
        dir_path = os.path.join(base_dir, dir_name)
        
        if not os.path.exists(dir_path):
            print(f"目录 {dir_name} 不存在，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理目录: {dir_name}")
        print(f"{'='*60}\n")
        
        # 查找所有需要评估的文件
        json_files = find_json_files(base_dir, dir_name)
        
        if not json_files:
            print(f"  目录 {dir_name} 下没有找到需要评估的文件")
            continue
        
        print(f"  找到 {len(json_files)} 个文件: {', '.join([f'{f}.json' for f in json_files])}\n")
        
        # 检查paper.json是否存在
        paper_path = os.path.join(dir_path, 'paper.json')
        if not os.path.exists(paper_path):
            print(f"  警告: {dir_name}/paper.json 不存在，跳过该目录")
            continue
        
        eval_path = os.path.join(dir_path, 'eval.jsonl')
        
        # 处理每个文件
        for file_base_name in json_files:
            print(f"\n  {'-'*50}")
            print(f"  处理实例: {file_base_name}.json")
            print(f"  {'-'*50}")
            
            # 检查是否已存在结果
            has_paperold, has_paperour = check_existing_result(eval_path, file_base_name)
            
            if has_paperold and has_paperour:
                print(f"  ✓ 跳过 {file_base_name}.json (结果已存在)")
                continue
            
            # 调用 llmevalpaper.py 进行评估
            script_path = os.path.join(code_dir, 'llmevalpaper.py')
            cmd = [
                sys.executable,
                script_path,
                '--dir', dir_name,
                '--file', file_base_name
            ]
            
            print(f"  执行命令: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=code_dir,
                    capture_output=False,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    print(f"  ✓ {file_base_name}.json 评估完成")
                else:
                    print(f"  ✗ {file_base_name}.json 评估失败 (返回码: {result.returncode})")
            except Exception as e:
                print(f"  ✗ {file_base_name}.json 评估出错: {e}")
        
        # 处理 paper.json (转换为 Paper.json 并评估，name设为 "G")
        print(f"\n  {'-'*50}")
        print(f"  处理实例: paper.json -> Paper.json (name: G)")
        print(f"  {'-'*50}")
        
        # 检查是否已存在结果（name为"G"）
        has_paperold, has_paperour = check_existing_result(eval_path, "G")
        
        if has_paperold and has_paperour:
            print(f"  ✓ 跳过 paper.json (结果已存在)")
        else:
            # 调用 llmevalpaper.py 进行评估
            script_path = os.path.join(code_dir, 'llmevalpaper.py')
            cmd = [
                sys.executable,
                script_path,
                '--dir', dir_name,
                '--file', 'paper'
            ]
            
            print(f"  执行命令: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=code_dir,
                    capture_output=False,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    print(f"  ✓ paper.json 评估完成")
                else:
                    print(f"  ✗ paper.json 评估失败 (返回码: {result.returncode})")
            except Exception as e:
                print(f"  ✗ paper.json 评估出错: {e}")
        
        print(f"\n  目录 {dir_name} 处理完成")
    
    print(f"\n{'='*60}")
    print("所有目录处理完成！")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

