#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOVE.py - 将各个工具的输出文件复制到EVAL目录下
"""

import os
import shutil
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent.parent.parent
EVAL_DIR = BASE_DIR / "EVAL"
SURVEYFORGE_RES_DIR = BASE_DIR / "SurveyForge-main" / "code" / "output" / "res"
SURVEYX_OUTPUTS_DIR = BASE_DIR / "SurveyX-main" / "outputs"
AUTOSURVEY_OUTPUT_DIR = BASE_DIR / "AutoSurvey-main" / "output"

def copy_file(src, dst):
    """复制文件，如果目标文件已存在则跳过"""
    if not src.exists():
        print(f"  ✗ 文件不存在: {src}")
        return False
    
    if dst.exists():
        print(f"  ⊙ 跳过（目标文件已存在）: {dst.name}")
        return False
    
    shutil.copy2(src, dst)
    print(f"  ✓ 复制: {src.name} -> {dst.name}")
    return True

def find_exp_dir(base_dir):
    """查找 exp 或 exp_1 目录"""
    exp_dir = base_dir / "exp"
    if exp_dir.exists():
        return exp_dir
    exp_1_dir = base_dir / "exp_1"
    if exp_1_dir.exists():
        return exp_1_dir
    return None

def process_surveyforge_folder(base_folder, target_dir, suffix=""):
    """处理 SurveyForge-main 文件夹"""
    surveyforge_dir = SURVEYFORGE_RES_DIR / base_folder
    if not surveyforge_dir.exists():
        return False
    
    exp_dir = find_exp_dir(surveyforge_dir)
    if not exp_dir:
        return False
    
    file_mappings = [
        ("f.json", f"f{suffix}.json" if suffix else "f.json"),
        ("fc.json", f"fc{suffix}.json" if suffix else "fc.json"),
        ("fo.txt", f"fo{suffix}.txt" if suffix else "fo.txt")
    ]
    
    for src_filename, dst_filename in file_mappings:
        src = exp_dir / src_filename
        dst = target_dir / dst_filename
        copy_file(src, dst)
    
    return True

def process_surveyx_folder(base_folder, target_dir, suffix=""):
    """处理 SurveyX-main 文件夹"""
    surveyx_dir = SURVEYX_OUTPUTS_DIR / base_folder
    if not surveyx_dir.exists():
        return False
    
    file_mappings = [
        ("x.json", f"x{suffix}.json" if suffix else "x.json"),
        ("xc.json", f"xc{suffix}.json" if suffix else "xc.json"),
        ("xo.json", f"xo{suffix}.json" if suffix else "xo.json")
    ]
    
    for src_filename, dst_filename in file_mappings:
        src = surveyx_dir / src_filename
        dst = target_dir / dst_filename
        copy_file(src, dst)
    
    return True

def process_autosurvey_folder(base_folder, target_dir, suffix=""):
    """处理 AutoSurvey-main 文件夹"""
    autosurvey_dir = AUTOSURVEY_OUTPUT_DIR / base_folder
    if not autosurvey_dir.exists():
        return False
    
    file_mappings = [
        ("a.json", f"a{suffix}.json" if suffix else "a.json"),
        ("ac.json", f"ac{suffix}.json" if suffix else "ac.json")
    ]
    
    for src_filename, dst_filename in file_mappings:
        src = autosurvey_dir / src_filename
        dst = target_dir / dst_filename
        copy_file(src, dst)
    
    return True

def process_folder(folder_name):
    """处理单个文件夹（如t1, t2等）及其变体（t1a, t1b等）"""
    print(f"\n处理 {folder_name}...")
    
    target_dir = EVAL_DIR / folder_name
    if not target_dir.exists():
        print(f"  ⚠ 目标目录不存在: {target_dir}")
        return
    
    # 1. 处理主文件夹（如 t1）
    print(f"  从主文件夹 {folder_name} 复制文件:")
    
    # SurveyForge-main
    process_surveyforge_folder(folder_name, target_dir)
    
    # SurveyX-main
    process_surveyx_folder(folder_name, target_dir)
    
    # AutoSurvey-main
    process_autosurvey_folder(folder_name, target_dir)
    
    # 2. 处理变体文件夹（如 t1a, t1b, t1c, t1d, t1e, t1f）
    variant_suffixes = ['a', 'b', 'c', 'd', 'e', 'f']
    for idx, suffix in enumerate(variant_suffixes, start=1):
        variant_folder = f"{folder_name}{suffix}"
        
        # 检查是否存在任何变体文件夹
        found_surveyforge = (SURVEYFORGE_RES_DIR / variant_folder).exists()
        found_surveyx = (SURVEYX_OUTPUTS_DIR / variant_folder).exists()
        found_autosurvey = (AUTOSURVEY_OUTPUT_DIR / variant_folder).exists()
        
        if found_surveyforge or found_surveyx or found_autosurvey:
            print(f"  从变体文件夹 {variant_folder} 复制文件（添加后缀{idx}）:")
        
        # SurveyForge-main
        process_surveyforge_folder(variant_folder, target_dir, str(idx))
        
        # SurveyX-main
        process_surveyx_folder(variant_folder, target_dir, str(idx))
        
        # AutoSurvey-main
        process_autosurvey_folder(variant_folder, target_dir, str(idx))

def main():
    """主函数"""
    print("=" * 60)
    print("开始复制文件到 EVAL 目录")
    print("=" * 60)
    
    # 处理 t1 到 t20
    for i in range(1, 21):
        folder_name = f"t{i}"
        process_folder(folder_name)
    
    print("\n" + "=" * 60)
    print("所有文件复制完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

