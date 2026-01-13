#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 title_433.jsonl 中根据 literature_review_id 匹配并提取 structure 内容
为 t1-t30 文件夹生成对应的 outline.json 文件
"""

import json
import os
from pathlib import Path

# 设置路径
# 脚本位于 EVAL/code/outlinegetALL.py，可以从 EVAL/code 目录直接运行
SCRIPT_DIR = Path(__file__).parent.absolute()  # EVAL/code 目录
BASE_DIR = SCRIPT_DIR.parent.absolute()  # EVAL 目录
JSONL_FILE = SCRIPT_DIR / "arxiv_dataset" / "title_433.jsonl"

def load_jsonl_data(jsonl_path):
    """加载 JSONL 文件，构建 literature_review_id 到 structure 的映射"""
    id_to_structure = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                literature_review_id = data.get('literature_review_id')
                structure = data.get('structure')
                if literature_review_id is not None and structure is not None:
                    id_to_structure[literature_review_id] = structure
            except json.JSONDecodeError as e:
                print(f"警告: 解析 JSONL 行时出错: {e}")
                continue
    return id_to_structure

def process_folder(folder_name, id_to_structure):
    """处理单个文件夹，生成 outline.json"""
    folder_path = BASE_DIR / folder_name
    paper_json_path = folder_path / "paper.json"
    outline_json_path = folder_path / "outline.json"
    
    # 检查 paper.json 是否存在
    if not paper_json_path.exists():
        print(f"提示: {folder_name}/paper.json 不存在，无法生成 outline.json")
        return False
    
    # 读取 paper.json 获取 literature_review_id
    try:
        with open(paper_json_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 {folder_name}/paper.json: {e}")
        return False
    except Exception as e:
        print(f"错误: 读取 {folder_name}/paper.json 时出错: {e}")
        return False
    
    literature_review_id = paper_data.get('literature_review_id')
    if literature_review_id is None:
        print(f"提示: {folder_name}/paper.json 中没有找到 literature_review_id，无法生成 outline.json")
        return False
    
    # 在映射中查找匹配的 structure
    if literature_review_id not in id_to_structure:
        print(f"提示: {folder_name} 的 literature_review_id {literature_review_id} 在 title_433.jsonl 中未找到匹配项，无法生成 outline.json")
        return False
    
    structure = id_to_structure[literature_review_id]
    
    # 保存到 outline.json
    try:
        with open(outline_json_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        print(f"成功: {folder_name}/outline.json 已生成")
        return True
    except Exception as e:
        print(f"错误: 写入 {folder_name}/outline.json 时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始处理...")
    
    # 检查 JSONL 文件是否存在
    if not JSONL_FILE.exists():
        print(f"错误: 找不到文件 {JSONL_FILE}")
        return
    
    # 加载 JSONL 数据
    print(f"正在加载 {JSONL_FILE}...")
    id_to_structure = load_jsonl_data(JSONL_FILE)
    print(f"已加载 {len(id_to_structure)} 条记录")
    
    # 处理 t1-t30 文件夹
    success_count = 0
    fail_count = 0
    
    for i in range(1, 31):
        folder_name = f"t{i}"
        if process_folder(folder_name, id_to_structure):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n处理完成!")
    print(f"成功: {success_count} 个文件夹")
    print(f"失败: {fail_count} 个文件夹")

if __name__ == "__main__":
    main()

