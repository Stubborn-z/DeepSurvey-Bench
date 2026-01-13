#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算HSR和HER指标的脚本（简化版 - 只保留数字行、reference和title）
将不同格式的大纲转换为标准Markdown格式，并计算指标
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 从本地导入metrics模块（metrics.py在同一目录下）
from metrics import heading_soft_recall, heading_entity_recall


def load_json(file_path: str) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_text(file_path: str, content: str):
    """保存文本文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def convert_xo_json_to_markdown(xo_data: Dict, title: str) -> str:
    """
    将xo.json格式转换为Markdown格式
    xo.json结构: {title, sections: [{section title, subsections: [{subsection title}]}]}
    """
    lines = [f"# {title}", ""]
    
    for section in xo_data.get("sections", []):
        section_title = section.get("section title", "").strip()
        if section_title:
            lines.append(f"# {section_title}")
            lines.append("")
        
        subsections = section.get("subsections", [])
        for subsection in subsections:
            subsection_title = subsection.get("subsection title", "").strip()
            if subsection_title:
                lines.append(f"## {subsection_title}")
                lines.append("")
    
    return "\n".join(lines)


def convert_fo_txt_to_markdown(fo_content: str, title: str) -> str:
    """
    将fo.txt转换为标准Markdown格式
    只保留以#开头的行，舍弃其他描述性内容
    """
    lines = fo_content.split('\n')
    markdown_lines = []
    found_first_heading = False
    
    for line in lines:
        stripped = line.strip()
        # 只保留以#开头的行
        if stripped.startswith('#'):
            if not found_first_heading:
                # 替换第一个标题为文章标题
                markdown_lines.append(f"# {title}")
                markdown_lines.append("")
                found_first_heading = True
            else:
                # 保留其他标题行
                markdown_lines.append(line)
                markdown_lines.append("")
    
    # 如果没有找到任何标题，至少添加主标题
    if not found_first_heading:
        markdown_lines.append(f"# {title}")
        markdown_lines.append("")
    
    return "\n".join(markdown_lines)


def get_heading_level(heading_text: str, prev_number_level: int = 0) -> Tuple[int, bool]:
    """
    根据标题文本确定应该使用的#数量
    规则：
    - 主标题：1个 #
    - "References" 或 "reference"（大小写宽容的完全匹配）：固定 1个 #
    - 整数编号（如 "1 Introduction"）：2个 ##
    - 一位小数（如 "1.1 Definition"）：3个 ###
    - 两位小数（如 "1.1.1"）：4个 ####
    - 非数字开头的行：统一比上一个数字行多一个 #（不递增）
    
    返回：(level, is_number_heading)
    - level: 应该使用的#数量
    - is_number_heading: 是否是数字开头的标题
    """
    # 去除前导空格
    heading_text = heading_text.strip()
    
    # 检查是否是 "References" 或 "reference"（大小写宽容的完全匹配）
    if heading_text.lower() == "references":
        return (1, False)  # 固定使用 1 个 #
    
    # 检查是否以数字开头
    match = re.match(r'^(\d+(?:\.\d+)*)', heading_text)
    if match:
        # 有数字编号
        number_part = match.group(1)
        # 计算小数点的数量（即层级深度）
        dot_count = number_part.count('.')
        # 层级 = 2（整数）+ dot_count（小数点数量）
        level = 2 + dot_count
        return (level, True)
    else:
        # 非数字开头，统一比上一个数字行多一个#（不递增）
        if prev_number_level > 0:
            return (prev_number_level + 1, False)
        else:
            # 如果没有上一个数字层级，默认使用3个#
            return (3, False)


def convert_a_json_to_markdown_easy(a_data: Dict, title: str) -> str:
    """
    从a.json的survey键中提取大纲并转换为标准Markdown格式（简化版）
    只保留：数字行、reference行、title行
    舍弃：其他非数字行
    """
    survey_content = a_data.get("survey", "")
    if not survey_content:
        return f"# {title}\n\n"
    
    lines = survey_content.split('\n')
    markdown_lines = []
    seen_heading_texts = set()  # 用于跟踪已添加的标题文本（去除#），避免重复
    prev_number_level = 0  # 上一个数字标题的层级（用于非数字行的层级计算）
    
    # 查找第一个标题行并替换
    found_first_heading = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # 提取标题文本（去除所有#和空格）
            heading_text = stripped.strip('#').strip()
            
            if not found_first_heading:
                # 替换第一个标题为文章标题
                markdown_lines.append(f"# {title}")
                markdown_lines.append("")
                found_first_heading = True
                seen_heading_texts.add(title.lower())  # 记录主标题（转小写用于去重）
                prev_number_level = 1  # 主标题是1级
            else:
                # 使用标题文本（转小写）进行去重检查，忽略#的数量
                heading_text_lower = heading_text.lower()
                
                # 检查是否应该保留：
                # 1. 数字开头的行
                # 2. References/reference 行
                # 3. 其他非数字行舍弃
                is_number_heading = bool(re.match(r'^(\d+(?:\.\d+)*)', heading_text))
                is_references = heading_text_lower == "references"
                
                # 只保留数字行和References行
                if not (is_number_heading or is_references):
                    continue  # 舍弃非数字行（除了References）
                
                # 检查是否已存在（基于标题文本内容，忽略#的数量）
                if heading_text_lower not in seen_heading_texts:
                    # 根据标题文本确定正确的层级
                    level, is_number = get_heading_level(heading_text, prev_number_level)
                    
                    # 只有数字开头的标题才更新 prev_number_level
                    # References 和 非数字开头的标题不更新 prev_number_level
                    if is_number:
                        prev_number_level = level
                    
                    # 构建正确层级的标题
                    markdown_heading = "#" * level + " " + heading_text
                    markdown_lines.append(markdown_heading)
                    markdown_lines.append("")  # 添加空行以保持格式一致
                    seen_heading_texts.add(heading_text_lower)
                # 如果已存在，跳过（消除重复）
        # 跳过非标题内容（只保留标题结构）
    
    # 如果没有找到任何标题，至少添加主标题
    if not found_first_heading:
        markdown_lines.append(f"# {title}")
        markdown_lines.append("")
    
    return "\n".join(markdown_lines)


def convert_outline_json_to_markdown(outline_data: List, title: str) -> str:
    """
    将outline.json格式转换为Markdown格式
    outline.json结构: [{section_title, level, children: [...]}]
    """
    lines = [f"# {title}", ""]
    
    def process_node(node: Dict, current_level: int = 1):
        """递归处理节点"""
        section_title = node.get("section_title", "").strip()
        level = node.get("level", 1)
        
        # 跳过一些特殊标题
        if section_title in ["\\textsc{Coda", "Acknowledgments", "Acknowledgment"]:
            return
        
        if section_title:
            # 根据level确定#的数量
            markdown_level = "#" * level
            lines.append(f"{markdown_level} {section_title}")
            lines.append("")
        
        # 处理子节点
        children = node.get("children", [])
        for child in children:
            process_node(child, level + 1)
    
    for node in outline_data:
        process_node(node)
    
    return "\n".join(lines)


def extract_headings_from_markdown(markdown_content: str) -> List[str]:
    """
    从Markdown文本中提取所有标题
    使用与eval_outline_quality.py相同的逻辑
    """
    # 将数字编号格式转换为#
    content = re.sub(r"\d+\.\ ", '#', markdown_content)
    
    headings = []
    for line in content.split('\n'):
        line = line.strip()
        # 遇到References等关键词时停止
        if "# References" in line:
            break
        if line.startswith('#'):
            if any(keyword in line.lower() for keyword in ["references", "external links", "see also", "notes"]):
                break
            # 提取标题文本（去除#和空格）
            heading = line.strip('#').strip()
            if heading:
                headings.append(heading)
    
    return headings


def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent / "t1"
    code_dir = Path(__file__).parent
    
    # 读取paper.json获取标题
    paper_path = base_dir / "paper.json"
    paper_data = load_json(str(paper_path))
    title = paper_data.get("literature_review_title", "A Survey of Large Language Models")
    
    print(f"文章标题: {title}")
    
    # 1. 处理xo.json (baseline x)
    print("\n处理 xo.json (baseline x)...")
    xo_path = base_dir / "xo.json"
    xo_data = load_json(str(xo_path))
    xo_markdown = convert_xo_json_to_markdown(xo_data, title)
    xo_output_path = base_dir / "xO.txt"
    save_text(str(xo_output_path), xo_markdown)
    print(f"已保存到: {xo_output_path}")
    
    # 提取xo的标题
    xo_headings = extract_headings_from_markdown(xo_markdown)
    print(f"提取到 {len(xo_headings)} 个标题")
    
    # 2. 处理fo.txt (baseline f)
    print("\n处理 fo.txt (baseline f)...")
    fo_path = base_dir / "fo.txt"
    with open(fo_path, 'r', encoding='utf-8') as f:
        fo_content = f.read()
    fo_markdown = convert_fo_txt_to_markdown(fo_content, title)
    fo_output_path = base_dir / "fO.txt"
    save_text(str(fo_output_path), fo_markdown)
    print(f"已保存到: {fo_output_path}")
    
    # 提取fo的标题
    fo_headings = extract_headings_from_markdown(fo_markdown)
    print(f"提取到 {len(fo_headings)} 个标题")
    
    # 3. 处理outline.json (真实大纲)
    print("\n处理 outline.json (真实大纲)...")
    outline_path = base_dir / "outline.json"
    outline_data = load_json(str(outline_path))
    outline_markdown = convert_outline_json_to_markdown(outline_data, title)
    outline_output_path = base_dir / "Outline.txt"
    save_text(str(outline_output_path), outline_markdown)
    print(f"已保存到: {outline_output_path}")
    
    # 提取真实大纲的标题
    outline_headings = extract_headings_from_markdown(outline_markdown)
    print(f"提取到 {len(outline_headings)} 个标题")
    
    # 4. 计算HSR和HER指标
    print("\n计算指标...")
    
    # 对baseline x计算
    x_hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=xo_headings)
    x_her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=xo_headings)
    print(f"Baseline x - HSR: {x_hsr:.4f}, HER: {x_her:.4f}")
    
    # 对baseline f计算
    f_hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=fo_headings)
    f_her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=fo_headings)
    print(f"Baseline f - HSR: {f_hsr:.4f}, HER: {f_her:.4f}")
    
    # 3.5 处理a.json (baseline a) - 使用简化版转换函数
    print("\n处理 a.json (baseline a - 简化版，只保留数字行、reference和title)...")
    a_path = base_dir / "a.json"
    if a_path.exists():
        a_data = load_json(str(a_path))
        a_markdown = convert_a_json_to_markdown_easy(a_data, title)
        a_output_path = base_dir / "aO.txt"
        save_text(str(a_output_path), a_markdown)
        print(f"已保存到: {a_output_path}")
        
        # 提取a的标题
        a_headings = extract_headings_from_markdown(a_markdown)
        print(f"提取到 {len(a_headings)} 个标题")
        
        # 对baseline a计算
        a_hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=a_headings)
        a_her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=a_headings)
        print(f"Baseline a - HSR: {a_hsr:.4f}, HER: {a_her:.4f}")
    else:
        print(f"警告: {a_path} 不存在，跳过 baseline a")
        a_hsr = None
        a_her = None
    
    # 5. 保存结果到eval.jsonl（追加模式，因为其他算法也会往里面写）
    eval_output_path = base_dir / "eval.jsonl"
    # 将 numpy float32 转换为 Python float，以便 JSON 序列化
    results = [
        {"name": "x", "hsr": float(x_hsr)},
        {"name": "x", "her": float(x_her)},
        {"name": "f", "hsr": float(f_hsr)},
        {"name": "f", "her": float(f_her)}
    ]
    
    # 如果处理了baseline a，添加其结果
    if a_hsr is not None and a_her is not None:
        results.extend([
            {"name": "a", "hsr": float(a_hsr)},
            {"name": "a", "her": float(a_her)}
        ])
    
    # 使用追加模式 'a'，以便其他算法也能写入
    with open(eval_output_path, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n结果已保存到: {eval_output_path}")
    print("\n完成！")


if __name__ == "__main__":
    main()

