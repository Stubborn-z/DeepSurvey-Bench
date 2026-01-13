#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算HSR和HER指标的脚本
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


def convert_a_json_to_markdown(a_data: Dict, title: str) -> str:
    """
    从a.json的survey键中提取大纲并转换为标准Markdown格式
    survey键中：# 是标题，## 是一级章节，### 是二级章节等
    只提取标题行，忽略正文内容，并消除重复标题
    根据编号格式自动调整#的数量
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
                
                # 检查是否已存在（基于标题文本内容，忽略#的数量）
                if heading_text_lower not in seen_heading_texts:
                    # 根据标题文本确定正确的层级
                    level, is_number_heading = get_heading_level(heading_text, prev_number_level)
                    
                    # 只有数字开头的标题才更新 prev_number_level
                    # 非数字开头的标题统一使用 prev_number_level + 1，不更新 prev_number_level
                    if is_number_heading:
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


def process_single_file(base_dir: Path, file_type: str, file_suffix: str, outline_headings: List[str], title: str, compute_hsr: bool = True, compute_her: bool = True) -> Tuple[float, float, bool]:
    """
    处理单个文件（xo<b>.json, fo<b>.txt, 或 a<b>.json）
    
    参数:
        base_dir: 基础目录路径
        file_type: 文件类型 ('x', 'f', 'a')
        file_suffix: 文件编号后缀（空字符串或 '1'-'9'）
        outline_headings: 真实大纲的标题列表
        title: 文章标题
        compute_hsr: 是否计算HSR，默认为True
        compute_her: 是否计算HER，默认为True
    
    返回:
        (hsr, her, success): HSR值、HER值、是否成功处理（如果未计算则为None）
    """
    suffix_str = file_suffix if file_suffix else ""
    
    if file_type == 'x':
        # 处理 xo<b>.json
        xo_path = base_dir / f"xo{suffix_str}.json"
        if not xo_path.exists():
            return None, None, False
        
        xo_data = load_json(str(xo_path))
        xo_markdown = convert_xo_json_to_markdown(xo_data, title)
        xo_output_path = base_dir / f"xO{suffix_str}.txt"
        save_text(str(xo_output_path), xo_markdown)
        
        xo_headings = extract_headings_from_markdown(xo_markdown)
        hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=xo_headings) if compute_hsr else None
        her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=xo_headings) if compute_her else None
        print(f"已处理: {xo_path.name} -> {xo_output_path.name}, 提取到 {len(xo_headings)} 个标题")
        return hsr, her, True
        
    elif file_type == 'f':
        # 处理 fo<b>.txt
        fo_path = base_dir / f"fo{suffix_str}.txt"
        if not fo_path.exists():
            return None, None, False
        
        with open(fo_path, 'r', encoding='utf-8') as f:
            fo_content = f.read()
        fo_markdown = convert_fo_txt_to_markdown(fo_content, title)
        fo_output_path = base_dir / f"fO{suffix_str}.txt"
        save_text(str(fo_output_path), fo_markdown)
        
        fo_headings = extract_headings_from_markdown(fo_markdown)
        hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=fo_headings) if compute_hsr else None
        her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=fo_headings) if compute_her else None
        print(f"已处理: {fo_path.name} -> {fo_output_path.name}, 提取到 {len(fo_headings)} 个标题")
        return hsr, her, True
        
    elif file_type == 'a':
        # 处理 a<b>.json
        a_path = base_dir / f"a{suffix_str}.json"
        if not a_path.exists():
            return None, None, False
        
        a_data = load_json(str(a_path))
        a_markdown = convert_a_json_to_markdown(a_data, title)
        a_output_path = base_dir / f"aO{suffix_str}.txt"
        save_text(str(a_output_path), a_markdown)
        
        a_headings = extract_headings_from_markdown(a_markdown)
        hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=a_headings) if compute_hsr else None
        her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=a_headings) if compute_her else None
        print(f"已处理: {a_path.name} -> {a_output_path.name}, 提取到 {len(a_headings)} 个标题")
        return hsr, her, True
    
    return None, None, False


def main(t_id: int = 1, file_type: str = None, file_suffix: str = "", compute_hsr: bool = True, compute_her: bool = True):
    """
    主函数
    
    参数:
        t_id: t目录的编号（1-20），默认为1
        file_type: 文件类型 ('x', 'f', 'a')，默认为None（处理所有类型）
        file_suffix: 文件编号后缀（空字符串或 '1'-'9'），默认为空字符串
        compute_hsr: 是否计算HSR，默认为True
        compute_her: 是否计算HER，默认为True
    """
    # 设置路径
    base_dir = Path(__file__).parent.parent / f"t{t_id}"
    code_dir = Path(__file__).parent
    
    # 读取paper.json获取标题
    paper_path = base_dir / "paper.json"
    if not paper_path.exists():
        print(f"警告: {paper_path} 不存在，跳过 t{t_id}")
        return
    
    paper_data = load_json(str(paper_path))
    title = paper_data.get("literature_review_title", "A Survey of Large Language Models")
    
    # 处理outline.json (真实大纲)
    outline_path = base_dir / "outline.json"
    if not outline_path.exists():
        print(f"警告: {outline_path} 不存在，跳过 t{t_id}")
        return
    
    outline_data = load_json(str(outline_path))
    outline_markdown = convert_outline_json_to_markdown(outline_data, title)
    outline_output_path = base_dir / "Outline.txt"
    save_text(str(outline_output_path), outline_markdown)
    
    # 提取真实大纲的标题
    outline_headings = extract_headings_from_markdown(outline_markdown)
    
    # 如果没有指定file_type和file_suffix，保持原有行为（处理所有默认文件）
    if file_type is None and file_suffix == "":
        # 原有逻辑：处理 xo.json, fo.txt, a.json
        print(f"文章标题: {title}")
        results = []
        
        # 1. 处理xo.json (baseline x)
        print("\n处理 xo.json (baseline x)...")
        xo_path = base_dir / "xo.json"
        if xo_path.exists():
            xo_data = load_json(str(xo_path))
            xo_markdown = convert_xo_json_to_markdown(xo_data, title)
            xo_output_path = base_dir / "xO.txt"
            save_text(str(xo_output_path), xo_markdown)
            print(f"已保存到: {xo_output_path}")
            
            xo_headings = extract_headings_from_markdown(xo_markdown)
            print(f"提取到 {len(xo_headings)} 个标题")
            
            x_hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=xo_headings)
            x_her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=xo_headings)
            print(f"Baseline x - HSR: {x_hsr:.4f}, HER: {x_her:.4f}")
            
            results.extend([
                {"name": "x", "hsr": float(x_hsr)},
                {"name": "x", "her": float(x_her)}
            ])
        
        # 2. 处理fo.txt (baseline f)
        print("\n处理 fo.txt (baseline f)...")
        fo_path = base_dir / "fo.txt"
        if fo_path.exists():
            with open(fo_path, 'r', encoding='utf-8') as f:
                fo_content = f.read()
            fo_markdown = convert_fo_txt_to_markdown(fo_content, title)
            fo_output_path = base_dir / "fO.txt"
            save_text(str(fo_output_path), fo_markdown)
            print(f"已保存到: {fo_output_path}")
            
            fo_headings = extract_headings_from_markdown(fo_markdown)
            print(f"提取到 {len(fo_headings)} 个标题")
            
            f_hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=fo_headings)
            f_her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=fo_headings)
            print(f"Baseline f - HSR: {f_hsr:.4f}, HER: {f_her:.4f}")
            
            results.extend([
                {"name": "f", "hsr": float(f_hsr)},
                {"name": "f", "her": float(f_her)}
            ])
        
        # 3. 处理a.json (baseline a)
        print("\n处理 a.json (baseline a)...")
        a_path = base_dir / "a.json"
        if a_path.exists():
            a_data = load_json(str(a_path))
            a_markdown = convert_a_json_to_markdown(a_data, title)
            a_output_path = base_dir / "aO.txt"
            save_text(str(a_output_path), a_markdown)
            print(f"已保存到: {a_output_path}")
            
            a_headings = extract_headings_from_markdown(a_markdown)
            print(f"提取到 {len(a_headings)} 个标题")
            
            a_hsr = heading_soft_recall(golden_headings=outline_headings, predicted_headings=a_headings)
            a_her = heading_entity_recall(golden_headings=outline_headings, predicted_headings=a_headings)
            print(f"Baseline a - HSR: {a_hsr:.4f}, HER: {a_her:.4f}")
            
            results.extend([
                {"name": "a", "hsr": float(a_hsr)},
                {"name": "a", "her": float(a_her)}
            ])
        
        # 保存结果
        if results:
            eval_output_path = base_dir / "eval.jsonl"
            with open(eval_output_path, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"\n结果已保存到: {eval_output_path}")
        
        print("\n完成！")
    else:
        # 新逻辑：处理指定的文件
        # 确定要处理的文件类型
        file_types = [file_type] if file_type else ['x', 'f', 'a']
        
        results = []
        
        for ft in file_types:
            hsr, her, success = process_single_file(base_dir, ft, file_suffix, outline_headings, title, compute_hsr, compute_her)
            if success:
                name = f"{ft}{file_suffix}" if file_suffix else ft
                if hsr is not None:
                    results.append({"name": name, "hsr": float(hsr)})
                if her is not None:
                    results.append({"name": name, "her": float(her)})
                hsr_str = f"HSR={hsr:.4f}" if hsr is not None else ""
                her_str = f"HER={her:.4f}" if her is not None else ""
                metrics_str = ", ".join(filter(None, [hsr_str, her_str]))
                print(f"处理完成: {name} - {metrics_str}")
            else:
                name = f"{ft}{file_suffix}" if file_suffix else ft
                suffix_str = file_suffix if file_suffix else ""
                if ft == 'x':
                    file_path = base_dir / f"xo{suffix_str}.json"
                elif ft == 'f':
                    file_path = base_dir / f"fo{suffix_str}.txt"
                elif ft == 'a':
                    file_path = base_dir / f"a{suffix_str}.json"
                else:
                    file_path = None
                if file_path:
                    print(f"跳过 {name}: 文件 {file_path.name} 不存在")
        
        # 保存结果到eval.jsonl（追加模式）
        if results:
            eval_output_path = base_dir / "eval.jsonl"
            with open(eval_output_path, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"结果已保存到: {eval_output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='计算HSR和HER指标')
    parser.add_argument('--t_id', type=int, default=1, help='t目录的编号（1-20），默认为1')
    parser.add_argument('--file_type', type=str, default=None, choices=['x', 'f', 'a'], help='文件类型 (x/f/a)，默认为None（处理所有类型）')
    parser.add_argument('--file_suffix', type=str, default="", help='文件编号后缀（空字符串或1-9），默认为空字符串')
    parser.add_argument('--compute_hsr', action='store_true', default=True, help='是否计算HSR，默认为True')
    parser.add_argument('--no_compute_hsr', dest='compute_hsr', action='store_false', help='不计算HSR')
    parser.add_argument('--compute_her', action='store_true', default=True, help='是否计算HER，默认为True')
    parser.add_argument('--no_compute_her', dest='compute_her', action='store_false', help='不计算HER')
    args = parser.parse_args()
    main(t_id=args.t_id, file_type=args.file_type, file_suffix=args.file_suffix, compute_hsr=args.compute_hsr, compute_her=args.compute_her)

