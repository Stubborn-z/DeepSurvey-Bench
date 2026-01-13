#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 arxiv_433.jsonl 中根据 literature_review_title 完全匹配查找数据
"""

import json
import argparse
import os
from pathlib import Path


def find_topic(topic: str, input_file: str = "arxiv_dataset/arxiv_433.jsonl", output_file: str = "paper.json"):
    """
    在 arxiv_433.jsonl 中查找匹配 literature_review_title 的数据（大小写不敏感）
    
    Args:
        topic: 要匹配的标题（大小写不敏感匹配）
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    print(f"正在搜索: {topic}")
    print(f"输入文件: {input_file}")
    
    found = False
    line_count = 0
    
    # 逐行读取，避免一次性加载大文件到内存
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count = line_num
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # 检查 literature_review_title 是否匹配（大小写不敏感）
                    title = data.get("literature_review_title", "")
                    if title.casefold() == topic.casefold():
                        # 找到匹配的数据，确保输出目录存在
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        # 写入输出文件
                        try:
                            with open(output_path, 'w', encoding='utf-8') as out_f:
                                json.dump(data, out_f, ensure_ascii=False, indent=2)
                            print(f"✓ 找到匹配数据！")
                            print(f"  行号: {line_num}")
                            print(f"  literature_review_id: {data.get('literature_review_id')}")
                            print(f"  输出文件: {output_path.absolute()}")
                            print(f"  文件已成功创建")
                            found = True
                        except Exception as write_error:
                            print(f"错误: 写入文件失败: {write_error}")
                            return False
                        break
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                    continue
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return False
    
    if not found:
        print(f"✗ 未找到匹配的数据")
        print(f"  已搜索 {line_count} 行")
        print(f"  请检查标题是否正确（匹配时忽略大小写）")
    
    return found


def main():
    parser = argparse.ArgumentParser(
        description="从 arxiv_433.jsonl 中根据 literature_review_title 匹配查找数据（大小写不敏感）"
    )
    parser.add_argument(
        "topic",
        type=str,
        help="要匹配的文献综述标题（大小写不敏感匹配）"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="arxiv_dataset/arxiv_433.jsonl",
        help="输入文件路径（默认: arxiv_dataset/arxiv_433.jsonl）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="paper.json",
        help="输出文件路径（默认: paper.json）"
    )
    
    args = parser.parse_args()
    
    find_topic(args.topic, args.input, args.output)


if __name__ == "__main__":
    main()

