#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用 topicget.py 处理多个 topic
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="批量调用 topicget.py 处理多个 topic"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="处理的 topic 数量（默认: 3）"
    )
    parser.add_argument(
        "--topic-file",
        type=str,
        default="topic.json",
        help="topic 文件路径（默认: topic.json）"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="../",
        help="EVAL 目录路径（默认: ../）"
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    topic_file = script_dir / args.topic_file
    base_dir = script_dir / args.base_dir
    topicget_script = script_dir / "topicget.py"
    
    # 检查文件是否存在
    if not topic_file.exists():
        print(f"错误: topic 文件 {topic_file} 不存在")
        sys.exit(1)
    
    if not topicget_script.exists():
        print(f"错误: topicget.py 文件 {topicget_script} 不存在")
        sys.exit(1)
    
    # 读取 topic.json
    try:
        with open(topic_file, 'r', encoding='utf-8') as f:
            topics = json.load(f)
    except Exception as e:
        print(f"错误: 读取 topic 文件失败: {e}")
        sys.exit(1)
    
    # 检查数据格式
    if not isinstance(topics, list):
        print(f"错误: topic 文件格式不正确，应该是数组")
        sys.exit(1)
    
    # 检查数量
    n = args.n
    if n > len(topics):
        print(f"警告: 请求处理 {n} 个 topic，但文件只有 {len(topics)} 个，将处理 {len(topics)} 个")
        n = len(topics)
    
    if n <= 0:
        print(f"错误: n 必须大于 0")
        sys.exit(1)
    
    print(f"开始处理 {n} 个 topic...")
    print(f"基础目录: {base_dir.absolute()}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    # 处理前 n 个 topic
    for i in range(n):
        topic_data = topics[i]
        topic_id = topic_data.get("id", i + 1)
        topic = topic_data.get("topic", "")
        
        if not topic:
            print(f"\n[跳过] 第 {i+1} 个条目 (id={topic_id}): topic 为空")
            fail_count += 1
            continue
        
        # 创建对应的文件夹 t1, t2, t3...
        folder_name = f"t{topic_id}"
        output_dir = base_dir / folder_name
        output_file = output_dir / "paper.json"
        
        # 创建目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{i+1}/{n}] 处理 topic (id={topic_id}):")
        print(f"  标题: {topic[:80]}{'...' if len(topic) > 80 else ''}")
        print(f"  输出目录: {output_dir.absolute()}")
        
        # 构建命令
        # 计算相对于 topicget.py 所在目录（script_dir）的路径
        try:
            output_path_relative = output_file.relative_to(script_dir)
        except ValueError:
            # 如果无法计算相对路径，使用绝对路径
            output_path_relative = output_file
        
        cmd = [
            sys.executable,
            str(topicget_script),
            topic,
            "-o", str(output_path_relative)
        ]
        
        # 执行命令
        try:
            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # 检查输出中是否包含"未找到匹配的数据"
            output_text = result.stdout + result.stderr
            not_found_keywords = ["未找到匹配的数据", "✗ 未找到匹配的数据", "未找到匹配"]
            
            if any(keyword in output_text for keyword in not_found_keywords):
                print(f"  ⚠ 警告: 在数据库中未找到匹配的数据")
                print(f"    标题: {topic}")
                print(f"    请检查标题是否正确或数据库中是否存在该论文")
                fail_count += 1
            elif result.returncode == 0:
                print(f"  ✓ 成功")
                success_count += 1
            else:
                print(f"  ✗ 失败 (返回码: {result.returncode})")
                if result.stderr:
                    print(f"  错误信息: {result.stderr[:200]}")
                fail_count += 1
        except Exception as e:
            print(f"  ✗ 执行失败: {e}")
            fail_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"  成功: {success_count}/{n}")
    print(f"  失败: {fail_count}/{n}")
    print(f"输出目录: {base_dir.absolute()}")


if __name__ == "__main__":
    main()

