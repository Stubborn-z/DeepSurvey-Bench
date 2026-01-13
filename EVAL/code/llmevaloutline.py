#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import argparse
from openai import OpenAI

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AKEY_FILE = os.path.join(BASE_DIR, "code", "akey.txt")
PROMPT_FILE = os.path.join(BASE_DIR, "code", "promptoutline.txt")

# 模型配置
MODEL = "gpt-5"
BASE_URL = "https://api.ai-gaochao.cn/v1"

# 评价维度（按顺序）
DIMENSIONS = [
    "><Guidance for Content Generation",
    "><Hierarchical Clarity",
    "><Logical Coherence"
]


def read_api_key():
    """从akey.txt读取API key，如果有多行则取最后一行"""
    with open(AKEY_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines[-1] if lines else None


def parse_prompt_templates():
    """解析promptoutline.txt，提取3个维度的提示词模板"""
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    templates = {}
    # 按维度分割内容（使用维度标志作为分隔符）
    current_dim = None
    current_template = []
    lines = content.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        # 检查是否是维度标志（以><开头）
        is_dimension = False
        for dim in DIMENSIONS:
            if line_stripped.startswith(dim):
                # 如果之前有正在处理的维度，保存它
                if current_dim and current_template:
                    templates[current_dim] = '\n'.join(current_template).strip()
                # 开始新的维度
                current_dim = dim
                current_template = []
                # 跳过维度标志这一行，从下一行开始
                is_dimension = True
                break
        
        if not is_dimension and current_dim:
            current_template.append(line)
    
    # 保存最后一个维度
    if current_dim and current_template:
        templates[current_dim] = '\n'.join(current_template).strip()
    
    return templates


def read_outline(filename, outline_dir):
    """读取大纲文件内容"""
    filepath = os.path.join(outline_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def read_topic_from_paper(paper_json):
    """从paper.json读取literature_review_title作为topic"""
    try:
        with open(paper_json, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        topic = paper_data.get("literature_review_title", "Large Language Models")
        return topic
    except Exception as e:
        print(f"警告: 无法读取paper.json，使用默认主题: {e}")
        return "Large Language Models"


def call_llm(client, prompt, max_retries=3):
    """调用LLM API获取评分，带重试机制和超时设置"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion.choices[0].message.content.strip()
            # 尝试提取数字分数
            # 优先尝试解析整个响应为单个数字
            try:
                score = int(response.strip())
                if 1 <= score <= 5:
                    return score
            except:
                pass
            
            # 如果整个响应不是单个数字，尝试查找独立的1-5数字
            # 使用单词边界或空格来确保是独立的数字
            score_match = re.search(r'\b([1-5])\b', response)
            if score_match:
                return int(score_match.group(1))
            
            # 如果还是没找到，尝试查找任何1-5的数字（作为后备方案）
            score_match = re.search(r'[1-5]', response)
            if score_match:
                return int(score_match.group())
            
            # 默认返回3（如果无法解析）
            print(f"警告: 无法解析LLM响应为分数，响应内容: {response}")
            return 3
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间：5秒、10秒、15秒
                    print(f"  超时，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  超时，已达到最大重试次数，使用默认分数3")
                    return 3
            else:
                print(f"调用LLM API时出错: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  {wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    return 3
    
    return 3


def evaluate_outline(outline_name, outline_content, topic, templates, client):
    """评价一个大纲，返回3个维度的分数"""
    scores = []
    
    print(f"\n评价大纲: {outline_name}")
    print(f"主题: {topic}")
    
    for dim in DIMENSIONS:
        template = templates[dim]
        # 替换占位符：{topic} 替换为从paper.json读取的主题，{content} 替换为大纲内容
        prompt = template.replace("{topic}", topic).replace("{content}", outline_content)
        
        # 验证替换是否成功（检查是否还有未替换的占位符）
        if "{topic}" in prompt or "{content}" in prompt:
            print(f"警告: {dim} 的提示词中仍有未替换的占位符！")
            print(f"  未替换的占位符: {[k for k in ['{topic}', '{content}'] if k in prompt]}")
        
        # 调用LLM获取分数
        score = call_llm(client, prompt)
        scores.append(score)
        
        print(f"  {dim}: {score}")
    
    return scores


def evaluate_single_file(filename, name, outline_dir, output_file, paper_json, templates, client, verbose=True):
    """评价单个大纲文件"""
    filepath = os.path.join(outline_dir, filename)
    if not os.path.exists(filepath):
        if verbose:
            print(f"警告: 文件不存在 {filepath}")
        return None
    
    # 读取大纲内容
    outline_content = read_outline(filename, outline_dir)
    
    # 从paper.json读取topic
    topic = read_topic_from_paper(paper_json)
    
    if verbose:
        print(f"从paper.json读取的主题: {topic}")
    
    # 评价大纲
    scores = evaluate_outline(name, outline_content, topic, templates, client)
    
    # 输出结果到终端
    if verbose:
        print(f"\n{name}的得分:")
        for dim, score in zip(DIMENSIONS, scores):
            print(f"  {dim}-{score}")
    
    # 保存结果到JSONL文件
    result = {
        "name": name,
        "outline": scores
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    if verbose:
        print(f"结果已追加到 {output_file}")
    
    return scores


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估大纲文件')
    parser.add_argument('--dir', '-d', type=str, default='t1', help='目录名（如t1, t2等），默认为t1')
    parser.add_argument('--file', '-f', type=str, nargs='+', help='要评估的文件名（如fO.txt, fO1.txt等），可指定多个。如果不指定，则使用默认的fO.txt, xO.txt, aO.txt')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式，减少输出')
    args = parser.parse_args()
    
    # 设置路径
    outline_dir = os.path.join(BASE_DIR, args.dir)
    output_file = os.path.join(outline_dir, "eval.jsonl")
    paper_json = os.path.join(outline_dir, "paper.json")
    
    verbose = not args.quiet
    
    # 读取API key
    api_key = read_api_key()
    if not api_key:
        print("错误: 无法读取API key")
        return
    
    # 初始化OpenAI客户端（设置超时时间）
    client = OpenAI(
        api_key=api_key, 
        base_url=BASE_URL,
        timeout=120.0  # 设置120秒超时
    )
    
    # 解析提示词模板
    templates = parse_prompt_templates()
    if len(templates) != 3:
        print(f"错误: 无法解析3个维度的提示词模板，只找到{len(templates)}个")
        return
    
    # 验证模板格式（仅在verbose模式下）
    if verbose:
        print("\n验证提示词模板格式:")
        for dim in DIMENSIONS:
            if dim in templates:
                template = templates[dim]
                has_topic = "{topic}" in template
                has_content = "{content}" in template
                has_dash = "--" in template
                print(f"  {dim}: 包含{{topic}}={has_topic}, 包含{{content}}={has_content}, 包含--={has_dash}")
            else:
                print(f"  {dim}: 未找到模板")
    
    # 确定要评价的文件
    if args.file:
        # 使用命令行指定的文件
        outline_files = []
        for f in args.file:
            # 检查是否是Outline.txt
            if f == "Outline.txt":
                outline_files.append((f, "G"))
            else:
                # 从文件名提取name（如fO.txt -> f, fO1.txt -> f1）
                match = re.match(r'([fax])O(\d*)\.txt', f)
                if match:
                    prefix = match.group(1)
                    suffix = match.group(2) if match.group(2) else ""
                    name = prefix + suffix
                    outline_files.append((f, name))
                else:
                    print(f"警告: 无法解析文件名 {f}，跳过")
    else:
        # 使用默认文件
        outline_files = [
            ("fO.txt", "f"),
            ("xO.txt", "x"),
            ("aO.txt", "a"),
            ("Outline.txt", "G")
        ]
    
    # 评价每个大纲
    for filename, name in outline_files:
        evaluate_single_file(filename, name, outline_dir, output_file, paper_json, templates, client, verbose)
    
    if verbose:
        print("\n评价完成!")


if __name__ == "__main__":
    main()

