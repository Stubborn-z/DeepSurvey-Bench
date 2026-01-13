#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import argparse
from openai import OpenAI

# 估算token数量（粗略估算：1 token ≈ 4 字符）
def estimate_tokens(text):
    """粗略估算文本的token数量"""
    if not text:
        return 0
    return len(text) // 4

# 截断内容以适应上下文长度限制
def truncate_content(content, max_tokens=100000, prompt_tokens=5000):
    """
    截断内容以适应模型的上下文长度限制
    max_tokens: 模型的最大上下文长度（默认100000，留一些余量）
    prompt_tokens: 提示词大约占用的token数（默认5000）
    """
    if not content:
        return content
    
    # 计算可用空间
    available_tokens = max_tokens - prompt_tokens
    
    # 估算当前内容的token数
    current_tokens = estimate_tokens(content)
    
    if current_tokens <= available_tokens:
        return content
    
    # 需要截断
    # 按字符数截断（保留90%的可用空间，更安全）
    max_chars = int(available_tokens * 4 * 0.9)
    
    # 截断内容，保留开头部分
    truncated = content[:max_chars]
    
    # 尝试在句子边界截断（找到最后一个句号、问号或感叹号）
    last_sentence_end = max(
        truncated.rfind('。'),
        truncated.rfind('.'),
        truncated.rfind('！'),
        truncated.rfind('!'),
        truncated.rfind('？'),
        truncated.rfind('?')
    )
    
    if last_sentence_end > max_chars * 0.8:  # 如果找到的句子边界不太靠前
        truncated = truncated[:last_sentence_end + 1]
    
    print(f"  警告: 内容过长 ({current_tokens} tokens)，已截断至约 {estimate_tokens(truncated)} tokens")
    
    return truncated

# 获取API key（多行则取最后一行）
def get_api_key():
    akey_path = os.path.join(os.path.dirname(__file__), 'akey.txt')
    with open(akey_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines[-1].strip() if lines else None

# 读取JSON文件
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 将paper.json的structure转换为survey格式
def convert_paper_to_survey(paper_data):
    """将paper.json的structure数组转换为survey字符串"""
    structure = paper_data.get('structure', [])
    if not structure:
        return None
    
    survey_parts = []
    
    for section in structure:
        section_title = section.get('section_title', '')
        level = section.get('level', '1')
        content = section.get('content', '')
        
        # 跳过Pre-section等不需要的标题，但保留内容
        if 'Pre-section' in section_title or not section_title.strip():
            if content.strip():
                survey_parts.append(content.strip())
            continue
        
        # 根据level生成Markdown标题
        try:
            level_int = int(level)
            if level_int == 1:
                survey_parts.append(f"# {section_title}")
            elif level_int == 2:
                survey_parts.append(f"## {section_title}")
            elif level_int == 3:
                survey_parts.append(f"### {section_title}")
            elif level_int == 4:
                survey_parts.append(f"#### {section_title}")
            else:
                survey_parts.append(f"{'#' * level_int} {section_title}")
        except:
            # 如果level不是数字，默认使用##级别
            survey_parts.append(f"## {section_title}")
        
        # 添加内容
        if content.strip():
            survey_parts.append(content.strip())
    
    return '\n\n'.join(survey_parts)

# 将paper.json转换为Paper.json格式
def convert_paper_json(eval_dir):
    """将paper.json转换为Paper.json格式"""
    paper_path = os.path.join(eval_dir, 'paper.json')
    paper_json_path = os.path.join(eval_dir, 'Paper.json')
    
    # 如果Paper.json已存在，跳过
    if os.path.exists(paper_json_path):
        print(f"  Paper.json 已存在，跳过转换")
        return True
    
    if not os.path.exists(paper_path):
        print(f"  警告: {paper_path} 不存在")
        return False
    
    try:
        paper_data = load_json(paper_path)
        survey_content = convert_paper_to_survey(paper_data)
        
        if not survey_content:
            print(f"  警告: 无法从paper.json提取survey内容")
            return False
        
        # 创建Paper.json，格式类似a.json
        paper_json_data = {
            "survey": survey_content
        }
        
        with open(paper_json_path, 'w', encoding='utf-8') as f:
            json.dump(paper_json_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Paper.json 已创建")
        return True
    except Exception as e:
        print(f"  错误: 转换paper.json失败: {e}")
        return False

# 读取promptpaperour-yange.txt，解析7个维度的提示词
def load_paperour_prompts():
    prompt_path = os.path.join(os.path.dirname(__file__), 'promptpaperour-yange.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析7个维度，包含标题
    dimensions = []
    
    # 1. Objective Clarity (1.1)
    match = re.search(r'(## 1\.1 Objective Clarity.*?)(?=## 1\.2|$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    # 2. Classification-Evolution Coherence (1.2)
    match = re.search(r'(## 1\.2 Classification-Evolution Coherence.*?)(?=## 1\.3|$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    # 3. Dataset & Metric Coverage (1.3)
    match = re.search(r'(## 1\.3 Dataset & Metric Coverage.*?)(?=## 2\.1|$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    # 4. In-depth Comparison (2.1)
    match = re.search(r'(## 2\.1 In-depth Comparison.*?)(?=## 2\.2|$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    # 5. Critical Analysis (2.2)
    match = re.search(r'(## 2\.2 Critical Analysis.*?)(?=## 3\.1|$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    # 6. Research Gaps (3.1)
    match = re.search(r'(## 3\.1 Research Gaps.*?)(?=## 3\.2|$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    # 7. Prospectiveness (3.2)
    match = re.search(r'(## 3\.2 Prospectiveness.*?)(?=$)', content, re.DOTALL)
    if match:
        dimensions.append(match.group(1).strip())
    
    if len(dimensions) != 7:
        print(f"警告: 只提取到 {len(dimensions)} 个维度，期望7个")
    
    return dimensions

# 调用LLM API获取分数和完整回复
def get_score_from_llm(client, prompt, return_response=False, max_retries=3, timeout=120, model="gpt-5", temperature=0.0):
    """调用LLM API，支持重试和超时设置"""
    import time
    
    for attempt in range(max_retries):
        try:
            # 设置超时时间和模型参数
            create_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                "timeout": timeout,
                "temperature": temperature
            }
            
            completion = client.chat.completions.create(**create_params)
            response = completion.choices[0].message.content.strip()
            
            # 多种方式尝试提取分数
            # 1. 查找 "分数: X" 或 "score: X" 格式
            score_match = re.search(r'(?:分数|score|Score)[:：]\s*([1-5])', response, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                return (score, response) if return_response else score
            
            # 2. 查找 "X points" 或 "X分" 格式
            score_match = re.search(r'\b([1-5])\s*(?:points?|分)', response, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                return (score, response) if return_response else score
            
            # 3. 查找单独的数字1-5（优先查找开头的数字）
            score_match = re.search(r'^\s*([1-5])\b', response)
            if score_match:
                score = int(score_match.group(1))
                return (score, response) if return_response else score
            
            # 4. 查找任何1-5之间的数字
            score_match = re.search(r'\b([1-5])\b', response)
            if score_match:
                score = int(score_match.group(1))
                return (score, response) if return_response else score
            
            # 5. 如果都没找到，尝试查找任何数字并验证范围
            num_match = re.search(r'\b(\d+)\b', response)
            if num_match:
                score = int(num_match.group(1))
                if 1 <= score <= 5:
                    return (score, response) if return_response else score
            
            print(f"警告: 无法从响应中提取分数: {response[:200]}")
            return (None, response) if return_response else None
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间：5秒、10秒、15秒
                    print(f"  超时错误 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"错误: LLM API调用超时，已重试{max_retries}次")
                    return (None, None) if return_response else None
            else:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  错误 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"错误: LLM API调用失败: {error_msg}")
                    return (None, None) if return_response else None
    
    return (None, None) if return_response else None

# 评估paperour（7个维度）
def evaluate_paperour(client, topic, content, model="gpt-5", temperature=0.0):
    prompts = load_paperour_prompts()
    if not prompts or len(prompts) != 7:
        print("错误: 无法加载paperour提示词或维度数量不正确")
        return None, None
    
    dimension_names = [
        "Objective Clarity",
        "Classification-Evolution Coherence",
        "Dataset & Metric Coverage",
        "In-depth Comparison",
        "Critical Analysis",
        "Research Gaps",
        "Prospectiveness"
    ]
    
    # 根据模型设置最大上下文长度
    max_context_tokens = 128000 if "gpt-4o" in model.lower() else 200000
    
    scores = []
    reasons = []
    
    # 估算前缀和提示词的token数
    prefix_template = f"Here is an academic survey to evaluate about the topic {topic}:\n\n---\n\n\n\n---\n\n"
    prefix_tokens = estimate_tokens(prefix_template)
    
    for i, prompt_template in enumerate(prompts):
        # 估算提示词的token数
        prompt_tokens = estimate_tokens(prompt_template)
        total_prompt_tokens = prefix_tokens + prompt_tokens
        
        # 截断content以适应上下文限制
        truncated_content = truncate_content(content, max_tokens=max_context_tokens, prompt_tokens=total_prompt_tokens)
        
        # 组合前缀和提示词
        prefix = f"Here is an academic survey to evaluate about the topic {topic}:\n\n---\n\n{truncated_content}\n\n---\n\n"
        prompt = prefix + prompt_template
        
        print(f"  正在评估 {dimension_names[i]}...")
        score, response = get_score_from_llm(client, prompt, return_response=True, model=model, temperature=temperature)
        if score is not None:
            scores.append(score)
            reasons.append(response)
            print(f"    {dimension_names[i]}: {score}")
        else:
            print(f"    警告: {dimension_names[i]} 评估失败")
            scores.append(None)
            reasons.append(response if response else "")
    
    return scores, reasons

# 检查evalZour0.jsonl中是否已存在该实例的结果
def check_existing_result(eval_path, name):
    """检查evalZour0.jsonl中是否已存在该实例的结果"""
    if not os.path.exists(eval_path):
        return False
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('name') == name and 'paperour' in data:
                    return True
            except:
                continue
    
    return False

# 追加结果到evalZour0.jsonl
def append_result(filepath, result):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

# 评估单个文件（只评估paperour）
def evaluate_single_file(client, eval_dir, file_base_name, topic, model="gpt-5", temperature=0.0):
    """评估单个文件，只评估paperour"""
    file_path = os.path.join(eval_dir, f'{file_base_name}.json')
    eval_path = os.path.join(eval_dir, 'evalZour0.jsonl')
    
    # 检查是否已存在结果
    if check_existing_result(eval_path, file_base_name):
        print(f"  跳过 {file_base_name}.json (结果已存在)")
        return True
    
    if not os.path.exists(file_path):
        print(f"  警告: 文件 {file_path} 不存在，跳过")
        return False
    
    # 读取survey内容
    file_data = load_json(file_path)
    content = file_data.get('survey', '')
    
    if not content:
        print(f"  警告: 文件 {file_base_name}.json 中没有survey键，跳过")
        return False
    
    # 评估paperour
    print(f"  开始评估 paperour (7个维度)...")
    paperour_scores, paperour_reasons = evaluate_paperour(client, topic, content, model=model, temperature=temperature)
    
    if paperour_scores and all(s is not None for s in paperour_scores):
        result_our = {
            "name": file_base_name,
            "paperour": paperour_scores,
            "reason": paperour_reasons
        }
        append_result(eval_path, result_our)
        print(f"  paperour结果已保存: {result_our}")
        return True
    else:
        print(f"  警告: {file_base_name} 的paperour评估不完整")
        return False

# 评估Paper.json（name设为"G"）
def evaluate_paper_json(client, eval_dir, topic, model="gpt-5", temperature=0.0):
    """评估Paper.json，name设为G，只评估paperour"""
    result_name = "G"
    print(f"\n处理文件: Paper.json (name: {result_name})")
    
    eval_path = os.path.join(eval_dir, 'evalZour0.jsonl')
    
    # 检查是否已存在结果
    if check_existing_result(eval_path, result_name):
        print(f"  跳过 Paper.json (结果已存在)")
        return True
    
    # 先转换 paper.json 为 Paper.json
    if not convert_paper_json(eval_dir):
        print(f"  警告: 无法转换 paper.json")
        return False
    
    paper_json_path = os.path.join(eval_dir, 'Paper.json')
    if not os.path.exists(paper_json_path):
        print(f"  警告: Paper.json 不存在")
        return False
    
    paper_json_data = load_json(paper_json_path)
    content = paper_json_data.get('survey', '')
    
    if not content:
        print(f"  警告: Paper.json 中没有survey键")
        return False
    
    # 评估paperour
    print(f"  开始评估 paperour (7个维度)...")
    paperour_scores, paperour_reasons = evaluate_paperour(client, topic, content, model=model, temperature=temperature)
    
    if paperour_scores and all(s is not None for s in paperour_scores):
        result_our = {
            "name": result_name,
            "paperour": paperour_scores,
            "reason": paperour_reasons
        }
        append_result(eval_path, result_our)
        print(f"  paperour结果已保存: {result_our}")
        return True
    else:
        print(f"  警告: Paper.json 的paperour评估不完整")
        return False

# 主函数
def main():
    # 内部参数：要评估的文件列表（如果设置，则只评估列表中的文件）
    # 例如：target_files = ['a1', 'a2', 'G'] 表示只评估 a1.json, a2.json, G.json
    # 如果设为 None，则使用命令行参数或默认行为
    target_files = ['a2']  # 可以在这里直接设置，如：target_files = ['a1', 'a2', 'G']
    
    parser = argparse.ArgumentParser(description='评估论文质量（只评估paperour）')
    parser.add_argument('--dir', type=str, default='t1', help='评估目录，如 t1, t2 等')
    parser.add_argument('--file', type=str, default=None, help='要评估的文件名（不含.json），如 a, f, x, a1, f2, G 等')
    parser.add_argument('--files', type=str, default=None, help='要评估的文件列表（逗号分隔），如 a1,a2,G')
    
    args = parser.parse_args()
    
    # 获取路径
    base_dir = os.path.dirname(os.path.dirname(__file__))
    eval_dir = os.path.join(base_dir, args.dir)
    code_dir = os.path.dirname(__file__)
    
    # 获取API key并初始化客户端
    api_key = get_api_key()
    if not api_key:
        print("错误: 无法获取API key")
        return
    
    client = OpenAI(api_key=api_key, base_url="https://api.ai-gaochao.cn/v1")
    
    # 读取paper.json获取topic
    paper_path = os.path.join(eval_dir, 'paper.json')
    if not os.path.exists(paper_path):
        print(f"错误: 文件 {paper_path} 不存在")
        return
    
    paper_data = load_json(paper_path)
    topic = paper_data.get('literature_review_title', '')
    
    if not topic:
        print("错误: 无法从paper.json获取literature_review_title")
        return
    
    print(f"主题: {topic}\n")
    print(f"评估目录: {eval_dir}\n")
    print(f"模型: gpt-5, 温度: 0.0\n")
    
    # 确定要处理的文件列表
    # 优先使用内部设置的 target_files，如果设置了则忽略命令行参数
    if target_files is not None:
        # 使用内部设置的列表，忽略命令行参数
        file_list = target_files
        print(f"使用内部设置的文件列表: {file_list} (忽略命令行参数)\n")
        
        for file_name in file_list:
            if file_name.upper() == 'G' or file_name.lower() == 'paper':
                # 处理 Paper.json
                evaluate_paper_json(client, eval_dir, topic, model="gpt-5", temperature=0.0)
            else:
                # 处理普通文件
                print(f"\n处理文件: {file_name}.json")
                evaluate_single_file(client, eval_dir, file_name, topic, model="gpt-5", temperature=0.0)
    elif args.files:
        # 如果指定了文件列表，解析并处理
        file_list = [f.strip() for f in args.files.split(',')]
        print(f"处理指定文件列表: {file_list}\n")
        
        for file_name in file_list:
            if file_name.upper() == 'G' or file_name.lower() == 'paper':
                # 处理 Paper.json
                evaluate_paper_json(client, eval_dir, topic, model="gpt-5", temperature=0.0)
            else:
                # 处理普通文件
                print(f"\n处理文件: {file_name}.json")
                evaluate_single_file(client, eval_dir, file_name, topic, model="gpt-5", temperature=0.0)
    elif args.file:
        # 如果指定了单个文件，只处理该文件
        if args.file.lower() == 'paper' or args.file.upper() == 'G':
            # 处理 Paper.json
            evaluate_paper_json(client, eval_dir, topic, model="gpt-5", temperature=0.0)
        else:
            # 处理普通文件
            print(f"\n处理文件: {args.file}.json")
            evaluate_single_file(client, eval_dir, args.file, topic, model="gpt-5", temperature=0.0)
    else:
        # 默认处理 a.json, f.json, x.json, 以及 Paper.json
        files = ['a', 'f', 'x']
        
        for file_name in files:
            print(f"\n处理文件: {file_name}.json")
            evaluate_single_file(client, eval_dir, file_name, topic, model="gpt-5", temperature=0.0)
        
        # 处理 Paper.json
        evaluate_paper_json(client, eval_dir, topic, model="gpt-5", temperature=0.0)
    
    print("\n所有评估完成！")

if __name__ == '__main__':
    main()

