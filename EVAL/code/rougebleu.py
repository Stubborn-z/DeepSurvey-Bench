#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算ROUGE和BLEU指标的脚本
对a.json, x.json, f.json中的生成综述与paper.json中的真实综述进行比较
支持通过命令行参数指定具体的文件路径
"""

import json
import os
import sys
import argparse
from statistics import mean
from rouge_score import rouge_scorer
import sacrebleu
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def split_text_into_chunks(text, min_length=100):
    """
    将文本分割成片段，每个片段长度至少为min_length字符
    
    Args:
        text: 输入文本
        min_length: 最小片段长度
    
    Returns:
        文本片段列表
    """
    # 按段落分割（双换行符）
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_length = len(para)
        
        # 如果当前段落本身就很长，单独作为一个chunk
        if para_length >= min_length:
            # 先保存之前的chunk
            if current_length >= min_length:
                chunks.append('\n\n'.join(current_chunk))
            # 当前段落作为新chunk
            chunks.append(para)
            current_chunk = []
            current_length = 0
        else:
            # 累积段落
            current_chunk.append(para)
            current_length += para_length + 2  # +2 for '\n\n'
            
            # 如果累积长度足够，保存chunk
            if current_length >= min_length:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
    
    # 保存最后一个chunk（如果长度足够）
    if current_length >= min_length:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def sample_chunks(chunks, max_count=100, seed=42):
    """
    对片段列表进行均匀抽样
    
    Args:
        chunks: 片段列表
        max_count: 最大保留数量
        seed: 随机种子（默认42，确保可复现）
    
    Returns:
        抽样后的片段列表
    """
    if len(chunks) <= max_count:
        return chunks
    
    # 设置固定随机种子确保可复现
    random.seed(seed)
    # 均匀抽样
    indices = sorted(random.sample(range(len(chunks)), max_count))
    return [chunks[i] for i in indices]


def calculate_average_rouge_bleu(generated_chunks, reference_chunks, method_name=""):
    """
    计算生成文本和参考文本之间的平均ROUGE和BLEU分数
    
    Args:
        generated_chunks: 生成文本片段列表
        reference_chunks: 参考文本片段列表
        method_name: 方法名称（用于日志显示，不影响抽样）
    
    Returns:
        (rouge1, rouge2, rougeL, bleu) 元组
    """
    if not generated_chunks or not reference_chunks:
        return 0.0, 0.0, 0.0, 0.0
    
    # 对片段进行抽样，最多保留100个
    # 使用固定种子42确保可复现性
    max_chunks = 100
    if len(generated_chunks) > max_chunks:
        generated_chunks = sample_chunks(generated_chunks, max_chunks, seed=42)
    if len(reference_chunks) > max_chunks:
        reference_chunks = sample_chunks(reference_chunks, max_chunks, seed=42)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    
    total = len(generated_chunks)
    for idx, gen_chunk in enumerate(generated_chunks):
        if (idx + 1) % 10 == 0:
            print(f"    处理进度: {idx + 1}/{total}")
        
        max_rouge1, max_rouge2, max_rougeL = 0.0, 0.0, 0.0
        
        # 对每个生成片段，与所有参考片段比较，取最大分数
        for ref_chunk in reference_chunks:
            scores = scorer.score(ref_chunk, gen_chunk)
            max_rouge1 = max(max_rouge1, scores['rouge1'].fmeasure)
            max_rouge2 = max(max_rouge2, scores['rouge2'].fmeasure)
            max_rougeL = max(max_rougeL, scores['rougeL'].fmeasure)
        
        rouge1_scores.append(max_rouge1)
        rouge2_scores.append(max_rouge2)
        rougeL_scores.append(max_rougeL)
        
        # 计算BLEU分数（sacrebleu需要参考文本作为列表）
        bleu_score = sacrebleu.sentence_bleu(gen_chunk, reference_chunks).score
        bleu_scores.append(bleu_score)
    
    # 计算平均值
    avg_rouge1 = mean(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = mean(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = mean(rougeL_scores) if rougeL_scores else 0.0
    avg_bleu = mean(bleu_scores) if bleu_scores else 0.0
    
    return avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu


def load_generated_survey(json_path):
    """
    加载生成的综述文本
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        综述文本字符串
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('survey', '')


def clean_latex_content(text):
    """
    清理LaTeX内容，提取实际文本
    
    Args:
        text: 包含LaTeX命令的文本
    
    Returns:
        清理后的文本
    """
    import re
    # 如果LaTeX命令比例过高，返回空字符串
    if len(text) > 0 and text.count('\\') / len(text) > 0.15:
        return ""
    
    # 移除LaTeX命令（\command{...}）
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    # 移除单独的LaTeX命令（\command）
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # 移除特殊字符
    text = re.sub(r'[{}]', '', text)
    # 移除数学公式标记
    text = re.sub(r'\$[^$]*\$', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_reference_survey(json_path):
    """
    加载参考综述的结构化内容
    
    Args:
        json_path: paper.json文件路径
    
    Returns:
        参考文本片段列表（每个section的content）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reference_chunks = []
    structure = data.get('structure', [])
    
    for section in structure:
        content = section.get('content', '').strip()
        if not content:
            continue
        
        # 清理LaTeX内容
        cleaned_content = clean_latex_content(content)
        
        # 只保留长度>=100字符的内容
        if len(cleaned_content) >= 100:
            reference_chunks.append(cleaned_content)
    
    return reference_chunks


def evaluate_survey_worker(args):
    """
    工作函数，用于并行处理（不写入文件，只返回结果）
    
    Args:
        args: 元组 (method_name, generated_path, reference_path)
    
    Returns:
        (method_name, rouge1, rouge2, rougeL, bleu) 元组
    """
    method_name, generated_path, reference_path = args
    
    # 加载生成文本
    generated_text = load_generated_survey(generated_path)
    if not generated_text:
        return (method_name, 0.0, 0.0, 0.0, 0.0)
    
    # 加载参考文本
    reference_chunks = load_reference_survey(reference_path)
    if not reference_chunks:
        return (method_name, 0.0, 0.0, 0.0, 0.0)
    
    # 将生成文本分割成片段
    generated_chunks = split_text_into_chunks(generated_text, min_length=100)
    if not generated_chunks:
        return (method_name, 0.0, 0.0, 0.0, 0.0)
    
    # 计算指标（内部会进行抽样）
    rouge1, rouge2, rougeL, bleu = calculate_average_rouge_bleu(generated_chunks, reference_chunks, method_name)
    
    return (method_name, rouge1, rouge2, rougeL, bleu)


def evaluate_survey(method_name, generated_path, reference_path, output_path):
    """
    评估单个方法的生成综述
    
    Args:
        method_name: 方法名称（'a', 'x', 'f'）
        generated_path: 生成综述JSON文件路径
        reference_path: 参考综述JSON文件路径
        output_path: 输出结果文件路径
    """
    print(f"正在评估方法: {method_name}")
    
    # 加载生成文本
    generated_text = load_generated_survey(generated_path)
    if not generated_text:
        print(f"警告: {generated_path} 中没有找到survey字段")
        return
    
    # 加载参考文本
    reference_chunks = load_reference_survey(reference_path)
    if not reference_chunks:
        print(f"警告: {reference_path} 中没有找到有效的参考内容")
        return
    
    print(f"  生成文本长度: {len(generated_text)} 字符")
    print(f"  参考片段数量: {len(reference_chunks)}")
    
    # 将生成文本分割成片段
    generated_chunks = split_text_into_chunks(generated_text, min_length=100)
    print(f"  生成片段数量: {len(generated_chunks)}")
    
    if not generated_chunks:
        print(f"警告: 生成文本无法分割成有效片段")
        return
    
    # 显示抽样信息
    if len(generated_chunks) > 100:
        print(f"  生成片段将抽样至100个（从{len(generated_chunks)}个中均匀抽样）")
    if len(reference_chunks) > 100:
        print(f"  参考片段将抽样至100个（从{len(reference_chunks)}个中均匀抽样）")
    
    # 计算指标（内部会进行抽样）
    rouge1, rouge2, rougeL, bleu = calculate_average_rouge_bleu(generated_chunks, reference_chunks, method_name)
    
    print(f"  ROUGE-1: {rouge1:.4f}")
    print(f"  ROUGE-2: {rouge2:.4f}")
    print(f"  ROUGE-L: {rougeL:.4f}")
    print(f"  BLEU: {bleu:.4f}")
    
    # 追加结果到输出文件
    with open(output_path, 'a', encoding='utf-8') as f:
        # 写入ROUGE结果
        rouge_result = {
            "name": method_name,
            "rouge": [rouge1, rouge2, rougeL]
        }
        f.write(json.dumps(rouge_result, ensure_ascii=False) + '\n')
        
        # 写入BLEU结果
        bleu_result = {
            "name": method_name,
            "bleu": bleu
        }
        f.write(json.dumps(bleu_result, ensure_ascii=False) + '\n')
    
    print(f"  结果已追加到 {output_path}")


def evaluate_single_instance(method_name, generated_path, reference_path, output_path):
    """
    评估单个实例并立即写入结果
    
    Args:
        method_name: 方法名称（如 'a', 'a2', 'f3'等）
        generated_path: 生成综述JSON文件路径
        reference_path: 参考综述JSON文件路径
        output_path: 输出结果文件路径
    
    Returns:
        (rouge1, rouge2, rougeL, bleu) 元组，如果失败返回None
    """
    try:
        # 加载生成文本
        generated_text = load_generated_survey(generated_path)
        if not generated_text:
            print(f"警告: {generated_path} 中没有找到survey字段")
            return None
        
        # 加载参考文本
        reference_chunks = load_reference_survey(reference_path)
        if not reference_chunks:
            print(f"警告: {reference_path} 中没有找到有效的参考内容")
            return None
        
        # 将生成文本分割成片段
        generated_chunks = split_text_into_chunks(generated_text, min_length=100)
        if not generated_chunks:
            print(f"警告: 生成文本无法分割成有效片段")
            return None
        
        # 计算指标（内部会进行抽样）
        rouge1, rouge2, rougeL, bleu = calculate_average_rouge_bleu(generated_chunks, reference_chunks, method_name)
        
        # 立即写入结果
        with open(output_path, 'a', encoding='utf-8') as f:
            # 写入ROUGE结果
            rouge_result = {
                "name": method_name,
                "rouge": [rouge1, rouge2, rougeL]
            }
            f.write(json.dumps(rouge_result, ensure_ascii=False) + '\n')
            
            # 写入BLEU结果
            bleu_result = {
                "name": method_name,
                "bleu": bleu
            }
            f.write(json.dumps(bleu_result, ensure_ascii=False) + '\n')
        
        return (rouge1, rouge2, rougeL, bleu)
    except Exception as e:
        print(f"评估 {method_name} 时出错: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='评估ROUGE和BLEU指标')
    parser.add_argument('--generated', type=str, help='生成综述JSON文件路径（可选）')
    parser.add_argument('--reference', type=str, help='参考综述JSON文件路径（可选）')
    parser.add_argument('--output', type=str, help='输出结果文件路径（可选）')
    parser.add_argument('--method', type=str, help='方法名称（可选，如a, a2, f3等）')
    parser.add_argument('--t_dir', type=str, help='t目录路径（可选，如t1, t2等）')
    
    args = parser.parse_args()
    
    # 如果提供了参数，使用参数模式（单实例评估）
    if args.generated and args.reference and args.output and args.method:
        print(f"评估实例: {args.method}")
        print(f"生成文件: {args.generated}")
        print(f"参考文件: {args.reference}")
        print(f"输出文件: {args.output}")
        print("=" * 60)
        
        result = evaluate_single_instance(args.method, args.generated, args.reference, args.output)
        if result:
            rouge1, rouge2, rougeL, bleu = result
            print(f"评估完成: {args.method}")
            print(f"  ROUGE-1: {rouge1:.4f}")
            print(f"  ROUGE-2: {rouge2:.4f}")
            print(f"  ROUGE-L: {rougeL:.4f}")
            print(f"  BLEU: {bleu:.4f}")
        else:
            print(f"评估失败: {args.method}")
        return
    
    # 默认模式：处理t1目录下的a.json, x.json, f.json（保持向后兼容）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t1_dir = os.path.join(base_dir, 't1')
    
    # 文件路径
    a_path = os.path.join(t1_dir, 'a.json')
    x_path = os.path.join(t1_dir, 'x.json')
    f_path = os.path.join(t1_dir, 'f.json')
    paper_path = os.path.join(t1_dir, 'paper.json')
    output_path = os.path.join(t1_dir, 'eval.jsonl')
    
    # 检查文件是否存在
    for path, name in [(a_path, 'a.json'), (x_path, 'x.json'), 
                       (f_path, 'f.json'), (paper_path, 'paper.json')]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 {path}")
            return
    
    print("开始评估ROUGE和BLEU指标...")
    print("=" * 60)
    
    # 评估三种方法
    methods = [
        ('a', a_path),
        ('x', x_path),
        ('f', f_path)
    ]
    
    # 使用并行处理加速评估
    num_workers = min(3, multiprocessing.cpu_count())  # 最多使用3个进程（对应3种方法）
    print(f"使用 {num_workers} 个进程并行评估...")
    print()
    
    # 准备参数（不包含output_path，因为文件写入在主进程中进行）
    tasks = [(method_name, gen_path, paper_path) 
             for method_name, gen_path in methods]
    
    # 使用进程池并行执行
    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(evaluate_survey_worker, task): task[0] 
                   for task in tasks}
        
        # 等待所有任务完成并收集结果
        for future in as_completed(futures):
            method_name = futures[future]
            try:
                result = future.result()
                method_name, rouge1, rouge2, rougeL, bleu = result
                results[method_name] = (rouge1, rouge2, rougeL, bleu)
                print(f"方法 {method_name} 评估完成")
                print(f"  ROUGE-1: {rouge1:.4f}")
                print(f"  ROUGE-2: {rouge2:.4f}")
                print(f"  ROUGE-L: {rougeL:.4f}")
                print(f"  BLEU: {bleu:.4f}")
                print()
            except Exception as exc:
                print(f"方法 {method_name} 评估出错: {exc}")
                print()
    
    # 在主进程中统一写入结果（避免并发写入问题）
    # 使用追加模式（'a'）写入，每次运行会追加到文件末尾
    print("正在写入结果...")
    with open(output_path, 'a', encoding='utf-8') as f:
        for method_name in ['a', 'x', 'f']:
            if method_name in results:
                rouge1, rouge2, rougeL, bleu = results[method_name]
                # 写入ROUGE结果
                rouge_result = {
                    "name": method_name,
                    "rouge": [rouge1, rouge2, rougeL]
                }
                f.write(json.dumps(rouge_result, ensure_ascii=False) + '\n')
                
                # 写入BLEU结果
                bleu_result = {
                    "name": method_name,
                    "bleu": bleu
                }
                f.write(json.dumps(bleu_result, ensure_ascii=False) + '\n')
    print("结果已追加写入文件")
    
    print()
    print("=" * 60)
    print("评估完成！")


if __name__ == '__main__':
    main()

