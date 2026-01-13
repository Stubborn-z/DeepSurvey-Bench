#!/usr/bin/env python3
"""
计算检索方法的 Recall@K 指标
"""

import json
import os
import argparse
import sys

def load_retrieveref(filepath):
    """从 JSON 文件中加载 retrieveref 字段，返回按顺序的 arxivid 列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    retrieveref = data.get('retrieveref', {})
    # 将字典转换为按键（数字字符串）排序的列表
    arxivid_list = []
    for key in sorted(retrieveref.keys(), key=lambda x: int(x)):
        arxivid = retrieveref[key]
        if arxivid:  # 只添加非空值
            arxivid_list.append(arxivid)
    
    return arxivid_list

def load_ground_truth_refs(filepath):
    """从 ref.jsonl 中加载所有非空的 arxivid 作为真实集合"""
    ground_truth = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                arxivid = item.get('arxivid', '')
                if arxivid:  # 只添加非空值
                    ground_truth.add(arxivid)
            except json.JSONDecodeError:
                continue
    
    return ground_truth

def calculate_recall_at_k(retrieved_list, ground_truth, k):
    """
    计算 Recall@K
    
    Args:
        retrieved_list: 检索到的 arxivid 列表（按顺序）
        ground_truth: 真实 arxivid 集合
        k: 前 k 个结果
    
    Returns:
        Recall@K 值
    """
    if len(ground_truth) == 0:
        return 0.0
    
    # 取前 k 个检索结果
    top_k = retrieved_list[:k]
    
    # 计算在前 k 个结果中有多少个在真实集合中
    hits = sum(1 for arxivid in top_k if arxivid in ground_truth)
    
    # Recall@K = 命中数 / 真实集合总数
    recall = hits / len(ground_truth)
    
    return recall

def evaluate_method(method_name, json_path, ref_jsonl_path, eval_jsonl_path, k_values, verbose=True):
    """
    评估单个方法的 Recall@K
    
    Args:
        method_name: 方法名称（如 'a', 'a1', 'f', 'f2' 等）
        json_path: JSON 文件路径
        ref_jsonl_path: ref.jsonl 文件路径
        eval_jsonl_path: eval.jsonl 文件路径
        k_values: K 值列表
        verbose: 是否打印详细信息
    
    Returns:
        结果字典，如果文件不存在或处理失败则返回 None
    """
    # 检查文件是否存在
    if not os.path.exists(json_path):
        if verbose:
            print(f"错误: 文件不存在: {json_path}")
        return None
    
    if verbose:
        print(f"正在评估文件: {json_path}")
    
    try:
        # 加载真实参考文献集合
        if verbose:
            print(f"加载真实参考文献集合...")
        ground_truth = load_ground_truth_refs(ref_jsonl_path)
        if verbose:
            print(f"真实参考文献总数: {len(ground_truth)}")
        
        # 加载检索结果
        if verbose:
            print(f"\n处理方法: {method_name}")
        retrieved_list = load_retrieveref(json_path)
        if verbose:
            print(f"检索到的论文总数: {len(retrieved_list)}")
        
        # 计算各个 K 值的 Recall@K
        recall_values = []
        for k in k_values:
            recall = calculate_recall_at_k(retrieved_list, ground_truth, k)
            recall_values.append(recall)
            if verbose:
                print(f"  Recall@{k}: {recall:.6f}")
        
        # 构建结果
        result = {
            "name": method_name,
            "recallak": recall_values
        }
        
        # 追加结果到 eval.jsonl
        if verbose:
            print(f"\n将结果追加到 {eval_jsonl_path}")
        with open(eval_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        if verbose:
            print("完成！")
        
        return result
    
    except Exception as e:
        if verbose:
            print(f"处理 {method_name} 时出错: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='计算检索方法的 Recall@K 指标')
    parser.add_argument('--t', type=str, default='1', help='t目录编号，如 1, 2, ..., 20 (默认: 1)')
    parser.add_argument('--method', type=str, default=None, help='方法名称，如 a, a1, a2, f, f1, f2 等 (默认: 处理 a 和 f)')
    
    args = parser.parse_args()
    
    # 文件路径
    base_dir = os.path.join(os.path.dirname(__file__), '..', f't{args.t}')
    ref_jsonl_path = os.path.join(base_dir, 'ref.jsonl')
    eval_jsonl_path = os.path.join(base_dir, 'eval.jsonl')
    
    # K 值列表
    k_values = [20, 30, 100, 200, 500, 1000]
    
    # 如果指定了方法，只处理该方法；否则处理 a 和 f
    if args.method:
        methods = [(args.method, os.path.join(base_dir, f'{args.method}.json'))]
    else:
        # 默认处理 a 和 f
        methods = [
            ('a', os.path.join(base_dir, 'a.json')),
            ('f', os.path.join(base_dir, 'f.json'))
        ]
    
    results = []
    
    for method_name, json_path in methods:
        result = evaluate_method(method_name, json_path, ref_jsonl_path, eval_jsonl_path, k_values, verbose=True)
        if result:
            results.append(result)
    
    if not args.method:
        print(f"\n共处理 {len(results)} 个方法")

if __name__ == '__main__':
    main()

