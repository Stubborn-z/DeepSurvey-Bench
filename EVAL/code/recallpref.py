#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 a, x, f 方法的 recall, precision, F1 指标
通过 arxiv API 获取引用标题，与黄金标准进行比较
"""

import json
import os
import sys
import time
import argparse
import requests
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple, Optional

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1_DIR = os.path.join(BASE_DIR, "t1")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# 文件路径
A_JSON = os.path.join(T1_DIR, "a.json")
X_JSON = os.path.join(T1_DIR, "x.json")
F_JSON = os.path.join(T1_DIR, "f.json")
REF_JSONL = os.path.join(T1_DIR, "ref.jsonl")
EVAL_JSONL = os.path.join(T1_DIR, "eval.jsonl")


def string_similarity(s1: str, s2: str) -> float:
    """计算两个字符串的相似度"""
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    return SequenceMatcher(None, s1, s2).ratio()


def is_similar(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """判断两个字符串是否相似（基于阈值）"""
    return string_similarity(s1, s2) >= threshold


def clean_arxiv_id(arxivid: str) -> str:
    """清理 arxiv ID，移除版本号"""
    if not arxivid:
        return ""
    # 移除版本号，如 "2201.08239v3" -> "2201.08239"
    if 'v' in arxivid:
        arxivid = arxivid.rsplit('v', 1)[0]
    return arxivid.strip()


def get_arxiv_title(arxivid: str, arxiv_to_title: Dict[str, str] = None, 
                   cache: Dict[str, str] = None) -> str:
    """
    获取论文标题
    优先从 ref.jsonl 中查找，找不到再调用 arxiv API
    
    Args:
        arxivid: arxiv ID
        arxiv_to_title: 从 ref.jsonl 加载的 {arxivid: title} 字典
        cache: API 缓存
    
    Returns:
        论文标题
    """
    if arxiv_to_title is None:
        arxiv_to_title = {}
    if cache is None:
        cache = {}
    
    # 清理 arxiv ID
    clean_id = clean_arxiv_id(arxivid)
    if not clean_id:
        return ""
    
    # 首先尝试从 ref.jsonl 中查找
    if clean_id in arxiv_to_title:
        return arxiv_to_title[clean_id]
    
    # 检查 API 缓存
    if clean_id in cache:
        return cache[clean_id]
    
    # 调用 arxiv API
    try:
        # arxiv API URL
        url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
        
        # 发送请求
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 解析 XML 响应
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        # 查找标题
        # arxiv API 返回的命名空间
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if entries:
            title_elem = entries[0].find('atom:title', ns)
            if title_elem is not None:
                title = title_elem.text.strip()
                # 移除换行符和多余空格
                title = ' '.join(title.split())
                cache[clean_id] = title
                return title
        
        # 如果没找到，缓存空字符串
        cache[clean_id] = ""
        return ""
        
    except Exception as e:
        print(f"获取 arxiv {clean_id} 标题失败: {e}")
        cache[clean_id] = ""
        return ""


def load_ref_jsonl(ref_jsonl_path: str) -> Dict[str, Dict[str, str]]:
    """
    从 REF.jsonl 加载引用数据（格式：id, arxiv, title, abstract）
    返回: {标号: {"arxivid": ..., "title": ..., "abstract": ...}} 的字典
    """
    ref_data = {}
    
    with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ref_id = data.get("id", "")
                arxivid = data.get("arxiv", "")  # 注意：REF.jsonl 中使用 "arxiv" 而不是 "arxivid"
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                
                if ref_id:
                    ref_data[ref_id] = {
                        "arxivid": arxivid,
                        "title": title if title else "",  # 空字符串表示缺失
                        "abstract": abstract if abstract else ""
                    }
            except json.JSONDecodeError as e:
                print(f"解析 REF.jsonl 行失败: {e}")
                continue
    
    return ref_data


def load_gold_references(ref_jsonl_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    从 ref.jsonl 加载黄金标准的引用
    返回: (黄金标准标题列表, {arxivid: title} 的字典用于查找)
    """
    gold_titles = []
    arxiv_to_title = {}
    
    with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                arxivid = data.get("arxivid", "")
                title = data.get("title", "")
                
                if title:
                    # 添加到黄金标准标题列表
                    gold_titles.append(title)
                    
                    # 如果有 arxivid，添加到查找字典
                    if arxivid:
                        clean_id = clean_arxiv_id(arxivid)
                        if clean_id:
                            arxiv_to_title[clean_id] = title
            except json.JSONDecodeError as e:
                print(f"解析 ref.jsonl 行失败: {e}")
                continue
    
    return gold_titles, arxiv_to_title


def load_generated_references(json_path: str) -> Dict[str, str]:
    """
    从 a.json, x.json, f.json 加载生成的引用
    返回: {标号: arxivid} 的字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    references = data.get("reference", {})
    return references


def calculate_metrics(
    generated_titles: List[str],
    gold_titles: List[str],
    threshold: float = 0.8,
    return_details: bool = False
) -> Tuple[float, float, float]:
    """
    计算 recall, precision, F1
    
    Args:
        generated_titles: 生成的引用标题列表
        gold_titles: 黄金标准的引用标题列表
        threshold: 相似度阈值
    
    Returns:
        (recall, precision, f1)
    """
    if not gold_titles:
        return 0.0, 0.0, 0.0
    
    if not generated_titles:
        return 0.0, 0.0, 0.0
    
    # 使用集合记录已匹配的索引，避免重复匹配
    matched_gold_indices = set()  # 已匹配的黄金标准标题索引
    matched_gen_indices = set()   # 已匹配的生成标题索引
    
    # 对每个黄金标准标题，检查是否有生成的标题与之匹配
    for gold_idx, gold_title in enumerate(gold_titles):
        if gold_idx in matched_gold_indices:
            continue  # 已匹配，跳过
        
        for gen_idx, gen_title in enumerate(generated_titles):
            if gen_idx in matched_gen_indices:
                continue  # 已匹配，跳过
            
            if is_similar(gold_title, gen_title, threshold):
                matched_gold_indices.add(gold_idx)
                matched_gen_indices.add(gen_idx)
                break  # 每个黄金标题只匹配一次
    
    # 计算指标
    # Recall: 匹配的黄金标准标题数 / 总黄金标准标题数
    matched_gold_count = len(matched_gold_indices)
    total_gold = len(gold_titles)
    recall = matched_gold_count / total_gold if total_gold > 0 else 0.0
    
    # Precision: 匹配的生成标题数 / 总生成标题数
    matched_count = len(matched_gen_indices)
    total_generated = len(generated_titles)
    
    # 检查分母是否为0
    if total_generated == 0:
        precision = 0.0
        print(f"警告: 生成标题总数为0，precision设为0.0")
    else:
        precision = matched_count / total_generated
        # 确保不超过 1.0（理论上不应该超过）
        if precision > 1.0:
            print(f"错误: Precision ({precision:.6f}) 大于 1.0！")
            print(f"  匹配数: {matched_count}, 生成标题总数: {total_generated}")
            print(f"  matched_gen_indices: {sorted(matched_gen_indices)}")
            precision = 1.0  # 强制限制为1.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    if return_details:
        return recall, precision, f1, matched_count, total_generated, matched_gold_count, total_gold
    
    return recall, precision, f1


def process_method(method_name: str, json_path: str, gold_titles: List[str],
                   arxiv_to_title: Dict[str, str], arxiv_cache: Dict[str, str],
                   ref_jsonl_data: Optional[Dict[str, Dict[str, str]]] = None) -> Tuple[float, float, float]:
    """
    处理一个方法（a, x, 或 f）
    
    Args:
        method_name: 方法名称 ('a', 'x', 或 'f')
        json_path: JSON 文件路径
        gold_titles: 黄金标准标题列表
        arxiv_to_title: 从 ref.jsonl 加载的 {arxivid: title} 字典
        arxiv_cache: arxiv API 缓存
    
    Returns:
        (recall, precision, f1)
    """
    print(f"\n处理方法: {method_name}")
    
    # 加载生成的引用
    generated_refs = load_generated_references(json_path)
    print(f"生成的引用数量: {len(generated_refs)}")
    
    # 获取生成的引用标题
    # 只保留成功获取标题的引文，无法获取标题的引文将被舍弃
    generated_titles = []
    skipped_count = 0  # 统计被舍弃的引文数量
    api_call_count = 0
    
    for ref_num, arxivid in generated_refs.items():
        # 优先使用 REF.jsonl 数据
        if ref_jsonl_data and ref_num in ref_jsonl_data:
            ref_info = ref_jsonl_data[ref_num]
            title = ref_info.get("title", "")
            if title:
                # 成功获取标题，加入计算列表
                generated_titles.append(title)
                print(f"  引用 {ref_num}: {arxivid} -> {title[:60]}... (来自REF.jsonl)")
            else:
                # 标题为空，舍弃该引文
                skipped_count += 1
                print(f"  引用 {ref_num}: {arxivid} -> (REF.jsonl中标题为空) -> 舍弃，不计入统计")
            continue
        
        # 如果没有 REF.jsonl，使用原有逻辑
        if not arxivid:
            skipped_count += 1
            print(f"  引用 {ref_num}: (空 arxivid) -> 舍弃")
            continue
        
        clean_id = clean_arxiv_id(arxivid)
        if not clean_id:
            skipped_count += 1
            print(f"  引用 {ref_num}: {arxivid} -> (无效 arxivid) -> 舍弃")
            continue
        
        # 检查是否需要调用 API
        need_api = clean_id not in arxiv_to_title and clean_id not in arxiv_cache
        
        title = get_arxiv_title(arxivid, arxiv_to_title, arxiv_cache)
        if title:
            # 成功获取标题，加入计算列表
            generated_titles.append(title)
            print(f"  引用 {ref_num}: {arxivid} -> {title[:60]}...")
        else:
            # 无法获取标题，舍弃该引文
            skipped_count += 1
            print(f"  引用 {ref_num}: {arxivid} -> (未找到标题) -> 舍弃，不计入统计")
        
        # 如果调用了 API，添加延迟避免 API 限制
        if need_api:
            api_call_count += 1
            if api_call_count % 3 == 0:
                time.sleep(1)
    
    print(f"\n引文统计:")
    print(f"  原始引文数量: {len(generated_refs)}")
    print(f"  成功获取标题: {len(generated_titles)}")
    print(f"  舍弃的引文: {skipped_count} (不计入计算)")
    print(f"黄金标准引用数量: {len(gold_titles)}")
    
    # 检查是否有重复标题（调试信息）
    from collections import Counter
    title_counts = Counter(generated_titles)
    unique_titles_count = len(title_counts)
    if unique_titles_count < len(generated_titles):
        print(f"  警告: 生成的标题中有 {len(generated_titles) - unique_titles_count} 个重复")
    
    # 计算指标（获取详细信息）
    result = calculate_metrics(generated_titles, gold_titles, return_details=True)
    recall, precision, f1, matched_count, total_gen, matched_gold_count, total_gold = result
    
    print(f"\n计算结果:")
    print(f"  生成标题: {total_gen} 个（唯一: {unique_titles_count} 个）")
    print(f"  黄金标准标题: {total_gold} 个")
    print(f"  匹配的生成标题数: {matched_count} / {total_gen}")
    print(f"  匹配的黄金标准标题数: {matched_gold_count} / {total_gold}")
    print(f"  Recall: {recall:.6f} = {matched_gold_count} / {total_gold}")
    print(f"  Precision: {precision:.6f} = {matched_count} / {total_gen}")
    print(f"  F1: {f1:.6f}")
    
    # 验证 precision 是否合理
    if precision == 1.0:
        if len(generated_titles) > 0:
            print(f"  注意: Precision = 1.0，所有 {len(generated_titles)} 个生成标题都被匹配")
        else:
            print(f"  错误: Precision = 1.0 但生成标题数为0！这不应该发生！")
    elif precision > 1.0:
        print(f"  错误: Precision ({precision:.6f}) 大于 1.0！")
    
    return recall, precision, f1


def append_to_eval_jsonl(method_name: str, recall: float, precision: float, f1: float, 
                        eval_jsonl_path: str):
    """追加结果到 eval.jsonl"""
    result = {
        "name": method_name,
        "recallpref": [recall, precision, f1]
    }
    
    with open(eval_jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"结果已追加到 {eval_jsonl_path}")


def is_already_evaluated(method_name: str, eval_jsonl_path: str) -> bool:
    """检查某个方法是否已经被评估过"""
    if not os.path.exists(eval_jsonl_path):
        return False
    
    with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("name") == method_name and "recallpref" in data:
                    return True
            except json.JSONDecodeError:
                continue
    
    return False


def evaluate_single_instance(base_dir: str, t_num: str, method_type: str, method_suffix: str = ""):
    """
    评估单个实例
    
    Args:
        base_dir: EVAL 目录路径
        t_num: t目录编号（如 "1", "2"）
        method_type: 方法类型（"a", "x", "f"）
        method_suffix: 方法后缀（如 "1", "2"，空字符串表示无后缀）
    """
    t_dir = os.path.join(base_dir, f"t{t_num}")
    method_name = f"{method_type}{method_suffix}" if method_suffix else method_type
    json_path = os.path.join(t_dir, f"{method_name}.json")
    ref_jsonl_path = os.path.join(t_dir, f"{method_name}REF.jsonl")
    gold_ref_jsonl_path = os.path.join(t_dir, "ref.jsonl")
    eval_jsonl_path = os.path.join(t_dir, "eval.jsonl")
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"  文件不存在: {json_path}")
        return False
    
    if not os.path.exists(gold_ref_jsonl_path):
        print(f"  黄金标准文件不存在: {gold_ref_jsonl_path}")
        return False
    
    # 检查是否已评估
    if is_already_evaluated(method_name, eval_jsonl_path):
        print(f"  实例 {method_name} 已评估，跳过")
        return True
    
    print(f"\n处理实例: {method_name}")
    print(f"  输入文件: {json_path}")
    
    # 加载黄金标准引用
    gold_titles, arxiv_to_title = load_gold_references(gold_ref_jsonl_path)
    print(f"  加载了 {len(gold_titles)} 个黄金标准引用")
    
    # 加载 REF.jsonl（如果存在）
    ref_jsonl_data = None
    if os.path.exists(ref_jsonl_path):
        print(f"  发现 REF.jsonl: {ref_jsonl_path}")
        ref_jsonl_data = load_ref_jsonl(ref_jsonl_path)
        print(f"  加载了 {len(ref_jsonl_data)} 条 REF.jsonl 数据")
    
    # arxiv API 缓存
    arxiv_cache = {}
    
    # 处理
    try:
        recall, precision, f1 = process_method(
            method_name, json_path, gold_titles, arxiv_to_title, arxiv_cache, ref_jsonl_data
        )
        
        print(f"\n评估结果:")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1: {f1:.4f}")
        
        # 追加结果
        append_to_eval_jsonl(method_name, recall, precision, f1, eval_jsonl_path)
        return True
        
    except Exception as e:
        print(f"处理 {method_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算 recall, precision, F1 指标')
    parser.add_argument('--t_num', type=str, help='t目录编号（如 "1", "2"），如果提供则评估指定实例')
    parser.add_argument('--method', type=str, help='方法类型（"a", "x", "f"）')
    parser.add_argument('--suffix', type=str, default='', help='方法后缀（如 "1", "2"，默认空字符串）')
    
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 如果提供了参数，评估单个实例
    if args.t_num and args.method:
        evaluate_single_instance(BASE_DIR, args.t_num, args.method, args.suffix)
        return
    
    # 否则使用原有逻辑（评估 t1 下的 a.json, x.json, f.json）
    print("=" * 60)
    print("计算 recall, precision, F1 指标")
    print("=" * 60)
    
    T1_DIR = os.path.join(BASE_DIR, "t1")
    A_JSON = os.path.join(T1_DIR, "a.json")
    X_JSON = os.path.join(T1_DIR, "x.json")
    F_JSON = os.path.join(T1_DIR, "f.json")
    REF_JSONL = os.path.join(T1_DIR, "ref.jsonl")
    EVAL_JSONL = os.path.join(T1_DIR, "eval.jsonl")
    
    # 加载黄金标准引用
    print("\n加载黄金标准引用...")
    gold_titles, arxiv_to_title = load_gold_references(REF_JSONL)
    print(f"加载了 {len(gold_titles)} 个黄金标准引用")
    print(f"其中 {len(arxiv_to_title)} 个有 arxivid")
    
    # arxiv API 缓存
    arxiv_cache = {}
    
    # 处理方法
    methods = [
        ("a", A_JSON),
        ("x", X_JSON),
        ("f", F_JSON),
    ]
    
    for method_name, json_path in methods:
        if not os.path.exists(json_path):
            print(f"警告: 文件不存在 {json_path}")
            continue
        
        try:
            recall, precision, f1 = process_method(
                method_name, json_path, gold_titles, arxiv_to_title, arxiv_cache
            )
            
            # 追加结果
            append_to_eval_jsonl(method_name, recall, precision, f1, EVAL_JSONL)
            
        except Exception as e:
            print(f"处理 {method_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

