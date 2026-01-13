#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 EVAL/t<a>/ 下的 a<b>.json, f<b>.json, x<b>.json 文件中读取 reference 键，
通过 arxivid 调用 arxiv API 获取标题和摘要，
生成对应的 a<b>REF.jsonl, f<b>REF.jsonl, x<b>REF.jsonl 文件。
"""

import json
import os
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Arxiv API 配置
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# 禁用代理（如果需要）
os.environ.setdefault('HTTP_PROXY', '')
os.environ.setdefault('HTTPS_PROXY', '')
os.environ.setdefault('http_proxy', '')
os.environ.setdefault('https_proxy', '')


def clean_arxiv_id(arxivid: str) -> str:
    """清理 arxiv ID，移除版本号"""
    if not arxivid:
        return ""
    # 移除版本号，如 "2201.08239v3" -> "2201.08239"
    if 'v' in arxivid:
        arxivid = arxivid.rsplit('v', 1)[0]
    return arxivid.strip()


def fetch_arxiv_info(arxivid: str) -> Tuple[str, str]:
    """
    通过 arxiv ID 获取论文标题和摘要
    
    Args:
        arxivid: arxiv ID (可能带版本号)
    
    Returns:
        (title, abstract) 元组，如果查不到则返回 ("", "")
    """
    if not arxivid:
        return ("", "")
    
    clean_id = clean_arxiv_id(arxivid)
    if not clean_id:
        return ("", "")
    
    try:
        # 调用 arxiv API
        url = f"{ARXIV_API_URL}?id_list={clean_id}"
        proxies = {'http': None, 'https': None}
        response = requests.get(url, timeout=30, proxies=proxies)
        response.raise_for_status()
        
        # 解析 XML 响应
        root = ET.fromstring(response.content)
        
        # arxiv API 返回的命名空间
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return ("", "")
        
        entry = entries[0]
        
        # 获取标题
        title = ""
        title_elem = entry.find('atom:title', ns)
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            # 移除换行符和多余空格
            title = ' '.join(title.split())
        
        # 获取摘要
        abstract = ""
        summary_elem = entry.find('atom:summary', ns)
        if summary_elem is not None and summary_elem.text:
            abstract = summary_elem.text.strip()
            # 移除换行符和多余空格
            abstract = ' '.join(abstract.split())
        
        return (title, abstract)
        
    except Exception as e:
        print(f"    警告: 获取 arxiv {clean_id} 信息失败: {e}")
        return ("", "")


def process_json_file(json_path: Path, ref_jsonl_path: Path) -> Tuple[bool, bool]:
    """
    处理单个 JSON 文件，生成对应的 REF.jsonl 文件
    
    Args:
        json_path: 输入的 JSON 文件路径
        ref_jsonl_path: 输出的 REF.jsonl 文件路径
    
    Returns:
        (是否成功, 是否跳过) 元组
    """
    # 检查输出文件是否已存在
    if ref_jsonl_path.exists():
        print(f"  跳过: {ref_jsonl_path.name} 已存在")
        return (True, True)  # 成功（已存在），跳过
    
    # 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  错误: 读取 {json_path.name} 失败: {e}")
        return (False, False)
    
    # 获取 reference 键
    reference = data.get("reference", {})
    if not reference:
        print(f"  警告: {json_path.name} 中没有 reference 键或为空")
        # 仍然创建空文件
        try:
            with open(ref_jsonl_path, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
            print(f"  已创建空的 {ref_jsonl_path.name}")
            return (True, False)  # 成功处理（创建空文件），不是跳过
        except Exception as e:
            print(f"  错误: 创建 {ref_jsonl_path.name} 失败: {e}")
            return (False, False)
    
    print(f"  找到 {len(reference)} 个文献")
    
    # 打开输出文件
    try:
        ref_file = open(ref_jsonl_path, 'w', encoding='utf-8')
    except Exception as e:
        print(f"  错误: 打开 {ref_jsonl_path.name} 失败: {e}")
        return (False, False)
    
    # 处理每个文献
    success_count = 0
    for idx, (ref_id, arxivid) in enumerate(reference.items(), 1):
        print(f"    处理 [{idx}/{len(reference)}]: id={ref_id}, arxiv={arxivid}")
        
        # 获取标题和摘要
        title, abstract = fetch_arxiv_info(arxivid)
        
        # 构建结果
        result = {
            "id": ref_id,
            "arxiv": arxivid,
            "title": title,
            "abstract": abstract
        }
        
        # 写入文件
        try:
            ref_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            ref_file.flush()  # 确保立即写入磁盘
            success_count += 1
        except Exception as e:
            print(f"      写入错误: {e}")
        
        # 避免请求过快（arxiv API 有速率限制）
        if idx < len(reference):
            time.sleep(0.3)
    
    # 关闭文件
    ref_file.close()
    print(f"  完成: 已保存 {success_count}/{len(reference)} 条结果到 {ref_jsonl_path.name}")
    return (True, False)  # 成功处理，不是跳过


def main():
    """主函数"""
    # 获取 EVAL 目录路径（脚本在 EVAL/code/ 下）
    script_dir = Path(__file__).parent
    eval_dir = script_dir.parent
    print(f"EVAL 目录: {eval_dir}\n")
    
    # 文件前缀列表: a, f, x
    prefixes = ['a', 'f', 'x']
    
    # 文件后缀列表: 空字符串, '1', '2', ..., '9'
    suffixes = [''] + [str(i) for i in range(1, 10)]
    
    # 遍历 t1 到 t20
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    for t_num in range(1, 21):
        t_dir = eval_dir / f"t{t_num}"
        
        if not t_dir.exists() or not t_dir.is_dir():
            print(f"跳过: t{t_num} 目录不存在")
            continue
        
        print(f"\n处理目录: t{t_num}")
        
        # 遍历所有可能的文件组合
        for prefix in prefixes:
            for suffix in suffixes:
                # 构建文件名，如 a.json, a2.json, f1.json 等
                json_filename = f"{prefix}{suffix}.json"
                json_path = t_dir / json_filename
                
                # 构建输出文件名，如 aREF.jsonl, a2REF.jsonl, f1REF.jsonl 等
                ref_jsonl_filename = f"{prefix}{suffix}REF.jsonl"
                ref_jsonl_path = t_dir / ref_jsonl_filename
                
                # 检查输入文件是否存在
                if not json_path.exists():
                    continue  # 文件不存在，跳过
                
                print(f"  处理: {json_filename} -> {ref_jsonl_filename}")
                
                # 处理文件
                success, skipped = process_json_file(json_path, ref_jsonl_path)
                
                if success:
                    if skipped:
                        total_skipped += 1
                    else:
                        total_processed += 1
                else:
                    total_failed += 1
                    print(f"  ✗ 处理失败: {json_filename}")
        
        print(f"  目录 t{t_num} 处理完成")
    
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"  成功处理: {total_processed} 个文件")
    print(f"  跳过（已存在）: {total_skipped} 个文件")
    print(f"  处理失败: {total_failed} 个文件")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

