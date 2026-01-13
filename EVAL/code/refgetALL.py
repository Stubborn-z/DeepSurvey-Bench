#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 t1-t30 文件夹中读取 paper.json 的 all_cites_title，
查询 Elasticsearch 和 arxiv API，生成 ref.jsonl
"""

import json
import os
import requests
import time
from pathlib import Path

# Elasticsearch 配置
ES_URL = "http://10.1.114.121:18080/papers_search/_search"
ES_HEADERS = {"Content-Type": "application/json"}

# Arxiv API 配置
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# 禁用代理（在 proxyoff 环境下）
os.environ.setdefault('HTTP_PROXY', '')
os.environ.setdefault('HTTPS_PROXY', '')
os.environ.setdefault('http_proxy', '')
os.environ.setdefault('https_proxy', '')


def query_elasticsearch(title):
    """查询 Elasticsearch，返回第一个结果的 abstract"""
    try:
        query = {
            "query": {
                "match": {
                    "title": title
                }
            }
        }
        # 确保不使用代理
        proxies = {'http': None, 'https': None}
        response = requests.get(ES_URL, headers=ES_HEADERS, json=query, timeout=30, proxies=proxies)
        response.raise_for_status()
        data = response.json()
        
        if data.get("hits") and data["hits"].get("hits") and len(data["hits"]["hits"]) > 0:
            first_hit = data["hits"]["hits"][0]
            source = first_hit.get("_source", {})
            abstract = source.get("abstract", "")
            return abstract
        return ""
    except Exception as e:
        print(f"  Elasticsearch 查询错误: {e}")
        return ""


def query_arxiv(title):
    """查询 arxiv API，返回最匹配文章的 arxiv id"""
    try:
        import xml.etree.ElementTree as ET
        
        # 使用标题搜索 arxiv，先尝试精确匹配
        params = {
            "search_query": f'ti:"{title}"',
            "start": 0,
            "max_results": 1
        }
        # 确保不使用代理
        proxies = {'http': None, 'https': None}
        response = requests.get(ARXIV_API_URL, params=params, timeout=30, proxies=proxies)
        response.raise_for_status()
        
        # 解析 XML 响应
        root = ET.fromstring(response.content)
        
        # 定义命名空间
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        
        if entries:
            # 获取第一个 entry 的 id
            first_entry = entries[0]
            arxiv_id_elem = first_entry.find('atom:id', namespace)
            if arxiv_id_elem is not None and arxiv_id_elem.text:
                arxiv_url = arxiv_id_elem.text
                # 提取 arxiv id，格式如 "http://arxiv.org/abs/1502.03951v3" -> "1502.03951v3"
                if '/abs/' in arxiv_url:
                    arxiv_id = arxiv_url.split('/abs/')[-1]
                    return arxiv_id
        
        # 如果没有找到，尝试更宽松的搜索（使用 all 字段）
        params = {
            "search_query": f'all:"{title}"',
            "start": 0,
            "max_results": 1
        }
        response = requests.get(ARXIV_API_URL, params=params, timeout=30, proxies=proxies)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        entries = root.findall('atom:entry', namespace)
        
        if entries:
            first_entry = entries[0]
            arxiv_id_elem = first_entry.find('atom:id', namespace)
            if arxiv_id_elem is not None and arxiv_id_elem.text:
                arxiv_url = arxiv_id_elem.text
                if '/abs/' in arxiv_url:
                    arxiv_id = arxiv_url.split('/abs/')[-1]
                    return arxiv_id
        
        return ""
    except Exception as e:
        print(f"  Arxiv 查询错误: {e}")
        return ""


def process_folder(folder_path):
    """处理单个文件夹，每处理完一个标题立即保存"""
    folder_name = os.path.basename(folder_path)
    print(f"\n处理文件夹: {folder_name}")
    
    paper_json_path = os.path.join(folder_path, "paper.json")
    ref_jsonl_path = os.path.join(folder_path, "ref.jsonl")
    
    # 检查 paper.json 是否存在
    if not os.path.exists(paper_json_path):
        print(f"  跳过: {paper_json_path} 不存在")
        return False
    
    # 读取 paper.json
    try:
        with open(paper_json_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
    except Exception as e:
        print(f"  读取 paper.json 错误: {e}")
        return False
    
    # 获取 all_cites_title（在 meta_info 键下）
    meta_info = paper_data.get("meta_info", {})
    all_cites_title = meta_info.get("all_cites_title", [])
    if not all_cites_title:
        print(f"  跳过: all_cites_title 为空")
        return False
    
    print(f"  找到 {len(all_cites_title)} 个文献标题")
    
    # 打开文件准备写入（覆盖模式，因为这是新开始处理）
    try:
        ref_file = open(ref_jsonl_path, 'w', encoding='utf-8')
    except Exception as e:
        print(f"  打开 ref.jsonl 文件错误: {e}")
        return False
    
    # 处理每个标题，每处理完一个立即写入
    success_count = 0
    for idx, title in enumerate(all_cites_title, 1):
        print(f"  处理 [{idx}/{len(all_cites_title)}]: {title[:60]}...")
        
        # 查询 Elasticsearch
        abstract = query_elasticsearch(title)
        time.sleep(0.1)  # 避免请求过快
        
        # 查询 arxiv
        arxivid = query_arxiv(title)
        time.sleep(0.3)  # arxiv API 有速率限制
        
        # 构建结果
        result = {
            "title": title,
            "abstract": abstract,
            "arxivid": arxivid
        }
        
        # 立即写入文件
        try:
            ref_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            ref_file.flush()  # 确保立即写入磁盘
            success_count += 1
        except Exception as e:
            print(f"    写入错误: {e}")
        
        print(f"    -> abstract: {'有' if abstract else '无'}, arxivid: {arxivid if arxivid else '无'}")
    
    # 关闭文件
    ref_file.close()
    print(f"  完成: 已保存 {success_count}/{len(all_cites_title)} 条结果到 {ref_jsonl_path}")
    return True


def main():
    """主函数，支持断点续传"""
    # 获取 EVAL 目录路径
    eval_dir = Path(__file__).parent.parent
    print(f"EVAL 目录: {eval_dir}")
    
    # 找出所有需要处理的文件夹（没有 ref.jsonl 的文件夹）
    folders_to_process = []
    for i in range(1, 31):
        folder_name = f"t{i}"
        folder_path = eval_dir / folder_name
        ref_jsonl_path = folder_path / "ref.jsonl"
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            if not os.path.exists(ref_jsonl_path):
                folders_to_process.append((i, folder_name, folder_path))
            else:
                print(f"跳过: {folder_name} 已有 ref.jsonl，跳过")
        else:
            print(f"跳过: {folder_name} 文件夹不存在")
    
    if not folders_to_process:
        print("\n所有文件夹都已处理完成！")
        return
    
    print(f"\n找到 {len(folders_to_process)} 个需要处理的文件夹")
    print(f"将从 {folders_to_process[0][1]} 开始处理...\n")
    
    # 按顺序处理每个文件夹
    for i, folder_name, folder_path in folders_to_process:
        success = process_folder(folder_path)
        if success:
            print(f"✓ {folder_name} 处理完成并已保存")
        else:
            print(f"✗ {folder_name} 处理失败")
    
    print("\n所有处理完成！")


if __name__ == "__main__":
    main()

