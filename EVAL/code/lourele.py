#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算引用评估指标：IoU (Insertion over Union)、Relevancesemantic、RelevanceLLM
"""

import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

# 配置路径
BASE_DIR = Path(__file__).parent.parent
AKEY_FILE = BASE_DIR / "code" / "akey.txt"

# LLM配置
BASE_URL = "https://api.ai-gaochao.cn/v1"
MODEL = "gpt-4o-mini"


def read_api_key():
    """从akey.txt读取API key，如果有多行则取最后一行"""
    try:
        with open(AKEY_FILE, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if lines:
            # 取最后一行，并移除可能的注释部分（空格后的内容）
            last_line = lines[-1].split()[0] if lines[-1].split() else lines[-1]
            return last_line
        return None
    except Exception as e:
        print(f"警告: 无法读取API key: {e}")
        return None


def extract_arxiv_id(arxiv_id_with_version):
    """从带版本的arxiv ID中提取ID（去掉版本号）"""
    return arxiv_id_with_version.split('v')[0] if 'v' in arxiv_id_with_version else arxiv_id_with_version


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_arxiv_abstract(arxiv_id):
    """从arXiv API获取论文摘要"""
    arxiv_id_clean = extract_arxiv_id(arxiv_id)
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id_clean}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_content = response.read().decode('utf-8')
            
            # 使用XML解析
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_content)
                ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
                summary_elem = root.find('.//arxiv:summary', ns)
                if summary_elem is not None and summary_elem.text:
                    abstract = summary_elem.text.strip()
                    abstract = ' '.join(abstract.split())
                    return abstract
            except:
                # 正则表达式备选
                summary_match = re.search(r'<summary>(.*?)</summary>', xml_content, re.DOTALL)
                if summary_match:
                    abstract = summary_match.group(1).strip()
                    abstract = abstract.replace('\n', ' ').replace('  ', ' ')
                    return abstract
            
            return None
    except Exception as e:
        print(f"错误: 获取arXiv摘要失败 {arxiv_id_clean}: {e}")
        return None


def load_ref_jsonl(ref_jsonl_path):
    """从REF.jsonl文件加载引用数据"""
    ref_data = {}  # {id: abstract}
    if ref_jsonl_path.exists():
        with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        ref_id = str(item.get('id', ''))
                        abstract = item.get('abstract', '')
                        # 如果abstract为空字符串或缺失，使用空字符串
                        if abstract == "" or abstract is None:
                            abstract = ""
                        ref_data[ref_id] = abstract
                    except json.JSONDecodeError:
                        continue
    return ref_data


def build_ref_cache(references, ref_jsonl_path=None):
    """构建引用ID到摘要的缓存"""
    ref_cache = {}
    
    # 如果存在REF.jsonl文件，直接使用
    if ref_jsonl_path and ref_jsonl_path.exists():
        print(f"使用REF.jsonl文件: {ref_jsonl_path}")
        ref_data = load_ref_jsonl(ref_jsonl_path)
        # 将references中的arxiv_id映射转换为id到abstract的映射
        for ref_key, arxiv_id in references.items():
            # 在ref_data中查找对应的id
            if ref_key in ref_data:
                ref_cache[arxiv_id] = ref_data[ref_key]
            else:
                ref_cache[arxiv_id] = ""
        return ref_cache
    
    # 否则从arXiv API获取
    unique_arxiv_ids = list(set(references.values()))
    
    print(f"需要获取 {len(unique_arxiv_ids)} 个唯一arxiv ID的摘要...")
    for arxiv_id in tqdm(unique_arxiv_ids, desc="获取arXiv摘要"):
        if arxiv_id not in ref_cache:
            abstract = fetch_arxiv_abstract(arxiv_id)
            ref_cache[arxiv_id] = abstract if abstract else ""
            time.sleep(0.5)  # 避免API频率限制
    
    return ref_cache


def extract_citations_and_claims(survey_text):
    """
    从survey文本中提取引用和对应的claim
    
    返回: [(claim, [ref_nums]), ...]
    """
    claims_with_refs = []
    
    # 更精确的提取方法：找到所有包含引用的句子
    # 先按句号、问号、感叹号分割成句子
    sentences = re.split(r'([.!?]\s+)', survey_text)
    
    # 合并分隔符回句子
    processed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        processed_sentences.append(sentence)
    
    # 处理最后一个句子（如果没有分隔符）
    if len(sentences) % 2 == 1:
        processed_sentences.append(sentences[-1])
    
    # 查找每个句子中的引用
    for sentence in processed_sentences:
        # 查找所有引用标记 [数字,数字] 或 [数字]
        ref_matches = re.finditer(r'\[([0-9,]+)\]', sentence)
        
        # 收集这个句子中的所有引用编号
        ref_nums_in_sentence = []
        for ref_match in ref_matches:
            ref_nums_str = ref_match.group(1)
            ref_nums = [int(x.strip()) for x in ref_nums_str.split(',')]
            ref_nums_in_sentence.extend(ref_nums)
        
        # 如果句子中有引用，提取claim（移除引用标记）
        if ref_nums_in_sentence:
            # 移除所有引用标记
            claim = re.sub(r'\[([0-9,]+)\]', '', sentence).strip()
            
            # 跳过太短的claim
            if len(claim) >= 10:
                claims_with_refs.append((claim, ref_nums_in_sentence))
    
    return claims_with_refs


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def nli_relevance_llm(client, claim, source_text):
    """使用LLM进行NLI判断claim与source的相关性（RelevanceLLM）"""
    prompt = """---
Claim:
{claim}
---
Source: 
{source}
---
Claim:
{claim}
---
Is the Claim faithful to the Source? 
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.
Only reply with 'Yes' or 'No':""".format(
        claim=claim, source=source_text
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        response = completion.choices[0].message.content.strip().lower()
        
        # 识别回复中是否包含"yes"或"no"
        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            # 如果没有明确的yes/no，默认返回False
            print(f"警告: LLM回复未包含yes/no: {response[:100]}")
            return False
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return False


def compute_semantic_similarity(text1, text2):
    """
    计算语义相似度（Relevancesemantic）
    使用Jaccard相似度作为语义相似度的近似
    实际应用中可以使用sentence-transformers或其他embedding模型计算余弦相似度
    """
    if not text1 or not text2:
        return 0.0
    
    # 简单的token重叠度计算（Jaccard相似度）
    # 转换为小写并分词
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    # Jaccard相似度
    jaccard = intersection / union if union > 0 else 0.0
    
    return jaccard


def calculate_iou(claims_with_refs, ref_cache, references, client):
    """
    计算IoU (Insertion over Union)
    IoU = 支持的claim-source对数量 / 总claim-source对数量
    这里理解为：所有引用对中，被LLM判断为相关的比例
    """
    total_pairs = 0
    supported_pairs = 0
    
    for claim, ref_nums in tqdm(claims_with_refs, desc="计算IoU"):
        for ref_num in ref_nums:
            total_pairs += 1
            
            # 获取对应的arxiv ID和摘要
            ref_key = str(ref_num)
            if ref_key in references:
                arxiv_id = references[ref_key]
                source_text = ref_cache.get(arxiv_id, "")
                
                if source_text:
                    # 使用LLM判断相关性
                    is_relevant = nli_relevance_llm(client, claim, source_text)
                    if is_relevant:
                        supported_pairs += 1
    
    iou = supported_pairs / total_pairs if total_pairs > 0 else 0.0
    return iou


def calculate_relevance_semantic(claims_with_refs, ref_cache, references, threshold=0.1):
    """
    计算Relevancesemantic
    使用语义相似度判断相关性
    """
    total_pairs = 0
    relevant_pairs = 0
    
    for claim, ref_nums in tqdm(claims_with_refs, desc="计算Relevancesemantic"):
        for ref_num in ref_nums:
            total_pairs += 1
            
            ref_key = str(ref_num)
            if ref_key in references:
                arxiv_id = references[ref_key]
                source_text = ref_cache.get(arxiv_id, "")
                
                if source_text:
                    similarity = compute_semantic_similarity(claim, source_text)
                    if similarity >= threshold:
                        relevant_pairs += 1
    
    relevance_semantic = relevant_pairs / total_pairs if total_pairs > 0 else 0.0
    return relevance_semantic


def calculate_relevance_llm(claims_with_refs, ref_cache, references, client):
    """
    计算RelevanceLLM
    使用LLM进行NLI判断相关性
    """
    total_pairs = 0
    relevant_pairs = 0
    
    for claim, ref_nums in tqdm(claims_with_refs, desc="计算RelevanceLLM"):
        for ref_num in ref_nums:
            total_pairs += 1
            
            ref_key = str(ref_num)
            if ref_key in references:
                arxiv_id = references[ref_key]
                source_text = ref_cache.get(arxiv_id, "")
                
                if source_text:
                    is_relevant = nli_relevance_llm(client, claim, source_text)
                    if is_relevant:
                        relevant_pairs += 1
    
    relevance_llm = relevant_pairs / total_pairs if total_pairs > 0 else 0.0
    return relevance_llm


def evaluate_method(method_name, client, t_dir=None, method_suffix=""):
    """
    评估单个方法
    
    Args:
        method_name: 方法名，如 'a', 'x', 'f'
        client: OpenAI客户端
        t_dir: 目标目录（如 t1, t2），如果为None则使用t1（兼容原有调用）
        method_suffix: 方法后缀（如 '1', '2'），空字符串表示无后缀
    """
    import time
    start_time = time.time()  # 记录开始时间
    
    if t_dir is None:
        t_dir = BASE_DIR / "t1"
    else:
        t_dir = BASE_DIR / t_dir
    
    # 构建文件名
    if method_suffix:
        json_file = t_dir / f"{method_name}{method_suffix}.json"
        ref_jsonl_file = t_dir / f"{method_name}{method_suffix}REF.jsonl"
        method_full_name = f"{method_name}{method_suffix}"
    else:
        json_file = t_dir / f"{method_name}.json"
        ref_jsonl_file = t_dir / f"{method_name}REF.jsonl"
        method_full_name = method_name
    
    output_file = t_dir / "eval.jsonl"
    
    if not json_file.exists():
        print(f"警告: 文件 {json_file} 不存在，跳过")
        return None
    
    print(f"\n{'='*60}")
    print(f"处理方法: {method_full_name} (文件: {json_file})")
    print(f"{'='*60}")
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    survey = data.get('survey', '')
    references = data.get('reference', {})
    
    if not survey:
        print(f"警告: {method_full_name}.json 中没有survey字段")
        return None
    
    if not references:
        print(f"警告: {method_full_name}.json 中没有reference字段")
        return None
    
    print(f"Survey长度: {len(survey)} 字符")
    print(f"引用数量: {len(references)}")
    
    # 提取claims和引用
    print("正在提取claims和引用...")
    claims_with_refs = extract_citations_and_claims(survey)
    print(f"找到 {len(claims_with_refs)} 个包含引用的claim")
    
    # 构建引用缓存（优先使用REF.jsonl文件）
    ref_cache = build_ref_cache(references, ref_jsonl_file)
    
    # 计算三个指标
    print("\n正在计算IoU (Insertion over Union)...")
    iou = calculate_iou(claims_with_refs, ref_cache, references, client)
    
    # Relevancesemantic和RelevanceLLM暂时注释掉，设为-1
    # print("正在计算Relevancesemantic...")
    # relevance_semantic = calculate_relevance_semantic(claims_with_refs, ref_cache, references)
    relevance_semantic = -1
    
    # print("正在计算RelevanceLLM...")
    # relevance_llm = calculate_relevance_llm(claims_with_refs, ref_cache, references, client)
    relevance_llm = -1
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"实例名: {method_full_name}")
    print(f"IoU (Insertion over Union): {iou:.4f}")
    print(f"Relevancesemantic: {relevance_semantic}")
    print(f"RelevanceLLM: {relevance_llm}")
    print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    return {
        "name": method_full_name,
        "lourele": [iou, relevance_semantic, relevance_llm],
        "_output_file": output_file
    }


def append_to_eval_jsonl(result):
    """追加结果到eval.jsonl文件"""
    output_file = result.pop("_output_file", None)
    if output_file is None:
        output_file = BASE_DIR / "t1" / "eval.jsonl"
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    """主函数（兼容原有调用方式）"""
    # 读取API key
    api_key = read_api_key()
    if not api_key:
        print("错误: 无法读取API key")
        return
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    # 原有调用方式：处理t1目录下的a.json, x.json, f.json
    methods = ['a', 'x', 'f']
    
    for method in methods:
        result = evaluate_method(method, client)
        if result:
            append_to_eval_jsonl(result)
    
    print("\n完成！结果已保存到:", BASE_DIR / "t1" / "eval.jsonl")


if __name__ == '__main__':
    main()
