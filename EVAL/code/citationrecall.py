#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import threading
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import urllib.request
import urllib.parse

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AKEY_FILE = os.path.join(BASE_DIR, "code", "akey.txt")
T1_DIR = os.path.join(BASE_DIR, "t1")
OUTPUT_FILE = os.path.join(BASE_DIR, "t1", "eval.jsonl")

# 模型配置
MODEL = "gpt-4o"
BASE_URL = "https://api.ai-gaochao.cn/v1"

# NLI Prompt模板
NLI_PROMPT = '''---
Claim:
[CLAIM]
---
Source: 
[SOURCE]
---
Claim:
[CLAIM]
---
Is the Claim faithful to the Source? 
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.

Only reply with 'Yes' or 'No':
'''


def read_api_key():
    """从akey.txt读取API key，如果有多行则取最后一行"""
    try:
        with open(AKEY_FILE, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines[-1] if lines else None
    except Exception as e:
        print(f"警告: 无法读取API key: {e}")
        return None


def extract_arxiv_id(arxiv_id_with_version):
    """从带版本的arxiv ID中提取ID（去掉版本号）"""
    # 例如 "2201.08239v3" -> "2201.08239"
    return arxiv_id_with_version.split('v')[0] if 'v' in arxiv_id_with_version else arxiv_id_with_version


def fetch_arxiv_abstract(arxiv_id):
    """从arXiv API获取论文摘要"""
    arxiv_id_clean = extract_arxiv_id(arxiv_id)
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id_clean}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                xml_content = response.read().decode('utf-8')
                
                # 使用XML解析库（如果可用）或正则表达式
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(xml_content)
                    # arXiv API 命名空间
                    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
                    summary_elem = root.find('.//arxiv:summary', ns)
                    if summary_elem is not None and summary_elem.text:
                        abstract = summary_elem.text.strip()
                        # 清理换行和多余空格
                        abstract = ' '.join(abstract.split())
                        return abstract
                except:
                    # 如果XML解析失败，使用正则表达式作为备选
                    summary_match = re.search(r'<summary>(.*?)</summary>', xml_content, re.DOTALL)
                    if summary_match:
                        abstract = summary_match.group(1).strip()
                        # 清理HTML实体和换行
                        abstract = abstract.replace('\n', ' ').replace('  ', ' ')
                        return abstract
                
            print(f"警告: 无法从arXiv API获取摘要 for {arxiv_id_clean}")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"错误: 获取arXiv摘要失败 {arxiv_id_clean}: {e}")
            return None
    
    return None


def extract_num(string):
    """从字符串中提取第一个数字"""
    numbers = re.findall(r'\d+', string)
    if len(numbers) == 0:
        return ''
    return int(numbers[0])


def extract_citations_and_claims(survey):
    """从survey文本中提取引用句子和对应的声明"""
    # 分割survey（如果有References部分则去掉）
    survey_text = survey.split('## References')[0] if '## References' in survey else survey
    
    # 使用正则匹配包含引用的句子
    # 匹配模式：[任意字符][引用标记][任意字符][句号/问号/感叹号]
    citation_pattern = re.compile(r'[^.!?]*\[[^\]]+\][^.!?]*[.!?]')
    sentences = citation_pattern.findall(survey_text)
    
    claims = []
    sources_ids = []
    
    for s in sentences:
        # 提取所有引用标记中的内容，例如 [1], [1;2], [1; 2]
        sources = re.findall(pattern=r'\[(.*?)\]', string=s)
        if len(sources) > 0:
            source_ids = set()
            for ref in sources:
                # 支持分号分隔的多个引用，例如 "1;2" 或 "1; 2"
                for num in ref.split(';'):
                    num_clean = num.strip()
                    number = extract_num(num_clean)
                    if number != '':
                        source_ids.add(number)
            
            if len(source_ids) > 0:
                # 移除引用标记得到声明文本
                claim = re.sub(pattern=r'\[(.*?)\]', repl='', string=s).strip()
                if claim:  # 确保声明不为空
                    claims.append(claim)
                    sources_ids.append(list(source_ids))
    
    return claims, sources_ids


def generate_nli_prompt(claim, source_text):
    """生成NLI prompt"""
    prompt = NLI_PROMPT.replace('[CLAIM]', claim).replace('[SOURCE]', source_text)
    return prompt


def call_llm_nli(client, prompt, max_retries=3):
    """调用LLM进行NLI判断"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                timeout=30
            )
            response = completion.choices[0].message.content.strip()
            
            # 检查响应中是否包含'yes'
            if 'yes' in response.lower():
                return 1
            else:
                return 0
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"错误: LLM调用失败: {e}")
            return 0
    return 0


def __nli_worker(sources, claim, res_l, idx, client):
    """NLI工作线程函数"""
    source_text = '\n'.join(sources)
    prompt = generate_nli_prompt(claim, source_text)
    res = call_llm_nli(client, prompt)
    res_l[idx] = res




def load_ref_jsonl(ref_jsonl_path):
    """从REF.jsonl文件加载论文信息
    返回: dict {id: {"arxivid": ..., "title": ..., "abstract": ...}}
    """
    ref_data = {}
    if not os.path.exists(ref_jsonl_path):
        return ref_data
    
    try:
        with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ref_id = item.get('id', '')
                if ref_id:
                    ref_data[ref_id] = {
                        "arxivid": item.get('arxiv', ''),
                        "title": item.get('title', ''),
                        "abstract": item.get('abstract', '')
                    }
    except Exception as e:
        print(f"警告: 读取 {ref_jsonl_path} 失败: {e}")
    
    return ref_data


def calculate_citation_metrics(survey, references, client, ref_jsonl_path=None):
    """计算Citation Recall和Precision
    ref_jsonl_path: 可选的REF.jsonl文件路径，如果提供则使用它而不是arXiv API
    """
    print("正在提取引用和声明...")
    claims, sources_ids = extract_citations_and_claims(survey)
    
    if len(claims) == 0:
        print("警告: 未找到任何包含引用的句子")
        return 0.0, 0.0
    
    print(f"找到 {len(claims)} 个包含引用的声明")
    
    # 建立引用编号到摘要的映射
    index_to_abstract = {}
    
    if ref_jsonl_path and os.path.exists(ref_jsonl_path):
        # 使用REF.jsonl文件
        print(f"从 {ref_jsonl_path} 加载论文信息...")
        ref_data = load_ref_jsonl(ref_jsonl_path)
        
        for index_str, arxiv_id in references.items():
            index = int(index_str)
            index_str_key = str(index)
            
            if index_str_key in ref_data:
                # 使用REF.jsonl中的数据
                abstract = ref_data[index_str_key].get('abstract', '')
                if abstract == '':
                    abstract = None  # 空字符串表示缺失
                index_to_abstract[index] = abstract if abstract else ""
            else:
                index_to_abstract[index] = ""
    else:
        # 从arXiv获取论文摘要
        print("正在从arXiv获取论文摘要...")
        arxiv_id_to_abstract = {}
        unique_arxiv_ids = list(set(references.values()))
        
        for arxiv_id in tqdm(unique_arxiv_ids, desc="获取摘要"):
            abstract = fetch_arxiv_abstract(arxiv_id)
            if abstract:
                arxiv_id_to_abstract[arxiv_id] = abstract
            else:
                print(f"警告: 无法获取 {arxiv_id} 的摘要")
                arxiv_id_to_abstract[arxiv_id] = ""  # 使用空字符串作为占位符
            time.sleep(0.5)  # 避免API频率限制
        
        # 建立引用编号到摘要的映射
        for index_str, arxiv_id in references.items():
            index = int(index_str)
            if arxiv_id in arxiv_id_to_abstract:
                index_to_abstract[index] = arxiv_id_to_abstract[arxiv_id]
            else:
                index_to_abstract[index] = ""
    
    # 计算Citation Recall
    print("正在计算Citation Recall...")
    thread_l = []
    scores = [0] * len(claims)
    
    for i in range(len(claims)):
        # 获取该声明对应的所有引用的摘要
        source_abstracts = []
        for source_id in sources_ids[i]:
            if source_id in index_to_abstract:
                abstract = index_to_abstract[source_id]
                if abstract:  # 只添加非空摘要
                    source_abstracts.append(abstract)
        
        if len(source_abstracts) > 0:
            thread = threading.Thread(target=__nli_worker, args=(source_abstracts, claims[i], scores, i, client))
            thread_l.append(thread)
            thread.start()
        else:
            # 如果没有有效的摘要，标记为不支持
            scores[i] = 0
    
    # 等待所有线程完成
    for thread in tqdm(thread_l, desc="Recall评估"):
        thread.join()
    
    citation_recall = np.array(scores).mean()
    
    # 计算Citation Precision
    print("正在计算Citation Precision...")
    citation_num = 0
    thread_l = []
    precision_counter = []  # 使用列表来收集所有precision分数
    precision_lock = threading.Lock()  # 线程锁
    
    def __relevant_worker_return(sources, com_sources, claim, client):
        """检查引用是否必要的 worker 函数，返回结果"""
        source_text = '\n'.join(sources)
        prompt = generate_nli_prompt(claim, source_text)
        res = call_llm_nli(client, prompt)
        
        if res == 1:
            # 如果支持，检查其他引用是否也能支持
            if len(com_sources) > 0:
                com_source_text = '\n'.join(com_sources)
                com_prompt = generate_nli_prompt(claim, com_source_text)
                com_res = call_llm_nli(client, com_prompt)
                if com_res == 1:
                    # 其他引用也能支持，当前引用不必要
                    return 0
            # 其他引用不能支持，当前引用必要
            return 1
        else:
            return 0
    
    for j, (claim, source_ids) in enumerate(zip(claims, sources_ids)):
        citation_num += len(source_ids)
        if scores[j] == 1:  # 只对支持的声明计算precision
            for source_id in source_ids:
                if source_id in index_to_abstract and index_to_abstract[source_id]:
                    sources = [index_to_abstract[source_id]]
                    # 其他引用
                    com_sources = [index_to_abstract[_] for _ in source_ids if _ != source_id and _ in index_to_abstract and index_to_abstract[_]]
                    
                    # 使用functools.partial或lambda来正确传递参数
                    def worker_wrapper(s, cs, c, cl):
                        result = __relevant_worker_return(s, cs, c, cl)
                        with precision_lock:
                            precision_counter.append(result)
                    
                    thread = threading.Thread(target=worker_wrapper, args=(sources, com_sources, claim, client))
                    thread_l.append(thread)
                    thread.start()
    
    # 等待所有线程完成
    for thread in tqdm(thread_l, desc="Precision评估"):
        thread.join()
    
    citation_precision = np.array(precision_counter).sum() / citation_num if citation_num > 0 else 0.0
    
    return citation_recall, citation_precision


def append_to_eval_jsonl(output_file, name, metric_name, value):
    """追加结果到eval.jsonl文件"""
    result = {"name": name, metric_name: value}
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def check_already_evaluated(output_file, name):
    """检查是否已经评估过
    返回: (has_recall, has_precision)
    """
    if not os.path.exists(output_file):
        return False, False
    
    has_recall = False
    has_precision = False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get('name') == name:
                        if 'citationrecall' in item:
                            has_recall = True
                        if 'citationprecision' in item:
                            has_precision = True
                except:
                    continue
    except Exception:
        pass
    
    return has_recall, has_precision


def evaluate_instance(t_dir, method_name, output_file, client):
    """评估单个实例
    t_dir: 目录路径（完整路径，如 /path/to/t1）
    method_name: 方法名（如 'a', 'a1', 'f2' 等）
    output_file: 输出文件路径
    client: OpenAI客户端
    返回: (recall, precision) 或 None（如果失败或已评估）
    """
    # 检查是否已评估
    has_recall, has_precision = check_already_evaluated(output_file, method_name)
    if has_recall and has_precision:
        print(f"  跳过 {method_name}（已评估）")
        return None
    
    json_file = os.path.join(t_dir, f"{method_name}.json")
    if not os.path.exists(json_file):
        return None
    
    # 检查是否有REF.jsonl文件
    ref_jsonl_file = os.path.join(t_dir, f"{method_name}REF.jsonl")
    ref_jsonl_path = ref_jsonl_file if os.path.exists(ref_jsonl_file) else None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        survey = data.get('survey', '')
        references = data.get('reference', {})
        
        if not survey:
            print(f"  警告: {method_name}.json 中没有survey字段")
            return None
        
        if not references:
            print(f"  警告: {method_name}.json 中没有reference字段")
            return None
        
        # 计算指标
        recall, precision = calculate_citation_metrics(survey, references, client, ref_jsonl_path)
        
        print(f"  {method_name}: Recall={recall:.4f}, Precision={precision:.4f}")
        
        # 写入结果
        append_to_eval_jsonl(output_file, method_name, "citationrecall", recall)
        append_to_eval_jsonl(output_file, method_name, "citationprecision", precision)
        
        return recall, precision
        
    except Exception as e:
        print(f"  错误: 处理 {method_name}.json 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数 - 支持命令行参数或默认行为"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估Citation Recall和Precision')
    parser.add_argument('--t_dir', type=str, default=None, help='目录名（如 t1），默认使用t1')
    parser.add_argument('--method', type=str, default=None, help='方法名（如 a, a1, f2），默认处理所有方法')
    
    args = parser.parse_args()
    
    # 读取API key
    api_key = read_api_key()
    if not api_key:
        print("错误: 无法读取API key")
        return
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    # 确定目录
    t_dir_name = args.t_dir if args.t_dir else 't1'
    t_dir = os.path.join(BASE_DIR, t_dir_name)
    output_file = os.path.join(t_dir, "eval.jsonl")
    
    # 确保目录存在
    if not os.path.isdir(t_dir):
        print(f"错误: 目录 {t_dir} 不存在")
        return
    
    if args.method:
        # 处理单个方法
        result = evaluate_instance(t_dir, args.method, output_file, client)
        if result is None:
            print(f"评估失败或已跳过")
    else:
        # 默认行为：处理 a, x, f
        methods = ['a', 'x', 'f']
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"处理方法: {method}")
            print(f"{'='*60}")
            evaluate_instance(t_dir, method, output_file, client)
    
    print("\n完成！")


if __name__ == '__main__':
    main()

