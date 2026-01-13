import os
import json
import argparse
from src.agents.outline_writer import outlineWriter
from src.agents.writer import subsectionWriter
from src.agents.judge import Judge
from src.database import database
from tqdm import tqdm
import time
from datetime import datetime

def get_api_key(akey_path=None):
    """
    从 akey.txt 文件读取 API key（多行时取最后一行）
    """
    if akey_path and os.path.exists(akey_path):
        possible_paths = [akey_path]
    else:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EVAL/code/akey.txt'),
            os.path.join(os.path.dirname(__file__), '../EVAL/code/akey.txt'),
            './EVAL/code/akey.txt',
            '../EVAL/code/akey.txt',
            '../../EVAL/code/akey.txt'
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 取最后一行，去除空白字符
                    if lines:
                        return lines[-1].strip()
            except Exception as e:
                print(f"Warning: Failed to read {path}: {e}")
                continue
    
    print(f"Warning: Could not find akey.txt file. Using empty API key.")
    return ''

def get_model_from_mid(mid, amodelid_path=None):
    """
    根据 mid 从 amodelid.json 获取模型名称
    """
    if amodelid_path and os.path.exists(amodelid_path):
        possible_paths = [amodelid_path]
    else:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EVAL/code/amodelid.json'),
            os.path.join(os.path.dirname(__file__), '../EVAL/code/amodelid.json'),
            './EVAL/code/amodelid.json',
            '../EVAL/code/amodelid.json',
            '../../EVAL/code/amodelid.json'
        ]
    
    model_data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                break
            except Exception as e:
                print(f"Warning: Failed to read {path}: {e}")
                continue
    
    if model_data is None:
        print(f"Warning: Could not find amodelid.json file. Using default model 'gpt-4o'.")
        return 'gpt-4o'
    
    # 查找匹配的 mid
    for item in model_data:
        if item.get('id') == mid:
            model_name = item.get('model_name')
            if model_name:
                return model_name
    
    print(f"Warning: Model id {mid} not found in amodelid.json. Using default model 'gpt-4o'.")
    return 'gpt-4o'

def get_topic_id(topic, topic_json_path=None, mid=0):
    """
    根据 topic 值在 topic.json 中查找对应的 id
    返回格式为 't{id}' 或 't{id}{suffix}' 的字符串，如果未找到则返回 None
    mid=0 时返回 tn，mid=1,2,3... 时返回 tna, tnb, tnc
    """
    # 尝试多个可能的路径
    if topic_json_path and os.path.exists(topic_json_path):
        possible_paths = [topic_json_path]
    else:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EVAL/code/topic.json'),
            os.path.join(os.path.dirname(__file__), '../EVAL/code/topic.json'),
            './EVAL/code/topic.json',
            '../EVAL/code/topic.json',
            '../../EVAL/code/topic.json'
        ]
    
    topic_data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    topic_data = json.load(f)
                break
            except Exception as e:
                print(f"Warning: Failed to read {path}: {e}")
                continue
    
    if topic_data is None:
        print(f"Warning: Could not find topic.json file. Using topic name as folder name.")
        return None
    
    # 查找匹配的 topic
    for item in topic_data:
        if item.get('topic') == topic:
            topic_id = item.get('id')
            if topic_id is not None:
                # 根据 mid 添加后缀
                if mid == 0:
                    return f"t{topic_id}"
                else:
                    # mid=1 -> 'a', mid=2 -> 'b', mid=3 -> 'c', ...
                    suffix = chr(ord('a') + mid - 1)
                    return f"t{topic_id}{suffix}"
    
    print(f"Warning: Topic '{topic}' not found in topic.json. Using topic name as folder name.")
    return None

def remove_descriptions(text):
    lines = text.split('\n')
    
    filtered_lines = [line for line in lines if not line.strip().startswith("Description")]
    
    result = '\n'.join(filtered_lines)
    
    return result

def write(topic, model, section_num, subsection_len, rag_num, refinement):
    outline, outline_wo_description = write_outline(topic, model, section_num)

    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = True)
        return refined_survey_with_references
    else:
        raw_survey, raw_survey_with_references, raw_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = False)
        return raw_survey_with_references

def write_outline(topic, model, section_num, outline_reference_num, db, api_key, api_url):
    outline_writer = outlineWriter(model=model, api_key=api_key, api_url = api_url, database=db)
    print(outline_writer.api_model.chat('hello'))
    outline = outline_writer.draft_outline(topic, outline_reference_num, 30000, section_num)
    return outline, remove_descriptions(outline), outline_writer

def write_subsection(topic, model, outline, subsection_len, rag_num, db, api_key, api_url, refinement = True):

    subsection_writer = subsectionWriter(model=model, api_key=api_key, api_url = api_url, database=db)
    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, refining = True)
        return raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references, subsection_writer
    else:
        raw_survey, raw_survey_with_references, raw_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, refining = False)
        return raw_survey, raw_survey_with_references, raw_references, subsection_writer

def load_config(config_file='mainconfig.txt'):
    """
    从配置文件加载参数
    配置文件格式：key=value，每行一个参数
    """
    config = {}
    config_paths = [
        config_file,
        os.path.join(os.path.dirname(__file__), config_file),
        './mainconfig.txt'
    ]
    
    config_path = None
    for path in config_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        config[key] = value
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")
    else:
        print(f"Warning: Config file not found. Using default values.")
    
    return config

def paras_args():
    # 先加载配置文件
    config = load_config()
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', default=config.get('gpu', '0'), type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path', default=config.get('saving_path', './output/'), type=str, help='Directory to save the output survey')
    parser.add_argument('--model', default=config.get('model', 'gpt-4o'), type=str, help='Model to use (will be overridden by mid if provided)')
    parser.add_argument('--topic', default='', type=str, help='Topic to generate survey for')
    parser.add_argument('--mid', default=0, type=int, help='Model id to select from amodelid.json (default: 0)')
    parser.add_argument('--section_num', default=int(config.get('section_num', '7')), type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len', default=int(config.get('subsection_len', '700')), type=int, help='Length of each subsection')
    parser.add_argument('--outline_reference_num', default=int(config.get('outline_reference_num', '1500')), type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num', default=int(config.get('rag_num', '60')), type=int, help='Number of references to use for RAG')
    parser.add_argument('--api_url', default=config.get('api_url', 'https://api.openai.com/v1/chat/completions'), type=str, help='url for API request')
    parser.add_argument('--api_key', default=None, type=str, help='API key for the model (if not provided, will read from akey.txt)')
    parser.add_argument('--db_path', default=config.get('db_path', './database'), type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model', default=config.get('embedding_model', 'nomic-ai/nomic-embed-text-v1'), type=str, help='Embedding model for retrieval.')
    topic_json_path = config.get('topic_json_path', None)
    if topic_json_path == '':
        topic_json_path = None
    parser.add_argument('--topic_json_path', default=topic_json_path, type=str, help='Path to topic.json file for topic id mapping.')
    parser.add_argument('--config', default='mainconfig.txt', type=str, help='Path to config file.')
    
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载
    if args.config != 'mainconfig.txt':
        config = load_config(args.config)
        # 更新参数（命令行参数优先级最高）
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # 处理空字符串
    if args.topic_json_path == '':
        args.topic_json_path = None
    
    # 从 akey.txt 读取 API key（如果命令行未提供）
    if args.api_key is None or args.api_key == '':
        args.api_key = get_api_key()
        if args.api_key:
            print(f"Loaded API key from akey.txt")
        else:
            print(f"Warning: No API key found. Please provide --api_key or ensure akey.txt exists.")
    
    # 根据 mid 从 amodelid.json 获取模型名称
    model_from_mid = get_model_from_mid(args.mid)
    if model_from_mid:
        args.model = model_from_mid
        print(f"Using model from amodelid.json (mid={args.mid}): {args.model}")
    
    return args

def main(args):

    # 根据 topic 和 mid 查找对应的 id，生成输出文件夹名
    topic_folder_name = get_topic_id(args.topic, args.topic_json_path, args.mid)
    if topic_folder_name is None:
        # 如果未找到匹配，使用 topic 名称（去除特殊字符）
        topic_folder_name = args.topic.replace('/', '_').replace('\\', '_').replace(':', '_')
        if args.mid > 0:
            # 添加 mid 后缀
            suffix = chr(ord('a') + args.mid - 1)
            topic_folder_name = f"{topic_folder_name}{suffix}"
        print(f"Using folder name: {topic_folder_name}")
    else:
        print(f"Found topic id mapping: '{args.topic}' (mid={args.mid}) -> {topic_folder_name}")
    
    # 创建基于 topic id 的输出目录
    topic_output_dir = os.path.join(args.saving_path, topic_folder_name)
    if not os.path.exists(topic_output_dir):
        os.makedirs(topic_output_dir)
    
    # 初始化时间和 token 监控字典
    time_monitor = {}
    token_monitor = {}
    
    db = database(db_path = args.db_path, embedding_model = args.embedding_model)
    api_key = args.api_key

    # 1. Generate outline 部分（内部会准确记录检索数据库的时间）
    outline_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline_start = time.time()
    
    outline_with_description, outline_wo_description, outline_writer = write_outline(
        args.topic, args.model, args.section_num, args.outline_reference_num, db, args.api_key, args.api_url
    )
    
    outline_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline_end = time.time()
    
    # 记录检索数据库部分的时间和 token（从 outline_writer 获取准确时间）
    if outline_writer.retrieval_time_info:
        time_monitor["retrieve database"] = outline_writer.retrieval_time_info
    else:
        # 如果未记录，使用默认值
        time_monitor["retrieve database"] = {
            "start": outline_start_time,
            "end": outline_start_time,
            "duration": 0.0
        }
    
    token_monitor["retrieve database"] = {
        args.model: {
            "input_tokens": 0,  # 检索部分无 token 消耗
            "output_tokens": 0,
            "total_cost": 0
        }
    }
    
    # 记录 generate outline 的时间和 token（使用 outline_writer 记录的生成时间）
    if outline_writer.outline_generation_time_info:
        time_monitor["generate outline"] = outline_writer.outline_generation_time_info
    else:
        # 如果未记录，使用整体时间
        time_monitor["generate outline"] = {
            "start": outline_start_time,
            "end": outline_end_time,
            "duration": round(outline_end - outline_start, 2)
        }
    
    token_monitor["generate outline"] = {
        args.model: {
            "input_tokens": outline_writer.input_token_usage,
            "output_tokens": outline_writer.output_token_usage,
            "total_cost": 0  # 如果需要计算成本，可以使用 outline_writer.compute_price()
        }
    }
    
    # 2. Generate content 部分
    content_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content_start = time.time()
    
    raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references, subsection_writer = write_subsection(
        args.topic, args.model, outline_with_description, args.subsection_len, args.rag_num, db, args.api_key, args.api_url
    )
    
    content_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content_end = time.time()
    content_duration = content_end - content_start
    
    # 记录 generate content 的时间和 token
    time_monitor["generate content"] = {
        "start": content_start_time,
        "end": content_end_time,
        "duration": round(content_duration, 2)
    }
    
    token_monitor["generate content"] = {
        args.model: {
            "input_tokens": subsection_writer.input_token_usage,
            "output_tokens": subsection_writer.output_token_usage,
            "total_cost": 0  # 如果需要计算成本，可以使用 subsection_writer.compute_price()
        }
    }
    
    # 收集所有检索到的文献 ID（按检索顺序，去重）
    all_retrieved_ids = []
    # 从 outline_writer 获取
    if hasattr(outline_writer, 'retrieved_paper_ids') and outline_writer.retrieved_paper_ids:
        all_retrieved_ids.extend(outline_writer.retrieved_paper_ids)
    # 从 subsection_writer 获取
    if hasattr(subsection_writer, 'retrieved_paper_ids') and subsection_writer.retrieved_paper_ids:
        all_retrieved_ids.extend(subsection_writer.retrieved_paper_ids)
    
    # 去重但保持顺序（使用字典的键顺序特性）
    seen = set()
    unique_retrieved_ids = []
    for paper_id in all_retrieved_ids:
        if paper_id not in seen:
            seen.add(paper_id)
            unique_retrieved_ids.append(paper_id)
    
    # 限制最多1000篇
    unique_retrieved_ids = unique_retrieved_ids[:1000]
    
    # 格式化为 reference 格式：{"1": "arxiv_id", "2": "arxiv_id", ...}
    retrieveref = {}
    for idx, paper_id in enumerate(unique_retrieved_ids, start=1):
        retrieveref[str(idx)] = paper_id
    
    # 保存输出文件
    # 1. 保存 a.json（原 {topic}.json）
    with open(f'{topic_output_dir}/a.json', 'w', encoding='utf-8') as f:
        save_dic = {}
        save_dic['survey'] = refined_survey_with_references
        save_dic['reference'] = refined_references
        save_dic['retrieveref'] = retrieveref
        json.dump(save_dic, f, indent=4, ensure_ascii=False)
    
    # 2. 保存 markdown 文件
    with open(f'{topic_output_dir}/{args.topic}.md', 'w', encoding='utf-8') as f:
        f.write(refined_survey_with_references)
    
    # 3. 保存 ac.json（合并时间和 token 监控）
    ac_data = {
        "time": time_monitor,
        "token": token_monitor
    }
    with open(f'{topic_output_dir}/ac.json', 'w', encoding='utf-8') as f:
        json.dump(ac_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    args = paras_args()

    main(args)