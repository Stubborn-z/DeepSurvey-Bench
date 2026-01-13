import os
# 必须在导入其他库之前设置 Hugging Face 镜像和离线模式
# 如果网络连接有问题，使用镜像: https://hf-mirror.com
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"Set HF_ENDPOINT to: {os.environ['HF_ENDPOINT']}")
# 强制使用离线模式（使用缓存中的文件）
if 'HF_HUB_OFFLINE' not in os.environ:
    os.environ['HF_HUB_OFFLINE'] = '1'
    print("Set HF_HUB_OFFLINE to 1 (using cached files)")

import json
import argparse
from src.agents.outline_writer import outlineWriter
from src.agents.writer import subsectionWriter
from src.database import database, database_survey
from src.rag import GeneralRAG_langchain
from tqdm import tqdm
import time
import re


def remove_descriptions_subquery(text):
    lines = text.split('\n')
    
    filtered_lines = [line for line in lines if line.strip().startswith("#")]
    
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

def write_outline(args, topic, model, ckpt, section_num, outline_reference_num, db, api_key, api_url, retrieved_ids_list):
    from datetime import datetime
    
    # 记录开始时间
    outline_start_time = datetime.now()
    outline_start_str = outline_start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    outline_writer = outlineWriter(args=args, model=model, ckpt=ckpt, api_key=api_key, api_url = api_url, database=db)
    print(outline_writer.api_model.chat('hello'))
    outline = outline_writer.draft_outline(topic, outline_reference_num, 30000, section_num, retrieved_ids_list)
    outline_writer.print_token_usage()
    
    # 记录结束时间
    outline_end_time = datetime.now()
    outline_end_str = outline_end_time.strftime("%Y-%m-%d %H:%M:%S")
    outline_duration = (outline_end_time - outline_start_time).total_seconds()
    
    # 保存文件并重命名为 fo.txt
    filename_1 = f"{args.saving_path}/fo.txt"
    with open(filename_1, "w") as f:
        f.write(outline + '\n\n')
    filename_2 = f"{args.saving_path}/outlines_without_des_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename_2, "w") as f:
        f.write(remove_descriptions_subquery(outline) + '\n\n')
        
    outline_writer.print_token_usage()
    
    print(outline)

    def duplicate_first_last_sections(markdown_content):

        pattern = r'(## \d+\.?\s*.*?(?=\n##|\Z))'
        sections = re.findall(pattern, markdown_content, re.DOTALL)
        
        if len(sections) < 2:
            return markdown_content  
        
        first_section = sections[0]
        last_section = sections[-1]
        

        first_section_number = re.search(r'## (\d+)', first_section).group(1)
        first_title = first_section.split('\n')[0].strip()
        first_content = '\n'.join(first_section.split('\n')[1:]).strip()
        new_first_section = (f"{first_title}\n{first_content}\n\n"
                            f"### {first_section_number}.1 {first_title.split(maxsplit=2)[-1]}\n"
                            f"Description: {first_content}\n\n")
        

        last_section_number = re.search(r'## (\d+)', last_section).group(1)
        last_title = last_section.split('\n')[0].strip()
        last_content = '\n'.join(last_section.split('\n')[1:]).strip()
        new_last_section = (f"{last_title}\n{last_content}\n\n"
                            f"### {last_section_number}.1 {last_title.split(maxsplit=2)[-1]}\n"
                            f"Description: {last_content}\n")
        

        markdown_content = markdown_content.replace(first_section, new_first_section)
        markdown_content = markdown_content.replace(last_section, new_last_section)
        
        return markdown_content

    outline = duplicate_first_last_sections(outline)

    # 返回时间和token信息（retrieved_ids_list通过引用传递，会被直接修改）
    return outline, remove_descriptions_subquery(outline), {
        'start': outline_start_str,
        'end': outline_end_str,
        'duration': outline_duration
    }, {
        'input_tokens': outline_writer.input_token_usage,
        'output_tokens': outline_writer.output_token_usage
    }

def write_subsection(args, topic, model, ckpt, outline, subsection_len, rag_num, rag_max_out, db, api_key, api_url, retrieved_ids_list, refinement = True):
    from datetime import datetime
    
    # 记录开始时间
    content_start_time = datetime.now()
    content_start_str = content_start_time.strftime("%Y-%m-%d %H:%M:%S")
    def remove_first_last_subsection_titles(markdown_content):
        subsection_pattern = r'\n(### \d+\.\d+[^\n]*)\n'
        subsections = re.findall(subsection_pattern, markdown_content)
        
        if len(subsections) < 2:
            return markdown_content
        
        first_subsection = subsections[0]
        last_subsection = subsections[-1]

        new_content = re.sub(r'\n' + re.escape(first_subsection) + r'\n', '\n', markdown_content)

        new_content = re.sub(r'\n' + re.escape(last_subsection) + r'\n', '\n', new_content)

        new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
        
        return new_content
    
    subsection_writer = subsectionWriter(args=args, model=model, ckpt=ckpt, api_key=api_key, api_url = api_url, database=db)
    
    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, rag_max_out=rag_max_out, refining = True, retrieved_ids_list=retrieved_ids_list)
        subsection_writer.print_token_usage()
        
        # 记录结束时间
        content_end_time = datetime.now()
        content_end_str = content_end_time.strftime("%Y-%m-%d %H:%M:%S")
        content_duration = (content_end_time - content_start_time).total_seconds()
        
        return raw_survey, raw_survey_with_references, raw_references, remove_first_last_subsection_titles(refined_survey), remove_first_last_subsection_titles(refined_survey_with_references), refined_references, {
            'start': content_start_str,
            'end': content_end_str,
            'duration': content_duration
        }, {
            'input_tokens': subsection_writer.input_token_usage,
            'output_tokens': subsection_writer.output_token_usage
        }
    else:
        raw_survey, raw_survey_with_references, raw_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, rag_max_out=rag_max_out, refining = False, retrieved_ids_list=retrieved_ids_list)
        subsection_writer.print_token_usage()
        
        # 记录结束时间
        content_end_time = datetime.now()
        content_end_str = content_end_time.strftime("%Y-%m-%d %H:%M:%S")
        content_duration = (content_end_time - content_start_time).total_seconds()
        
        return remove_first_last_subsection_titles(raw_survey), remove_first_last_subsection_titles(raw_survey_with_references), raw_references, {
            'start': content_start_str,
            'end': content_end_str,
            'duration': content_duration
        }, {
            'input_tokens': subsection_writer.input_token_usage,
            'output_tokens': subsection_writer.output_token_usage
        }
    

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the output survey')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model',default='gpt-4o', type=str, help='Model to use')
    parser.add_argument('--ckpt',default='', type=str, help='Checkpoint to use')
    parser.add_argument('--topic',default='Multimodal Large Language Models', type=str, help='Topic to generate survey for')
    parser.add_argument('--section_num',default=6, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=500, type=int, help='Length of each subsection')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=100, type=int, help='Number of references to use for RAG')
    parser.add_argument('--rag_max_out',default=60, type=int, help='Number of references to use for RAG')
    parser.add_argument('--api_url',default='https://api.openai.com/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key',default='', type=str, help='API key for the model')
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--survey_outline_path',default='', type=str, help='Directory of the outline database of survey.')
    parser.add_argument('--embedding_model',default='./gte-large-en-v1.5', type=str, help='Embedding model for retrieval.')
    args = parser.parse_args()
    return args

def main(args):
    from datetime import datetime
    
    print(args)
    print("########### Loading database and RAG Index... ###########")
    
    # 记录检索数据库的开始时间
    retrieval_start_time = datetime.now()
    retrieval_start_str = retrieval_start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    db_paper = database(db_path = args.db_path, embedding_model = args.embedding_model)
    db_survey = database_survey(db_path = args.db_path, embedding_model = args.embedding_model)

    abs_index_db_path = f'{args.db_path}/faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin'
    title_index_db_path = f'{args.db_path}/faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin'
    doc_db_path = f'{args.db_path}/arxiv_paper_db_with_cc.json'
    arxivid_to_index_path = f'{args.db_path}/arxivid_to_index_abs.json'
    
    rag_abstract4outline = GeneralRAG_langchain(args=args,
                                                retriever_type='vectorstore',
                                                index_db_path=abs_index_db_path,
                                                doc_db_path=doc_db_path,
                                                arxivid_to_index_path=arxivid_to_index_path,
                                                embedding_model=args.embedding_model)

    rag_abstract4suboutline = rag_abstract4outline
        
    rag_abstract4subsection = rag_abstract4outline

    rag_title4citation = GeneralRAG_langchain(args=args,
                                              retriever_type='vectorstore',
                                              index_db_path=title_index_db_path,
                                              doc_db_path=doc_db_path,
                                              arxivid_to_index_path=arxivid_to_index_path,
                                              embedding_model=args.embedding_model,
                                              embedding_function=rag_abstract4outline.embedding_function)  # 复用 embedding 函数以节省 GPU 内存

    # 记录检索数据库的结束时间
    retrieval_end_time = datetime.now()
    retrieval_end_str = retrieval_end_time.strftime("%Y-%m-%d %H:%M:%S")
    retrieval_duration = (retrieval_end_time - retrieval_start_time).total_seconds()

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)
    db = {
        "paper": db_paper, 
        "survey": db_survey,
        "rag_outline": rag_abstract4outline, 
        "rag_suboutline": rag_abstract4suboutline,
        "rag_subsection": rag_abstract4subsection,
        "rag_title4citation": rag_title4citation
    }
    
    # 创建列表来收集所有检索到的文献ID（最多1000篇）
    retrieved_ids_list = []
    
    print("########### Writing outline... ###########")
    
    outline_with_description, outline_wo_description, outline_time_info, outline_token_info = \
        write_outline(args, args.topic, args.model, args.ckpt, args.section_num, args.outline_reference_num, db, args.api_key, args.api_url, retrieved_ids_list)
    
    print("########### Writing content... ###########")

    raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references, content_time_info, content_token_info = \
        write_subsection(args, args.topic, args.model, args.ckpt, outline_with_description, args.subsection_len, args.rag_num, args.rag_max_out, db, args.api_key, args.api_url, retrieved_ids_list)

    # 将检索到的ID列表转换为reference格式（字典，键是序号字符串，值是arxiv ID）
    retrieveref = {}
    for idx, paper_id in enumerate(retrieved_ids_list[:1000], start=1):
        retrieveref[str(idx)] = paper_id

    # 重命名文件：{topic}.json -> f.json
    with open(f'{args.saving_path}/f.json', 'w') as f:
        save_dic = {}
        save_dic['survey'] = refined_survey_with_references
        save_dic['reference'] = refined_references
        save_dic['retrieveref'] = retrieveref
        f.write(json.dumps(save_dic, indent=4))
    
    # 保存原始md文件
    with open(f'{args.saving_path}/{args.topic}.md', 'w') as f:
        f.write(refined_survey_with_references)
    
    # 创建metrics目录
    metrics_dir = f'{args.saving_path}/metrics'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # 生成time_monitor.json
    time_monitor = {
        "retrieve database": {
            "start": retrieval_start_str,
            "end": retrieval_end_str,
            "duration": round(retrieval_duration, 2)
        },
        "generate outline": {
            "start": outline_time_info['start'],
            "end": outline_time_info['end'],
            "duration": round(outline_time_info['duration'], 2)
        },
        "generate content": {
            "start": content_time_info['start'],
            "end": content_time_info['end'],
            "duration": round(content_time_info['duration'], 2)
        }
    }
    
    with open(f'{metrics_dir}/time_monitor.json', 'w') as f:
        json.dump(time_monitor, f, indent=4)
    
    # 生成token_monitor.json
    token_monitor = {
        "retrieve database": {
            args.model: {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0
            }
        },
        "generate outline": {
            args.model: {
                "input_tokens": outline_token_info['input_tokens'],
                "output_tokens": outline_token_info['output_tokens'],
                "total_cost": 0.0  # 如果需要计算成本，可以在这里添加
            }
        },
        "generate content": {
            args.model: {
                "input_tokens": content_token_info['input_tokens'],
                "output_tokens": content_token_info['output_tokens'],
                "total_cost": 0.0  # 如果需要计算成本，可以在这里添加
            }
        }
    }
    
    with open(f'{metrics_dir}/token_monitor.json', 'w') as f:
        json.dump(token_monitor, f, indent=4)
    
    # 合并为fc.json
    fc_data = {
        "time": time_monitor,
        "token": token_monitor
    }
    
    with open(f'{args.saving_path}/fc.json', 'w') as f:
        json.dump(fc_data, f, indent=4)

if __name__ == '__main__':

    args = paras_args()

    main(args)
