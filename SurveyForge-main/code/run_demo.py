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

import subprocess
from datetime import datetime
import time
import re
import argparse
import json

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        return True
    return False

def extract_token_usage(log_content):
    """从日志内容中提取token使用信息"""
    token_info = {
        'outline_input': 0,
        'outline_output': 0,
        'subsection_input': 0,
        'subsection_output': 0
    }
    
    patterns = {
        'outline_input': r'OutlineWriter Input token usage: (\d+)',
        'outline_output': r'OutlineWriter Output token usage: (\d+)',
        'subsection_input': r'SubsectionWriter Input token usage: (\d+)',
        'subsection_output': r'SubsectionWriter Output token usage: (\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, log_content)
        if match:
            token_info[key] = int(match.group(1))
    
    return token_info

def run_experiment(topic, folder_name, exp_num, base_path, model_name, api_key):

    save_path = os.path.join(base_path, folder_name, f"exp_{exp_num}")
    

    # 检查目录是否存在，以及是否有真正的结果文件
    if os.path.exists(save_path):
        # 检查是否有真正的结果文件（.md 或 .json）
        result_files = [f for f in os.listdir(save_path) if f.endswith(('.md', '.json')) or f.startswith('outlines_')]
        if result_files:
            print(f"\nSkipping experiment {exp_num} for topic {topic} - results already exist: {save_path}")
            print(f"Found result files: {', '.join(result_files)}")
            return True
        else:
            # 目录存在但没有结果文件，删除目录重新运行
            print(f"\nDirectory {save_path} exists but no result files found. Removing and re-running...")
            import shutil
            shutil.rmtree(save_path)
    

    create_directory(save_path)
    

    cmd = [
        "python", "main.py",
        "--topic", topic,
        "--gpu", "0",
        "--saving_path", save_path,
        "--model", model_name,
        "--section_num", "7",
        "--subsection_len", "500",
        "--rag_num", "100",
        "--rag_max_out", "60",
        "--outline_reference_num", "1500",
        "--survey_outline_path", "./",
        "--db_path", "/home/liudingyuan/code/SurveyX/code/SurveyForge_database/database",
        "--embedding_model", "/home/liudingyuan/code/SurveyX/code/gte-large-en-v1.5",
        "--api_key", api_key,
        "--api_url", "https://api.ai-gaochao.cn/v1"
    ]

    
    exp_start_time = datetime.now()
    
    try:
        print(f"\nRunning experiment {exp_num} for topic: {topic}")
        print(f"Saving to: {save_path}")
        print(f"Start time: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Capture the output of the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        full_output = []
        while True:
            output = process.stdout.readline()

            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                full_output.append(output)
        
        return_code = process.poll()
        exp_end_time = datetime.now()
        duration = exp_end_time - exp_start_time
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        token_info = extract_token_usage(''.join(full_output))
        
        # Log experiment times
        log_file = os.path.join(base_path, "experiment_times.log")
        with open(log_file, "a") as f:
            f.write(f"Topic: {topic}, Exp {exp_num}\n")
            f.write(f"Start: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End: {exp_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"OutlineWriter Input token usage: {token_info['outline_input']}\n")
            f.write(f"OutlineWriter Output token usage: {token_info['outline_output']}\n")
            f.write(f"SubsectionWriter Input token usage: {token_info['subsection_input']}\n")
            f.write(f"SubsectionWriter Output token usage: {token_info['subsection_output']}\n")
            f.write("-" * 50 + "\n")
        
        print(f"Experiment {exp_num} for {topic} completed successfully")
        print(f"End time: {exp_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        exp_end_time = datetime.now()
        duration = exp_end_time - exp_start_time
        error_message = f"Error in experiment {exp_num} for {topic}: {e}"
        print(error_message)
        print(f"Failed experiment duration: {duration}")
        

        log_file = os.path.join(base_path, "experiment_times.log")
        with open(log_file, "a") as f:
            f.write(f"Topic: {topic}, Exp {exp_num} [FAILED]\n")
            f.write(f"Start: {exp_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End: {exp_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("-" * 50 + "\n")
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Run SurveyForge experiments')
    parser.add_argument('--topic', type=str, required=True, 
                       help='Topic for the survey (e.g., "A Survey of Large Language Models")')
    parser.add_argument('--exp_num', type=int, default=1,
                       help='Number of experiments to run (default: 1)')
    parser.add_argument('--mid', type=int, default=0,
                       help='Model ID (default: 0). mid=0 uses t1, mid=1,2,3... uses t1a,t1b,t1c...')
    parser.add_argument('--topic_json', type=str, default='../../EVAL/code/topic.json',
                       help='Path to topic.json file (default: ../../EVAL/code/topic.json)')
    parser.add_argument('--model_json', type=str, default='../../EVAL/code/amodelid.json',
                       help='Path to amodelid.json file (default: ../../EVAL/code/amodelid.json)')
    parser.add_argument('--api_key_file', type=str, default='../../EVAL/code/akey.txt',
                       help='Path to akey.txt file (default: ../../EVAL/code/akey.txt)')
    args = parser.parse_args()

    base_path = "./output/res"
    create_directory(base_path)
    
    # 使用命令行参数传递的主题
    topic = args.topic
    exp_count = args.exp_num
    mid = args.mid
    
    # 读取 API key（从文件最后一行获取）
    api_key = ""
    api_key_file = args.api_key_file
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    api_key = lines[-1].strip()
                    print(f"Loaded API key from {api_key_file}")
        except Exception as e:
            print(f"Error reading {api_key_file}: {e}")
    else:
        print(f"Warning: {api_key_file} not found")
    
    # 读取模型ID配置
    model_name = "gpt-4o"  # 默认模型
    model_json_path = args.model_json
    if os.path.exists(model_json_path):
        try:
            with open(model_json_path, 'r', encoding='utf-8') as f:
                models_data = json.load(f)
            
            # 查找匹配的 mid
            for item in models_data:
                if item.get('id') == mid:
                    model_name = item.get('model_name', 'gpt-4o')
                    print(f"Found matching model ID in {model_json_path}: id={mid}, model_name={model_name}")
                    break
            else:
                print(f"Warning: Model ID {mid} not found in {model_json_path}, using default gpt-4o")
        except Exception as e:
            print(f"Error reading {model_json_path}: {e}, using default gpt-4o")
    else:
        print(f"Warning: {model_json_path} not found, using default gpt-4o")
    
    # 读取 topic.json 并查找匹配的 topic
    topic_json_path = args.topic_json
    folder_name = topic  # 默认使用 topic 作为文件夹名
    topic_id = None
    
    if os.path.exists(topic_json_path):
        try:
            with open(topic_json_path, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
            
            # 查找匹配的 topic
            for item in topics_data:
                if item.get('topic') == topic:
                    topic_id = item.get('id')
                    if mid == 0:
                        folder_name = f"t{topic_id}"
                    else:
                        # mid=1,2,3... 对应 a,b,c...
                        suffix = chr(ord('a') + mid - 1)
                        folder_name = f"t{topic_id}{suffix}"
                    print(f"Found matching topic in {topic_json_path}: id={topic_id}, mid={mid}, folder_name={folder_name}")
                    break
            else:
                print(f"Warning: Topic '{topic}' not found in {topic_json_path}, using topic as folder name")
        except Exception as e:
            print(f"Error reading {topic_json_path}: {e}, using topic as folder name")
    else:
        print(f"Warning: {topic_json_path} not found, using topic as folder name")

    start_time = datetime.now()
    print(f"Starting experiments at: {start_time}")
    print(f"Topic: {topic}")
    print(f"Model ID (mid): {mid}")
    print(f"Model name: {model_name}")
    print(f"Folder name: {folder_name}")
    print(f"Number of experiments: {exp_count}")

    log_file = os.path.join(base_path, "experiment_times.log")
    log_exists = os.path.exists(log_file)
    
    with open(log_file, "a" if log_exists else "w") as f:
        if not log_exists:
            f.write(f"Experiment Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
        else:
            f.write(f"\nResuming experiments at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    

    print(f"\n{'='*50}")
    print(f"Starting experiments for topic: {topic}")
    print(f"{'='*50}")
    
    topic_start_time = datetime.now()
    successful_exps = 0
    
    for exp_num in range(1, exp_count + 1):  
        success = run_experiment(topic, folder_name, exp_num, base_path, model_name, api_key)
        if success:
            successful_exps += 1

        time.sleep(5)

    topic_end_time = datetime.now()
    topic_duration = topic_end_time - topic_start_time
    with open(log_file, "a") as f:
        f.write(f"\nTopic Summary: {topic}\n")
        f.write(f"Total Duration: {topic_duration}\n")
        f.write(f"Successful Experiments: {successful_exps}/{exp_count}\n")
        f.write("=" * 50 + "\n")
        

    end_time = datetime.now()
    duration = end_time - start_time
    
    with open(log_file, "a") as f:
        f.write(f"\nFinal Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {duration}\n")
        f.write(f"Total Topics: 1\n")
    
    print(f"\nAll experiments completed!")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {duration}")
    print(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main()
