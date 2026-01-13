#!/usr/bin/env python3
"""
修复 embedding 模型加载问题的辅助脚本
用于手动下载模型到本地，避免网络超时问题
"""

import os
from sentence_transformers import SentenceTransformer

def download_model_to_local():
    """下载 nomic-embed-text-v1 模型到本地"""
    model_name = "nomic-ai/nomic-embed-text-v1"
    local_path = "./model/nomic-embed-text-v1"
    
    print(f"正在下载模型 {model_name} 到 {local_path}...")
    
    # 创建目录
    os.makedirs("./model", exist_ok=True)
    
    try:
        # 使用镜像站点（如果网络有问题）
        # 设置环境变量使用镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.save(local_path)
        print(f"✓ 模型已成功下载到 {local_path}")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n备选方案：")
        print("1. 使用 HuggingFace 镜像站点")
        print("2. 配置代理")
        print("3. 手动从 https://huggingface.co/nomic-ai/nomic-embed-text-v1 下载")
        return False

if __name__ == "__main__":
    download_model_to_local()

