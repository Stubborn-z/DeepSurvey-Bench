import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer,  AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import h5py
from src.utils import tokenCounter
import json
from tqdm import tqdm
import faiss
from tinydb import TinyDB, Query

import logging

# 设置 Hugging Face 镜像和离线模式（如果网络连接有问题）
# 如果在中国，可以使用: https://hf-mirror.com
# 如果遇到网络连接问题，取消下面的注释来启用镜像
if 'HF_ENDPOINT' not in os.environ:
    # 默认使用镜像，如果不需要可以注释掉
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 强制使用离线模式（使用缓存中的文件，不尝试从网络下载）
# 如果文件已经在缓存中，取消下面的注释来启用离线模式
if 'HF_HUB_OFFLINE' not in os.environ:
    os.environ['HF_HUB_OFFLINE'] = '1'

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)


class database():

    def __init__(self, db_path, embedding_model) -> None:
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)

        self.embedding_model.to(torch.device('cuda'))

        self.db = TinyDB(f'{db_path}/arxiv_paper_db_with_cc.json')
        self.table = self.db.table('cs_paper_info')

        self.User = Query()
        self.token_counter = tokenCounter()
        self.title_loaded_index = faiss.read_index(f'{db_path}/faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin')

        self.abs_loaded_index = faiss.read_index(f'{db_path}/faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin')
        self.id_to_index, self.index_to_id = self.load_index_arxivid(db_path)

    def load_index_arxivid(self, db_path):
        json_path = f'{db_path}/arxivid_to_index_abs.json'
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                id_to_index = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON file {json_path}: {e}")
        id_to_index = {id: int(index) for id, index in id_to_index.items()}
        index_to_id = {int(index): id for id, index in id_to_index.items()}
        return id_to_index, index_to_id
    
    def get_embeddings(self, batch_text):
        # batch_text = ['search_query: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings

    def get_embeddings_documents(self, batch_text):
        # batch_text = ['search_document: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings
        
    def batch_search(self, query_vectors, top_k=1, title=False):
        query_vectors = np.array(query_vectors).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vectors, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vectors, top_k)
        results = []
        for i, query in tqdm(enumerate(query_vectors)):
            result = [(self.index_to_id[idx], distances[i][j]) for j, idx in enumerate(indices[i]) if idx != -1]
            results.append([_[0] for _ in result])
        return results

    def search(self, query_vector, top_k=1, title=False):
        query_vector = np.array([query_vector]).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vector, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vector, top_k)
        results = [(self.index_to_id[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1]
        return [_[0] for _ in results]

    def get_ids_from_query(self, query, num,  shuffle = False):
        q = self.get_embeddings([query])[0]
        return self.search(q, top_k=num)

    def get_paper_info_from_ids(self, ids):
        result = self.table.search(self.User.id.one_of(ids))
        return result


class database_survey():

    def __init__(self, db_path, embedding_model) -> None:
        
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)

        self.embedding_model.to(torch.device('cuda'))

        self.db = TinyDB(f'{db_path}/surveys_arxiv_paper_db.json')
        self.table = self.db.table('survey_paper_info')

        self.User = Query()
        self.token_counter = tokenCounter()
        self.title_loaded_index = faiss.read_index(f'{db_path}/faiss_survey_title_embeddings_FROM_1501_TO_2409_gte.bin')

        self.abs_loaded_index = faiss.read_index(f'{db_path}/faiss_survey_title_abs_embeddings_FROM_1501_TO_2409_gte.bin')
        self.id_to_index, self.index_to_id = self.load_index_arxivid(db_path)

    def load_index_arxivid(self, db_path):
        json_path = f'{db_path}/surveys_arxivid_to_index_abs.json'
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                id_to_index = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON file {json_path}: {e}")
        id_to_index = {id: int(index) for id, index in id_to_index.items()}
        index_to_id = {int(index): id for id, index in id_to_index.items()}
        return id_to_index, index_to_id
    
    def get_embeddings(self, batch_text):
        # batch_text = ['search_query: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings

    def get_embeddings_documents(self, batch_text):
        # batch_text = ['search_document: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings
        
    def batch_search(self, query_vectors, top_k=1, title=False):
        query_vectors = np.array(query_vectors).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vectors, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vectors, top_k)
        results = []
        for i, query in tqdm(enumerate(query_vectors)):
            result = [(self.index_to_id[idx], distances[i][j]) for j, idx in enumerate(indices[i]) if idx != -1]
            results.append([_[0] for _ in result])
        return results

    def search(self, query_vector, top_k=1, title=False):
        query_vector = np.array([query_vector]).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vector, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vector, top_k)
        results = [(self.index_to_id[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1]
        return [_[0] for _ in results]

    def get_ids_from_query(self, query, num,  shuffle = False):
        q = self.get_embeddings([query])[0]
        return self.search(q, top_k=num)
    
    def get_paper_info_from_ids(self, ids):
        result = self.table.search(self.User.id.one_of(ids))
        return result
    