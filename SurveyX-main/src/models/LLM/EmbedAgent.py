import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fcntl
import threading

import requests
from requests.adapters import HTTPAdapter
import time
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None
from tqdm import tqdm

from src.configs.config import (
    DEFAULT_EMBED_LOCAL_MODEL,
    DEFAULT_EMBED_ONLINE_MODEL,
    EMBED_REMOTE_URL,
    EMBED_TOKEN,
)
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger

logger = get_logger("src.models.LLM.EmbedAgent")

# 全局请求速率限制器（与 ChatAgent 共享相同的速率限制）
_last_embed_request_time = 0.0
_embed_rate_limit_lock = threading.Lock()

def _rate_limit_embed_request():
    """
    速率限制：确保请求之间的最小间隔，避免突发大量请求导致API限流。
    使用文件锁实现跨进程同步。
    """
    from src.configs.config import API_REQUEST_MIN_INTERVAL
    global _last_embed_request_time
    
    # 使用相同的锁文件，确保所有API请求共享速率限制
    lock_file_path = Path(f"{OUTPUT_DIR}/tmp/api_rate_limit.lock")
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用文件锁实现跨进程同步
        with open(lock_file_path, 'a') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                current_time = time.time()
                elapsed = current_time - _last_embed_request_time
                if elapsed < API_REQUEST_MIN_INTERVAL:
                    sleep_time = API_REQUEST_MIN_INTERVAL - elapsed
                    time.sleep(sleep_time)
                _last_embed_request_time = time.time()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        # 如果文件锁失败，使用线程锁作为备选方案
        with _embed_rate_limit_lock:
            current_time = time.time()
            elapsed = current_time - _last_embed_request_time
            if elapsed < API_REQUEST_MIN_INTERVAL:
                sleep_time = API_REQUEST_MIN_INTERVAL - elapsed
                time.sleep(sleep_time)
            _last_embed_request_time = time.time()
        logger.debug(f"EmbedAgent 文件锁失败，使用线程锁: {e}")


class EmbedAgent:
    """
    A class to handle remote text embedding using a specified API.
    Supports multi-threading for batch processing.
    """

    def __init__(self, token=None, remote_url=None) -> None:
        """
        Initialize the EmbedAgent.

        Args:
            token (str): Authentication token for the remote API. If None, uses current EMBED_TOKEN from config.
            remote_url (str): URL of the remote embedding API. If None, uses current EMBED_REMOTE_URL from config.
        """
        # Use provided values, or fall back to current config values (allows dynamic loading)
        if token is None:
            from src.configs.config import EMBED_TOKEN as CURRENT_EMBED_TOKEN
            token = CURRENT_EMBED_TOKEN
        if remote_url is None:
            from src.configs.config import EMBED_REMOTE_URL as CURRENT_EMBED_REMOTE_URL
            remote_url = CURRENT_EMBED_REMOTE_URL
        self.remote_url = remote_url
        self.token = token
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Connection": "close",  # 避免连接复用导致的问题
        }
        self.local_embedding_model = None
        
        # 创建优化的 Session，配置连接池和重试策略
        self._session = None
        self._init_session()
    
    def _init_session(self):
        """初始化优化的 requests Session"""
        if self._session is None:
            self._session = requests.Session()
            # 配置连接池参数，减少连接复用问题
            adapter = HTTPAdapter(
                pool_connections=1,  # 连接池大小
                pool_maxsize=1,      # 最大连接数
                max_retries=0,       # 不在 adapter 层重试，由自定义重试逻辑处理
                pool_block=False,
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
            # 设置默认 headers
            self._session.headers.update(self.header)
    
    def _reset_session(self):
        """重置 Session，强制关闭现有连接"""
        try:
            if self._session is not None:
                self._session.close()
                logger.debug("[EmbedAgent] Session 已关闭，将在下次请求时重新创建")
        except Exception as e:
            logger.debug(f"[EmbedAgent] 关闭 Session 时出错: {e}")
        finally:
            self._session = None
    
        # 初始化本地 embedding 模型
        if HuggingFaceEmbedding is not None:
            try:
                self.local_embedding_model = HuggingFaceEmbedding(
                    model_name=DEFAULT_EMBED_ONLINE_MODEL
                )
            except Exception as e:
                logger.info(
                    f"{e}\nFailed to load embedding model {DEFAULT_EMBED_ONLINE_MODEL}, try to use local model {DEFAULT_EMBED_LOCAL_MODEL}."
                )
                try:
                    self.local_embedding_model = HuggingFaceEmbedding(
                        model_name=DEFAULT_EMBED_LOCAL_MODEL
                    )
                except Exception as e2:
                    logger.warning(f"Failed to load local embedding model: {e2}. Local embedding will be disabled.")
                    self.local_embedding_model = None
        else:
            logger.info("HuggingFaceEmbedding not available. Local embedding will be disabled. Use remote_embed instead.")
    
    def __del__(self):
        """清理 Session"""
        try:
            self._reset_session()
        except Exception:
            pass  # 忽略清理时的错误

    def remote_embed(
        self,
        text: str,
        max_try: int = 15,
        debug: bool = False,
        model: str = "BAAI/bge-m3",
    ) -> list:
        """
        Embed text using the remote API.

        Args:
            text (str): Input text to embed.
            max_try (int, optional): Maximum number of retry attempts.
            debug (bool, optional): Whether to return debug information.
            model (str, optional): Model name for the remote API.

        Returns:
            list: Embedding vector or error message.
        """
        url = self.remote_url
        json_data = json.dumps(
            {"model": model, "input": text, "encoding_format": "float"}
        )

        # 确保 session 已初始化
        if self._session is None:
            self._init_session()

        # 速率限制：确保请求间隔，避免突发大量请求
        _rate_limit_embed_request()
        
        # 添加日志：发送请求前
        input_length = len(text) if isinstance(text, str) else len(str(text))
        request_start_time = time.time()
        logger.info(f"[EmbedAgent] 准备发送请求到 {url}, 模型: {model}, 输入长度: {input_length}")
        logger.debug(f"[EmbedAgent] 超时设置: connect=10s, read=60s, 使用 Connection: close")

        timeout = (10, 60)
        
        try:
            # 使用 Session 发送请求
            logger.debug(f"[EmbedAgent] 开始发送初始请求...")
            response = self._session.post(url, data=json_data, timeout=timeout, headers=self.header)
            elapsed_time = time.time() - request_start_time
            logger.info(f"[EmbedAgent] 初始请求成功，状态码: {response.status_code}, 耗时: {elapsed_time:.2f}秒")
        except requests.Timeout as e:
            elapsed_time = time.time() - request_start_time
            logger.error(f"[EmbedAgent] 初始请求超时: {e} (连接超时 10s, 读取超时 60s, 实际耗时: {elapsed_time:.2f}秒)")
            self._reset_session()
            response = None
        except requests.ConnectionError as e:
            elapsed_time = time.time() - request_start_time
            err_str = str(e)
            if "Connection reset by peer" in err_str or "104" in err_str:
                logger.error(
                    f"[EmbedAgent] 初始请求连接被重置 (Connection reset by peer, 耗时: {elapsed_time:.2f}秒) - "
                    f"将重置连接并重试"
                )
            else:
                logger.error(f"[EmbedAgent] 初始请求连接错误: {e}, 耗时: {elapsed_time:.2f}秒")
            self._reset_session()
            response = None
        except Exception as e:
            elapsed_time = time.time() - request_start_time
            logger.error(f"[EmbedAgent] 初始请求失败: {type(e).__name__}: {e}, 耗时: {elapsed_time:.2f}秒")
            self._reset_session()
            response = None
            
        if response is None:
            logger.warning(f"[EmbedAgent] 开始重试，最多重试 {max_try} 次")
            for attempt in range(max_try):
                try:
                    # 确保 session 已重新初始化（如果被重置）
                    if self._session is None:
                        self._init_session()
                    
                    # 等待后重试（指数退避）
                    wait_time = min(2 ** attempt, 30)  # 最多等待30秒
                    if attempt > 0:
                        logger.debug(f"[EmbedAgent] 等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    
                    logger.debug(f"[EmbedAgent] 重试 {attempt + 1}/{max_try}...")
                    retry_start_time = time.time()
                    response = self._session.post(url, data=json_data, timeout=timeout, headers=self.header)
                    retry_elapsed = time.time() - retry_start_time
                    
                    if response.status_code == 200:
                        logger.info(f"[EmbedAgent] 重试 {attempt + 1}/{max_try} 成功, 耗时: {retry_elapsed:.2f}秒")
                        break
                    else:
                        logger.warning(f"[EmbedAgent] 重试 {attempt + 1}/{max_try} 失败，状态码: {response.status_code}, 耗时: {retry_elapsed:.2f}秒")
                        response = None
                except requests.Timeout as e:
                    retry_elapsed = time.time() - retry_start_time if 'retry_start_time' in locals() else 0
                    logger.error(f"[EmbedAgent] 重试 {attempt + 1}/{max_try} 超时: {e}, 耗时: {retry_elapsed:.2f}秒")
                    self._reset_session()
                    response = None
                except requests.ConnectionError as e:
                    retry_elapsed = time.time() - retry_start_time if 'retry_start_time' in locals() else 0
                    err_str = str(e)
                    if "Connection reset by peer" in err_str or "104" in err_str:
                        logger.error(f"[EmbedAgent] 重试 {attempt + 1}/{max_try} 连接被重置 (Connection reset by peer), 耗时: {retry_elapsed:.2f}秒")
                    else:
                        logger.error(f"[EmbedAgent] 重试 {attempt + 1}/{max_try} 连接错误: {e}, 耗时: {retry_elapsed:.2f}秒")
                    self._reset_session()
                    response = None
                except Exception as e:
                    retry_elapsed = time.time() - retry_start_time if 'retry_start_time' in locals() else 0
                    logger.error(f"[EmbedAgent] 重试 {attempt + 1}/{max_try} 失败: {type(e).__name__}: {e}, 耗时: {retry_elapsed:.2f}秒")
                    self._reset_session()
                    response = None

        if response is None:
            error_msg = "embed response code: 000"
            if debug:
                return error_msg, response
            return []

        if response.status_code != 200:
            error_msg = f"embed response code: {response.status_code}\n{response.text}"
            logger.error(error_msg)
            if debug:
                return error_msg, response
            return []

        try:
            res = response.json()
            embedding = res["data"][0]["embedding"]
            if debug:
                return embedding, response
            return embedding
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            if debug:
                return "JSON decoding failed", response
            return []

    def __remote_embed_task(self, index: int, text: str):
        """
        Internal method to handle embedding tasks in threads.

        Args:
            index (int): Index of the text in the input list.
            text (str): Text to embed.

        Returns:
            tuple: (index, embedding)
        """
        embedding = self.remote_embed(text)
        return index, embedding

    def batch_remote_embed(
        self, texts: list[str], worker: int = 10, desc: str = "Batch Embedding..."
    ) -> list:
        """
        Batch process text embeddings using multi-threading.

        Args:
            texts (list[str]): List of texts to embed.
            worker (int, optional): Number of worker threads.
            desc (str, optional): Description for the progress bar.

        Returns:
            list: List of embedding vectors.
        """
        embeddings = ["no response"] * len(texts)
        with ThreadPoolExecutor(max_workers=worker) as executor:
            future_l = [
                executor.submit(self.__remote_embed_task, i, texts[i])
                for i in range(len(texts))
            ]
            for future in tqdm(
                as_completed(future_l),
                desc=desc,
                total=len(future_l),
                dynamic_ncols=True,
            ):
                i, embedding = future.result()
                embeddings[i] = embedding
        return embeddings

    def local_embed(self, text: str) -> list[float]:
        if self.local_embedding_model is None:
            raise RuntimeError("Local embedding model is not available. Please install llama-index-embeddings-huggingface or use remote_embed instead.")
        embedding = self.local_embedding_model.get_text_embedding(text)
        return embedding

    def batch_local_embed(self, text_l: list[str]) -> list[list[float]]:
        if self.local_embedding_model is None:
            raise RuntimeError("Local embedding model is not available. Please install llama-index-embeddings-huggingface or use batch_remote_embed instead.")
        embed_documents = self.local_embedding_model.get_text_embedding_batch(
            text_l, show_progress=True
        )
        return embed_documents


if __name__ == "__main__":
    text_list = ["text1", "text2", "text4"]
    embed_agent = EmbedAgent()
    embedding = embed_agent.batch_remote_embed(text_list)
    print(embedding)
    logger.info("Embedding complete.")

    embedding = embed_agent.batch_local_embed(text_list)
    print(embedding)
