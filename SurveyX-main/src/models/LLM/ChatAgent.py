"""
@reference:
1.发送本地图片： https://www.cnblogs.com/Vicrooor/p/18227547
"""

import fcntl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import threading

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from pathlib import Path

from src.configs.config import (
    REMOTE_URL,
    LOCAL_URL,
    TOKEN,
    BASE_DIR,
    DEFAULT_CHATAGENT_MODEL,
    CHAT_AGENT_WORKERS,
)
from src.configs.constants import OUTPUT_DIR

from src.configs.logger import get_logger
from src.models.LLM.utils import encode_image
from src.models.monitor.token_monitor import TokenMonitor

logger = get_logger("src.models.LLM.ChatAgent")
logger.debug(f"ChatAgent pid={os.getpid()}")

# 全局请求速率限制器（跨进程共享，使用文件锁确保同步）
_last_request_time = 0.0
_rate_limit_lock = threading.Lock()

def _rate_limit_request():
    """
    速率限制：确保请求之间的最小间隔，避免突发大量请求导致API限流。
    使用文件锁实现跨进程同步（适用于多个 offline_run.py 并行运行）。
    """
    from src.configs.config import API_REQUEST_MIN_INTERVAL
    global _last_request_time
    
    lock_file_path = Path(f"{OUTPUT_DIR}/tmp/api_rate_limit.lock")
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用文件锁实现跨进程同步
        with open(lock_file_path, 'a') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                current_time = time.time()
                elapsed = current_time - _last_request_time
                if elapsed < API_REQUEST_MIN_INTERVAL:
                    sleep_time = API_REQUEST_MIN_INTERVAL - elapsed
                    time.sleep(sleep_time)
                _last_request_time = time.time()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        # 如果文件锁失败，使用线程锁作为备选方案（仅在单进程内有效）
        with _rate_limit_lock:
            current_time = time.time()
            elapsed = current_time - _last_request_time
            if elapsed < API_REQUEST_MIN_INTERVAL:
                sleep_time = API_REQUEST_MIN_INTERVAL - elapsed
                time.sleep(sleep_time)
            _last_request_time = time.time()
        logger.debug(f"文件锁失败，使用线程锁: {e}")


class ChatAgent:
    Cost_file = Path(f"{OUTPUT_DIR}/tmp/cost.txt")
    Request_stats_file = Path(f"{OUTPUT_DIR}/tmp/request_stats.txt")
    Record_splitter = "||"
    Record_show_length = 200

    def __init__(
        self,
        token_monitor: TokenMonitor | None = None,
        token: str | None = None,
        remote_url: str = REMOTE_URL,
        local_url: str = LOCAL_URL,
    ) -> None:
        # Use provided token, or fall back to current TOKEN value (allows dynamic loading)
        if token is None:
            from src.configs.config import TOKEN as CURRENT_TOKEN
            token = CURRENT_TOKEN
        self.remote_url = remote_url
        self.token = token
        self.local_url = local_url
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Connection": "close",  # 避免连接复用导致的问题
        }
        self.batch_workers = CHAT_AGENT_WORKERS
        self.token_monitor = token_monitor
        
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
                max_retries=0,       # 不在 adapter 层重试，由 tenacity 处理
                pool_block=False,
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
            # 设置默认 headers
            self._session.headers.update(self.header)
    
    def __del__(self):
        """清理 Session"""
        try:
            self._reset_session()
        except Exception:
            pass  # 忽略清理时的错误

    @retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(min=2, max=300),  # 最小等待时间增加到2秒，给服务器恢复时间
        retry=retry_if_exception_type(requests.RequestException),
    )
    def remote_chat(
        self,
        text_content: str,
        image_urls: list[str] = None,
        local_images: list[Path] = None,
        temperature: float = 0.5,
        debug: bool = False,
        model: str | None = None,
    ) -> str:
        """chat with remote LLM, return result."""
        if model is None:
            from src.configs.config import DEFAULT_CHATAGENT_MODEL
            model = DEFAULT_CHATAGENT_MODEL
        url = self.remote_url
        # 确保 session 已初始化
        if self._session is None:
            self._init_session()
        header = self.header.copy()  # 使用副本，避免修改原始对象
        # text content
        messages = [{"role": "user", "content": text_content}]
        # insert image urls ----
        if (
            image_urls is not None
            and isinstance(image_urls, list)
            and len(image_urls) > 0
        ):
            image_url_frame = []
            for url_ in image_urls:
                image_url_frame.append(
                    {"type": "image_url", "image_url": {"url": url_}}
                )
            image_message_frame = {"role": "user", "content": image_url_frame}
            messages.append(image_message_frame)

        # insert local images ----
        if (
            local_images is not None
            and isinstance(local_images, list)
            and len(local_images) > 0
        ):
            local_image_frame = []
            for local_image in local_images:
                local_encoded_image = encode_image(local_image)
                local_image_frame.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{local_encoded_image}"
                        },
                    }
                )
            image_message_frame = {"role": "user", "content": local_image_frame}
            messages.append(image_message_frame)

        payload = {"model": model, "messages": messages, "temperature": temperature}

        # 支持代理配置（用于绕过地区限制）
        from src.configs.config import HTTP_PROXY, HTTPS_PROXY
        proxies = None
        if HTTP_PROXY or HTTPS_PROXY:
            proxies = {
                'http': HTTP_PROXY,
                'https': HTTPS_PROXY
            }
        
        # 设置超时时间（连接超时 10 秒，读取超时 120 秒）
        # 连接超时：建立连接的时间上限
        # 读取超时：等待服务器响应的时间上限（LLM 生成可能需要较长时间）
        timeout = (10, 120)  # (connect_timeout, read_timeout)
        
        # 速率限制：确保请求间隔，避免突发大量请求
        _rate_limit_request()
        
        # 添加日志：发送请求前
        request_start_time = time.time()
        logger.info(f"[ChatAgent] 准备发送请求到 {url}, 模型: {model}, 超时: {timeout}")
        logger.debug(f"[ChatAgent] 请求内容长度: {len(text_content)} 字符, 消息数: {len(messages)}")
        logger.debug(f"[ChatAgent] 使用 Connection: close 头和优化的 Session")
        
        try:
            # 使用 Session 发送请求，更好地管理连接
            response = self._session.post(
                url, 
                json=payload, 
                proxies=proxies, 
                timeout=timeout,
                headers=header
            )
            elapsed_time = time.time() - request_start_time
            logger.info(f"[ChatAgent] 收到响应, 状态码: {response.status_code}, 耗时: {elapsed_time:.2f}秒")
        except requests.Timeout as e:
            elapsed_time = time.time() - request_start_time
            logger.error(f"[ChatAgent] 请求超时: {e} (连接超时 {timeout[0]}s, 读取超时 {timeout[1]}s, 实际耗时: {elapsed_time:.2f}秒)")
            # 超时后重置 session，强制建立新连接
            self._reset_session()
            raise
        except requests.ConnectionError as e:
            elapsed_time = time.time() - request_start_time
            err_str = str(e)
            # 详细记录连接错误
            if "Connection reset by peer" in err_str or "104" in err_str:
                logger.error(
                    f"[ChatAgent] 连接被服务器重置 (Connection reset by peer, 耗时: {elapsed_time:.2f}秒) - "
                    f"可能原因: 1)服务器过载/限流 2)网络不稳定 3)请求频率过高。将重置连接并重试。"
                )
            elif "Failed to resolve" in err_str or "NameResolutionError" in err_str or "name resolution" in err_str.lower():
                logger.error(
                    f"[ChatAgent] DNS 解析失败 (DNS resolution failed, 耗时: {elapsed_time:.2f}秒) - "
                    f"无法解析域名。可能原因: 1)DNS服务器问题 2)网络连接中断 3)域名暂时不可用。"
                    f"解决方案（无需sudo）: 1)等待后重试（DNS问题通常是临时的，程序会自动重试） 2)检查网络连接 "
                    f"3)重启程序（新进程会重新进行DNS解析） 4)检查域名是否正确"
                )
            else:
                logger.error(f"[ChatAgent] 连接错误: {e}, 耗时: {elapsed_time:.2f}秒")
            # 连接错误后重置 session，强制建立新连接
            self._reset_session()
            raise
        except Exception as e:
            elapsed_time = time.time() - request_start_time
            logger.error(f"[ChatAgent] 请求失败: {type(e).__name__}: {e}, 耗时: {elapsed_time:.2f}秒")
            # 其他错误也重置 session
            self._reset_session()
            raise

        # 处理响应
        if response.status_code != 200:
            logger.error(
                f"chat response code: {response.status_code}\n{response.text[:500]}, retrying..."
            )
            status_code = 0 if response.status_code != 200 else 1

            # 加了线程锁
            self.update_record(
                status_code=status_code,
                response_code=response.status_code,
                request=text_content,
                response=response.text,
            )
            response.raise_for_status()
        
        try:
            res = json.loads(response.text)
            res_text = res["choices"][0]["message"]["content"]
            # 更新总开销
            # token monitor
            if self.token_monitor:
                self.token_monitor.add_token(
                    model=model,
                    input_tokens=res["usage"]["prompt_tokens"],
                    output_tokens=res["usage"]["completion_tokens"],
                )
        except Exception as e:
            res_text = f"Error: {e}"
            logger.error(f"There is an error: {e}")

        status_code = 0 if response.status_code != 200 else 1
        self.update_record(
            status_code=status_code,
            response_code=response.status_code,
            request=text_content,
            response=res_text,
        )

        if debug:
            return res_text, response
        return res_text
    
    def _reset_session(self):
        """重置 Session，强制关闭现有连接"""
        try:
            if self._session is not None:
                self._session.close()
                logger.debug("[ChatAgent] Session 已关闭，将在下次请求时重新创建")
        except Exception as e:
            logger.debug(f"[ChatAgent] 关闭 Session 时出错: {e}")
        finally:
            self._session = None

    # map chat index
    def __remote_chat(
        self,
        index,
        content,
        temperature: float = 0.5,
        debug: bool = False,
        model: str | None = None,
    ):
        if model is None:
            from src.configs.config import DEFAULT_CHATAGENT_MODEL
            model = DEFAULT_CHATAGENT_MODEL
        return index, self.remote_chat(
            text_content=content,
            image_urls=None,
            local_images=None,
            temperature=temperature,
            debug=debug,
            model=model,
        )

    def batch_remote_chat(
        self,
        prompt_l: list[str],
        desc: str = "batch_chating...",
        workers: int = None,
        temperature: float = 0.5,
        model: str = None,
    ) -> list[str]:
        """
        开启多线程进行对话
        """
        if workers is None:
            workers = self.batch_workers
        if model is None:
            from src.configs.config import DEFAULT_CHATAGENT_MODEL, CHAT_AGENT_WORKERS
            model = DEFAULT_CHATAGENT_MODEL
        # 创建线程池
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交任务
            future_l = [
                executor.submit(self.__remote_chat, i, prompt_l[i], temperature, False, model)
                for i in range(len(prompt_l))
            ]
            # 领取任务结果
            res_l = ["no response"] * len(prompt_l)
            for future in tqdm(
                as_completed(future_l),
                desc=desc,
                total=len(future_l),
                dynamic_ncols=True,
            ):
                i, resp = future.result()
                res_l[i] = resp
        return res_l

    @classmethod
    def update_record(
        cls, status_code: int, response_code: int, request: str, response: str
    ):
        "维护记录文件"
        content = (
            f"{status_code}{cls.Record_splitter}{response_code}{cls.Record_splitter}{request[: cls.Record_show_length]}{cls.Record_splitter}{response[: cls.Record_show_length]}".replace(
                "\n", ""
            )
            + "\n"
        )
        # 检查文件是否存在
        if not os.path.exists(cls.Request_stats_file):
            parent_dir = Path(cls.Request_stats_file).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            with open(cls.Request_stats_file, "w", encoding="utf-8") as fw:
                fcntl.flock(fw, fcntl.LOCK_EX)  # 加锁
                fw.write(content)
                logger.info(
                    f"record file {cls.Request_stats_file} did not exist, created and initialized with 0.0"
                )
                fcntl.flock(fw, fcntl.LOCK_UN)
        # 更新开销总计
        try:
            with open(cls.Request_stats_file, "a", encoding="utf-8") as fw:
                fcntl.flock(fw, fcntl.LOCK_EX)
                fw.write(content)
                fcntl.flock(fw, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to update cost: {e}")

    def local_chat(self, query, debug=False) -> str:
        """
        调用本地LLM进行推理, 保证端口已开启
        """
        query = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
            {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".format(query)

        payload = json.dumps(
            {
                "prompt": query,
                "temperature": 1.0,
                "max_tokens": 102400,
                "n": 1,
                # 可选的参数在这里：https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
            }
        )
        headers = {"Content-Type": "application/json"}
        res = requests.request("POST", self.local_url, headers=headers, data=payload)
        if res.status_code != 200:
            logger.info("chat response code: {}".format(res.status_code), query[:20])
            return "chat response code: {}".format(res.status_code)
        if debug:
            return res
        return res.json()["text"][0].replace(query, "")

    def __local_chat(self, index, query):
        return index, self.local_chat(query, debug=True)

    def batch_local_chat(self, query_l, worker=16, desc="bach local inferencing..."):
        """
        多线程本地推理
        """
        with ThreadPoolExecutor(max_workers=worker) as executor:
            # 提交任务
            future_l = [
                executor.submit(self.__local_chat, i, query_l[i])
                for i in range(len(query_l))
            ]
            # 领取任务结果
            res_l = ["no response"] * len(query_l)
            for future in tqdm(as_completed(future_l), desc=desc, total=len(future_l)):
                i, resp = future.result()
                res_l[i] = resp
        return res_l

    @staticmethod
    def show_request_stats():
        stats_file = ChatAgent.Request_stats_file
        logger.info(f"stats_file: {stats_file}")

        with stats_file.open("r", encoding="utf-8") as fr:
            succ_count = 0
            total_count = 0
            for line in fr:
                elements = line.strip().split(ChatAgent.Record_splitter)
                succ_count += int(elements[0])
                total_count += 1
            logger.info(f"请求成功率：{round(succ_count / total_count * 100, 2)}%")

    @staticmethod
    def clean_request_stats():
        stats_file = ChatAgent.Request_stats_file
        if stats_file.exists():
            logger.info(f"remove {stats_file}.")


if __name__ == "__main__":
    agent = ChatAgent()
    text_content = "图片里面有什么"

    # result = agent.remote_chat(text_content="今天天气怎么样",  model="gpt-4o")
    # print(result)
    #
    # image_urls = ["https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"]
    # result = agent.remote_chat( text_content=text_content, image_urls=image_urls, temperature=0.5, model="gpt-4o")
    # print(result)

    local_images = [f"{BASE_DIR}/resources/dummy_data/figs/dog_and_girl.jpeg"]
    result = agent.remote_chat(
        text_content=text_content,
        local_images=local_images,
        temperature=0.5,
        model="gpt-4o",
    )
    print(result)

    ChatAgent.show_request_stats()
