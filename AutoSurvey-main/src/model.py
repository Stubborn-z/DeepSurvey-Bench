import time
from openai import OpenAI
from tqdm import tqdm
import threading

class APIModel:

    def __init__(self, model, api_key, api_url) -> None:
        self.__api_key = api_key
        self.__api_url = api_url
        self.model = model
        
        # 处理 api_url 格式：如果包含 /chat/completions，需要去掉
        # OpenAI 客户端的 base_url 应该是基础 URL，如 https://api.openai.com/v1
        base_url = api_url
        if '/chat/completions' in base_url:
            base_url = base_url.replace('/chat/completions', '')
        # 确保以 /v1 结尾（如果没有的话）
        if not base_url.endswith('/v1'):
            if base_url.endswith('/'):
                base_url = base_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'
        
        # 使用 OpenAI 客户端，与 sk.py 中的方式一致
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def __req(self, text, temperature, max_try = 5):
        """
        使用 OpenAI 客户端发送请求，与 sk.py 中的方式一致
        """
        messages = [
            {"role": "user", "content": text}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            # 重试机制
            for attempt in range(max_try):
                try:
                    time.sleep(0.2 * (attempt + 1))  # 递增延迟
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature
                    )
                    return completion.choices[0].message.content
                except Exception as retry_error:
                    if attempt == max_try - 1:
                        # 最后一次重试失败，打印错误信息
                        print(f"API 调用失败 (尝试 {attempt + 1}/{max_try}): {retry_error}")
                    pass
            return None
    
    def chat(self, text, temperature=1):
        response = self.__req(text, temperature=temperature, max_try=5)
        return response

    def __chat(self, text, temperature, res_l, idx):
        
        response = self.__req(text, temperature=temperature)
        res_l[idx] = response
        return response
        
    def batch_chat(self, text_batch, temperature=0):
        max_threads=15 # limit max concurrent threads using model API
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self.__chat, args=(text, temperature, res_l, i))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads: 
                for t in thread_l:
                    if not t .is_alive():
                        thread_l.remove(t)
                time.sleep(0.3) # Short delay to avoid busy-waiting

        for thread in tqdm(thread_l):
            thread.join()
        return res_l
