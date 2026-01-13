import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src.configs.constants import BASE_DIR, OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.configs.config import ADVANCED_CHATAGENT_MODEL
from src.models.LLM.utils import load_prompt
from src.modules.utils import save_result, update_config, sanitize_filename
from src.configs.config import DEFAULT_DATA_FETCHER_ENABLE_CACHE

logger = get_logger("src.modules.preprocessor.utils")


class ArgsNamespace(argparse.Namespace):
    title: str
    key_words: str
    page: int
    time_s: str
    time_e: str
    enable_cache: bool


def parse_arguments_for_preprocessor() -> ArgsNamespace:
    parser = argparse.ArgumentParser(description="Fetch data and Clean them.")
    parser.add_argument(
        "--title",
        type=str,
        default="Attention Heads of Large Language Models: A Survey",
        help="Input the title to generate survey.",
    )
    parser.add_argument(
        "--key_words",
        type=str,
        default="",
        help="Input the key_words to search on databases.",
    )
    parser.add_argument(
        "--page",
        type=str,
        default="5",
        help="Number of pages to crawl on Google Scholar.",
    )
    parser.add_argument(
        "--time_s",
        type=str,
        default="2017",
        help="Start year for filtering search results.",
    )
    parser.add_argument(
        "--time_e",
        type=str,
        default="2024",
        help="End year for filtering search results.",
    )
    parser.add_argument(
        "--enable_cache",
        type=bool,
        default=DEFAULT_DATA_FETCHER_ENABLE_CACHE,
        help="Whether import cache for preprocessing.",
    )
    return parser.parse_args()


def parse_arguments_for_integration_test() -> str:
    parser = argparse.ArgumentParser(description="Give the --task_id parameter.")
    parser.add_argument("--task_id", type=str, required=True, help="The ID of the task")
    args = parser.parse_args()
    return args.task_id


def load_api_key() -> str:
    """Load API key from EVAL/code/akey.txt (takes last line if multiple lines)."""
    akey_path = BASE_DIR.parent / "EVAL" / "code" / "akey.txt"
    if not akey_path.exists():
        logger.warning(f"API key file not found: {akey_path}, using default from config")
        return None
    
    try:
        with open(akey_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                api_key = lines[-1]  # Take last line
                logger.info(f"Loaded API key from {akey_path}")
                return api_key
    except Exception as e:
        logger.error(f"Failed to read API key file: {e}")
    
    return None


def load_model_config(mid: int) -> dict:
    """Load model configuration from EVAL/code/amodelid.json based on mid."""
    amodelid_path = BASE_DIR.parent / "EVAL" / "code" / "amodelid.json"
    if not amodelid_path.exists():
        logger.warning(f"Model config file not found: {amodelid_path}")
        return None
    
    try:
        with open(amodelid_path, 'r', encoding='utf-8') as f:
            models = json.load(f)
            for model_item in models:
                if model_item.get("id") == mid:
                    model_name = model_item.get("model_name")
                    logger.info(f"Found model config for mid={mid}: {model_name}")
                    return {"id": mid, "model_name": model_name}
            logger.warning(f"No model found for mid={mid} in {amodelid_path}")
    except Exception as e:
        logger.error(f"Failed to read model config file: {e}")
    
    return None


def parse_arguments_for_offline() -> ArgsNamespace:
    parser = argparse.ArgumentParser(description="Fetch data and Clean them.")
    parser.add_argument(
        "--title",
        type=str,
        default="Attention Heads of Large Language Models: A Survey",
        help="Input the title to generate survey.",
    )
    parser.add_argument(
        "--key_words",
        type=str,
        default="",
        help="Input the key_words to search on databases.",
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: 250 from config).",
    )
    parser.add_argument(
        "--mid",
        type=int,
        default=0,
        help="Model ID (0=default gpt-4o-mini/gpt-4o, 1+ uses amodelid.json).",
    )

    return parser.parse_args()


def create_tmp_config(title: str, key_word: str, mid: int = 0):
    tmp_config = {}
    tmp_config["title"] = title
    logger.info(f"[create_tmp_config] 步骤 1/3: 生成关键词")
    tmp_config["key_words"] = gen_keyword(title, key_word)
    logger.info(f"[create_tmp_config] 步骤 2/3: 生成主题描述")
    tmp_config["topic"] = gen_topic(title, tmp_config["key_words"])
    logger.info(f"[create_tmp_config] 步骤 3/3: 完成")
    tmp_config["mid"] = mid
    
    # Try to match title with topic.json and use corresponding id
    base_task_id = None
    topic_json_path = BASE_DIR.parent / "EVAL" / "code" / "topic.json"
    if topic_json_path.exists():
        try:
            with open(topic_json_path, 'r', encoding='utf-8') as f:
                topics = json.load(f)
            # Find matching topic
            for topic_item in topics:
                if topic_item.get("topic", "").strip() == title.strip():
                    topic_id = topic_item.get("id")
                    if topic_id is not None:
                        base_task_id = f"t{topic_id}"
                        logger.info(f"Matched topic in topic.json, using base_task_id: {base_task_id}")
                        break
        except Exception as e:
            logger.warning(f"Failed to read or parse topic.json: {e}")
    
    # If no match found, fall back to sanitized title
    if base_task_id is None:
        base_task_id = sanitize_filename(title)
        logger.info(f"No match in topic.json, using sanitized title as base_task_id: {base_task_id}")
    
    # Generate task_id based on mid
    # mid=0: t1, t2, t3...
    # mid=1, 2, 3...: t1a, t1b, t1c... (where a=1, b=2, c=3...)
    if mid == 0:
        task_id = base_task_id
    else:
        suffix_letter = chr(ord('a') + mid - 1)  # a for mid=1, b for mid=2, etc.
        task_id = f"{base_task_id}{suffix_letter}"
    
    tmp_config["task_id"] = task_id
    logger.info(f"Generated task_id: {task_id} (base: {base_task_id}, mid: {mid})")

    update_config(tmp_config, Path(OUTPUT_DIR) / task_id / "tmp_config.json")
    logger.info(f"Created tmp_config: {json.dumps(tmp_config, indent=4)}")
    return tmp_config


def gen_keyword(title: str, key_words: str) -> str:
    if len(key_words.split(",")) >= 6:
        logger.info(f"[gen_keyword] 关键词数量已足够，跳过生成")
        return key_words

    logger.info(f"[gen_keyword] 开始生成关键词, 标题: {title[:100]}...")
    logger.info(f"[gen_keyword] 使用模型: {ADVANCED_CHATAGENT_MODEL}")
    
    chat_agent = ChatAgent()
    prompt = load_prompt(
        Path(f"{BASE_DIR}/resources/LLM/prompts/preprocessor/generate_keyword.md"),
        title=title,
        key_words=key_words,
    )

    logger.info(f"[gen_keyword] 正在调用 API 生成关键词...")
    res = chat_agent.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
    logger.info(f"[gen_keyword] API 调用成功，收到响应")
    new_keywords = re.findall(r"<Answer>(.*?)</Answer>", res)[0]
    if key_words:
        final_keywords = ",".join(
            key_words.split(",") + new_keywords.split(",")[:3]
        )  # only select first 3 generated keyword, to avoid misunderstanding
    else:
        final_keywords = new_keywords
    logger.info(f"Keywords: {final_keywords}")
    return final_keywords


def gen_topic(title: str, key_word: str) -> str:
    """Generate a detail description of the keyword in one sentence.
    This description is used to provide more infos about keyword.
    """
    logger.info(f"[gen_topic] 开始生成主题描述, 标题: {title[:100]}...")
    logger.info(f"[gen_topic] 使用模型: {ADVANCED_CHATAGENT_MODEL}")
    
    chat = ChatAgent()
    prompt = load_prompt(
        Path(f"{BASE_DIR}/resources/LLM/prompts/preprocessor/generate_topic.md"),
        title=title,
        key_word=key_word,
    )
    
    logger.info(f"[gen_topic] 正在调用 API 生成主题描述...")
    topic = chat.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
    logger.info(f"[gen_topic] API 调用成功，收到响应")
    return topic


def wait_for_crawling(seconds: int):
    """Sleep system for seconds."""
    for i in range(seconds):
        print(
            f"\rWaiting for crawling... remaining {seconds - i} seconds.   ",
            end="",
            flush=True,
        )
        time.sleep(1)
    print()


def save_papers(papers: list[dict], dir_path: Path):
    for paper in papers:
        p = Path(dir_path) / f"{paper.get('_id', paper['title'])}.json"
        save_result(json.dumps(paper, indent=4), p)


def get_tmp_config(task_id: str):
    tmp_config_path = Path(OUTPUT_DIR) / task_id / "tmp_config.json"
    with open(tmp_config_path, "r", encoding="utf-8") as file:
        tmp_config = json.load(file)
    return tmp_config
