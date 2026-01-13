import json
from pathlib import Path
import os

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent

# Global variables for dynamic model configuration
_CURRENT_MID = 0
_CURRENT_MODEL_NAME = None

# huggingface mirror
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Uncomment this line if you want to use a specific Hugging Face mirror
# os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache/")

# 使用 ai-gaochao.cn API（成功验证的 API）
REMOTE_URL = "https://api.ai-gaochao.cn/v1/chat/completions"
# TOKEN will be loaded dynamically from EVAL/code/akey.txt if available
TOKEN = "sk-PCMll7EPT1wdm6thBd3e860d8d174a50B1484fAc007c6574"  # Default fallback
DEFAULT_CHATAGENT_MODEL = "gpt-4o-mini"
# 用于与文献分析互动的模型（处理那200个文献）
# Will be set dynamically based on mid parameter
PAPER_ANALYSIS_MODEL = "gpt-4o-mini"
# 用于生成大纲、正文等高质量内容生成的模型
# Will be set dynamically based on mid parameter
ADVANCED_CHATAGENT_MODEL = "gpt-4o"

# 代理配置（用于绕过地区限制，如果 API Key 有效但受地区限制）
# 如果不需要代理，设置为 None 或空字符串
HTTP_PROXY = None # 例如: "http://proxy.example.com:8080"
HTTPS_PROXY = None# 例如: "http://proxy.example.com:8080"

LOCAL_URL = "LOCAL_URL"
LOCAL_LLM = "LOCAL_LLM"
DEFAULT_EMBED_LOCAL_MODEL = "DEFAULT_EMBED_LOCAL_MODEL"

## for embedding model
DEFAULT_EMBED_ONLINE_MODEL = "BAAI/bge-base-en-v1.5"
# 使用 ai-gaochao.cn API（与 Chat API 使用同一个服务）
EMBED_REMOTE_URL = "https://api.ai-gaochao.cn/v1/embeddings"
EMBED_TOKEN = "sk-PCMll7EPT1wdm6thBd3e860d8d174a50B1484fAc007c6574"
SPLITTER_WINDOW_SIZE = 6
SPLITTER_CHUNK_SIZE = 2048

## for preprocessing
CRAWLER_BASE_URL = ""
CRAWLER_GOOGLE_SCHOLAR_SEND_TASK_URL = ""
DEFAULT_DATA_FETCHER_ENABLE_CACHE = True
CUT_WORD_LENGTH = 10
MD_TEXT_LENGTH = 20000
ARXIV_PROJECTION = (
    "_id, title, authors, detail_url, abstract, md_text, reference, detail_id, image"
)

## Iteration and paper pool limits
DEFAULT_ITERATION_LIMIT = 3
DEFAULT_PAPER_POOL_LIMIT = 1024
# Maximum number of papers to process from ref.jsonl (filters and selects top papers)
MAX_PAPERS_TO_PROCESS = 200

## llamaindex OpenAI
DEFAULT_LLAMAINDEX_OPENAI_MODEL = "gpt-4o"
# DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
CHAT_AGENT_WORKERS = 4

## Rate limiting (请求速率限制)
# 每次API请求之间的最小间隔（秒），用于避免突发大量请求导致API限流
# 建议值：0.1-0.5秒，如果频繁遇到连接重置错误，可以增加到0.5-1.0秒
API_REQUEST_MIN_INTERVAL = 0.2  # 200ms


def load_api_key_and_models(mid: int = 0):
    """
    Load API key from EVAL/code/akey.txt and configure models based on mid.
    This function modifies the module-level variables TOKEN, EMBED_TOKEN, PAPER_ANALYSIS_MODEL, and ADVANCED_CHATAGENT_MODEL.
    
    New logic (regardless of mid):
    - PAPER_ANALYSIS_MODEL: Always "gpt-4o-mini" (for paper analysis interactions)
    - ADVANCED_CHATAGENT_MODEL: Load from amodelid.json based on mid (for outline, content generation, etc.)
    
    Args:
        mid: Model ID (0, 1, 2, ... uses amodelid.json)
    
    Returns:
        tuple: (api_key, paper_analysis_model, advanced_model)
    """
    global TOKEN, EMBED_TOKEN, PAPER_ANALYSIS_MODEL, ADVANCED_CHATAGENT_MODEL, _CURRENT_MID, _CURRENT_MODEL_NAME
    
    # Load API key
    akey_path = BASE_DIR.parent / "EVAL" / "code" / "akey.txt"
    if akey_path.exists():
        try:
            with open(akey_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if lines:
                    new_token = lines[-1]  # Take last line
                    TOKEN = new_token
                    EMBED_TOKEN = new_token  # Also update EMBED_TOKEN
                    import logging
                    logging.getLogger("src.configs.config").info(f"Loaded API key from {akey_path}")
        except Exception as e:
            import logging
            logging.getLogger("src.configs.config").warning(f"Failed to load API key: {e}")
    
    # Paper analysis model: Always use gpt-4o-mini regardless of mid
    PAPER_ANALYSIS_MODEL = "gpt-4o-mini"
    
    # Advanced model: Load from amodelid.json based on mid
    amodelid_path = BASE_DIR.parent / "EVAL" / "code" / "amodelid.json"
    if amodelid_path.exists():
        try:
            with open(amodelid_path, 'r', encoding='utf-8') as f:
                models = json.load(f)
                model_found = False
                for model_item in models:
                    if model_item.get("id") == mid:
                        model_name = model_item.get("model_name")
                        ADVANCED_CHATAGENT_MODEL = model_name
                        _CURRENT_MODEL_NAME = model_name
                        model_found = True
                        import logging
                        logging.getLogger("src.configs.config").info(
                            f"Loaded model config: mid={mid}, "
                            f"PAPER_ANALYSIS_MODEL={PAPER_ANALYSIS_MODEL}, "
                            f"ADVANCED_CHATAGENT_MODEL={ADVANCED_CHATAGENT_MODEL}"
                        )
                        break
                if not model_found:
                    import logging
                    logging.getLogger("src.configs.config").warning(
                        f"No model found for mid={mid} in amodelid.json, using default gpt-4o"
                    )
                    ADVANCED_CHATAGENT_MODEL = "gpt-4o"
                    _CURRENT_MODEL_NAME = None
        except Exception as e:
            import logging
            logging.getLogger("src.configs.config").error(f"Failed to load model config: {e}")
            ADVANCED_CHATAGENT_MODEL = "gpt-4o"
            _CURRENT_MODEL_NAME = None
    else:
        import logging
        logging.getLogger("src.configs.config").warning(
            f"Model config file not found: {amodelid_path}, using default gpt-4o"
        )
        ADVANCED_CHATAGENT_MODEL = "gpt-4o"
        _CURRENT_MODEL_NAME = None
    
    _CURRENT_MID = mid
    return TOKEN, PAPER_ANALYSIS_MODEL, ADVANCED_CHATAGENT_MODEL

## survey generation
COARSE_GRAINED_TOPK = 200
MIN_FILTERED_LIMIT = 150
NUM_PROCESS_LIMIT = 10

## fig retrieving
FIG_RETRIEVE_URL = ""
ENHANCED_FIG_RETRIEVE_URL = ""
FIG_CHUNK_SIZE = 8192
MATCH_TOPK = 3
FIG_RETRIEVE_Authorization = ""
FIG_RETRIEVE_TOKEN = ""
