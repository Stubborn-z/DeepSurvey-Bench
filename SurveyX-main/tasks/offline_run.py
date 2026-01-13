import subprocess
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configs.config import BASE_DIR
from src.configs.logger import get_logger
from src.models.generator import (ContentGenerator, LatexGenerator,
                                  OutlinesGenerator)
from src.models.LLM import ChatAgent
from src.models.post_refine import PostRefiner
from src.modules.preprocessor.data_cleaner import DataCleaner
from src.modules.preprocessor.utils import (create_tmp_config,
                                            parse_arguments_for_offline)

logger = get_logger("tasks.offline_run")


def check_latexmk_installed():
    try:
        # Try running the latexmk command with the --version option
        _ = subprocess.run(['latexmk', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug("latexmk is installed.")
        return True
    except subprocess.CalledProcessError as e:
        logger.debug("latexmk is not installed.")
        return False
    except FileNotFoundError:
        logger.debug("latexmk is not installed.")
        return False

def offline_generate(task_id: str, max_papers: int = None):
    # ChatAgent will use the current TOKEN value from config (which is loaded dynamically)
    chat_agent = ChatAgent()
    
    # preprocess references
    dc = DataCleaner()
    dc.offline_proc(task_id=task_id, max_papers=max_papers)

    # generate outlines
    outline_generator = OutlinesGenerator(task_id)
    outline_generator.run()

    # generate survey
    content_generator = ContentGenerator(task_id=task_id)
    content_generator.run()

    # post refine
    post_refiner = PostRefiner(task_id=task_id, chat_agent=chat_agent)
    post_refiner.run()

    # generate full survey
    latex_generator = LatexGenerator(task_id=task_id)
    latex_generator.generate_full_survey()

    # compile latex
    if check_latexmk_installed():
        logger.info(f"Start compiling with latexmk.")
        latex_generator.compile_single_survey()
    else:
        logger.error(f"Compiling failed, as there is no latexmk installed in this machine.")


if __name__ == "__main__":
    args = parse_arguments_for_offline()
    
    logger.info("=" * 70)
    logger.info("程序启动 - 开始加载配置")
    logger.info("=" * 70)
    
    # Load API key and configure models based on mid
    from src.configs.config import load_api_key_and_models
    logger.info(f"正在加载 API Key 和模型配置 (mid={args.mid})...")
    load_api_key_and_models(mid=args.mid)
    logger.info("API Key 和模型配置加载完成")
    
    logger.info(f"开始创建临时配置 (title: {args.title[:100]}...)")
    tmp_config = create_tmp_config(args.title, args.key_words, mid=args.mid)
    logger.info("临时配置创建完成")

    topic = tmp_config["topic"]
    task_id = tmp_config["task_id"]
    
    offline_generate(task_id=task_id, max_papers=args.max_papers)
