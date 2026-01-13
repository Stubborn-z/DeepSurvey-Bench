import argparse
import json
from pathlib import Path

from src.configs.constants import OUTPUT_DIR
from src.modules.utils import load_file_as_string


class Base:
    def __init__(self, task_id: str | None = None):
        tmp_config = Base.load_tmp_config(task_id)

        self.task_id = task_id
        self.title = tmp_config["title"]
        self.key_words = tmp_config["key_words"]
        self.topic = tmp_config["topic"]

    @staticmethod
    def load_tmp_config(task_id: str) -> dict:
        path = Path(OUTPUT_DIR) / task_id / "tmp_config.json"
        dic = json.loads(load_file_as_string(path))

        # Required keys
        required_keys = ["title", "key_words", "topic"]
        missing_keys = [key for key in required_keys if key not in dic]
        assert not missing_keys, f"Missing required keys: {', '.join(missing_keys)}"
        
        # Optional keys with defaults
        allowed_keys = ["task_id", "title", "key_words", "topic", "mid"]
        # Set default for mid if not present (for backward compatibility)
        if "mid" not in dic:
            dic["mid"] = 0

        return dic
