import json
import os
import re
from pathlib import Path
from typing import Union

from tqdm import tqdm

from src.configs.config import BASE_DIR, CHAT_AGENT_WORKERS, MD_TEXT_LENGTH, MAX_PAPERS_TO_PROCESS, PAPER_ANALYSIS_MODEL
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.models.LLM.utils import cut_text_by_token, load_prompt
from src.models.monitor.time_monitor import TimeMonitor
from src.modules.utils import (
    clean_chat_agent_format,
    load_file_as_string,
    sanitize_filename,
    save_result,
)

logger = get_logger("src.modules.preprocessor.DataCleaner")


class DataCleaner:
    def __init__(self, papers: list[dict] = []):
        self.papers: list[dict] = papers
        self.chat_agent_workers = CHAT_AGENT_WORKERS

    def _select_top_papers(self, papers: list[dict], max_count: int) -> list[dict]:
        """Select top papers based on quality metrics.
        
        Scoring criteria:
        1. Abstract length (longer abstracts are often more informative)
        2. Has arxivid (prioritize papers with arxiv ID)
        3. Title quality (non-empty, reasonable length)
        
        Args:
            papers: List of paper dictionaries
            max_count: Maximum number of papers to select
            
        Returns:
            Top selected papers
        """
        if len(papers) <= max_count:
            return papers
        
        # Score each paper
        scored_papers = []
        for paper in papers:
            score = 0
            
            # Abstract length score (normalized, max 100 points)
            abstract_len = len(paper.get("abstract", ""))
            if abstract_len > 0:
                # Give higher score for abstracts between 200-2000 chars
                if 200 <= abstract_len <= 2000:
                    score += 100
                elif abstract_len < 200:
                    score += (abstract_len / 200) * 80  # Up to 80 points for short abstracts
                else:
                    score += 100 - min((abstract_len - 2000) / 100, 50)  # Slight penalty for very long
            
            # Arxiv ID presence (50 points)
            if paper.get("arxivid", "").strip():
                score += 50
            
            # Title quality (20 points)
            title = paper.get("title", "").strip()
            if title and 10 <= len(title) <= 200:
                score += 20
            elif title:
                score += 10
            
            scored_papers.append((score, paper))
        
        # Sort by score (descending) and take top N
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        selected = [paper for _, paper in scored_papers[:max_count]]
        
        logger.info(f"Paper selection: top score={scored_papers[0][0]:.1f}, bottom score={scored_papers[max_count-1][0]:.1f}")
        
        return selected

    def load_json_dir(self, json_path_dir: Path):
        """load papers from json directory."""
        papers = []
        cnt_total = 0
        for file in os.listdir(json_path_dir):
            if file.endswith(".json"):
                p = os.path.join(json_path_dir, file)
                dic = json.loads(load_file_as_string(p))
                if (
                    "md_text" in dic
                ):  # Only consider those with `md_text` as available papers.
                    papers.append(dic)
                cnt_total += 1
        logger.info(f"Find {len(papers)} out of {cnt_total} papers available.")
        self.papers = papers

    def complete_title(self):
        for paper in tqdm(self.papers, desc="completing title..."):
            if "title" not in paper or not paper["title"]:
                # Fallback: try to extract from md_text if title is missing
                if "md_text" in paper:
                    paper["title"] = paper["md_text"].splitlines()[0].strip(" #")
                    paper["title"] = paper["title"][:32]  # avoid too long title
                else:
                    paper["title"] = "Unknown"

    def complete_abstract(self):
        # Abstract should already be set from jsonl, but provide fallback
        pattern = r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*"  # find "abstract" substring, with whitespace bettween letters.
        for paper in tqdm(self.papers, desc="completing abstract..."):
            # Ensure abstract field exists and is a string
            if "abstract" not in paper:
                paper["abstract"] = ""
            abstract = paper.get("abstract", "") or ""
            
            # If abstract is already long enough, skip
            if abstract and len(abstract) > 500:
                continue
            
            # If abstract is missing or empty, try to extract from md_text
            if not abstract:
                if "md_text" in paper and paper["md_text"]:
                    match = re.search(pattern, paper["md_text"], re.IGNORECASE)
                    if match:
                        index = match.start()
                        paper["abstract"] = paper["md_text"][index : index + 2000]
                    else:
                        paper["abstract"] = paper["md_text"][:2000]
                else:
                    # Ensure abstract is at least an empty string
                    paper["abstract"] = ""

    def complete_bib(self, bib_file_save_path: str):
        """Not only complete the bib_name, also need to save all bibnames into a references.bib file"""
        var_name_i = 0
        bib_all = []
        remove_non_ascii_chars = (
            lambda input_string: input_string.replace(",", "")
            .encode("ascii", "ignore")
            .decode("ascii")
        )

        for paper in tqdm(self.papers, desc="completing bibname..."):
            if "reference" in paper:
                bib_name = paper["reference"].splitlines()[0].split("{")[1].strip(",")
                new_bib_name = remove_non_ascii_chars(bib_name)

                paper["bib_name"] = new_bib_name
                paper["reference"] = paper["reference"].replace(bib_name, new_bib_name)
            else:
                title = remove_non_ascii_chars(paper["title"])
                bib_name = "".join([c for c in title if not c.isspace()][:10]) + str(
                    var_name_i
                )
                var_name_i += 1
                bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"

                paper["reference"] = bib_tex
                paper["bib_name"] = bib_name

            bib_all.append(paper["reference"])

        save_result("\n".join(bib_all), bib_file_save_path)

    def check_md_text_length(self):
        for paper in self.papers:
            if "md_text" not in paper:
                continue
            md_text = paper["md_text"]
            paper["md_text"] = cut_text_by_token(md_text, MD_TEXT_LENGTH)

    def __process_paper_type_response(self, res: str, paper_index: int):
        kinds = ["method", "benchmark", "theory", "survey"]
        for k in kinds:
            if k in res.lower():
                self.papers[paper_index]["paper_type"] = k
                return True
        logger.error(
            f"failed to extract papertype of {self.papers[paper_index]['title']}"
        )
        logger.error(f"The response from gpt is {res}")
        return False

    def get_paper_type(self, chat_agent: ChatAgent):
        """complete the paper type field with chatgpt."""
        # load prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            # Ensure abstract exists and is a string
            abstract = paper.get("abstract", "") or ""
            # Skip papers with empty abstract, set default paper_type
            if not abstract or len(abstract.strip()) < 10:
                logger.warning(f"Paper '{paper.get('title', 'unknown')}' has empty or very short abstract, setting default paper_type to 'method'")
                paper["paper_type"] = "method"
                continue
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/preprocessor/paper_type_classification.md",
                abstract=abstract,
            )
            prompts_and_index.append([prompt, i])
        # batch_chat
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(prompts, desc="getting paper type...", model=PAPER_ANALYSIS_MODEL)
            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_paper_type_response(res, paper_index)
            ]
            cnt += 1
        
        # Set default paper_type for any papers that still don't have one
        for paper in self.papers:
            if "paper_type" not in paper:
                logger.warning(f"Paper '{paper.get('title', 'unknown')}' still has no paper_type after classification attempts, setting default to 'method'")
                paper["paper_type"] = "method"

    def __process_attri_response(self, res: str, paper_index: int):
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            self.papers[paper_index]["attri"] = {**res_dic}
            return True
        except Exception as e:
            logger.debug(
                f"Failed to process {self.papers[paper_index]['title']}; The res: {res[:100]}; {e}"
            )
            return False

    def get_attri(self, chat_agent: ChatAgent):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            # 根据 paper_type 加载对应的 prompt
            # Use default "method" if paper_type is missing
            paper_type = paper.get("paper_type", "method").lower()
            if "md_text" not in paper:
                logger.warning(f"Paper '{paper.get('title', 'unknown')}' has no md_text, skipping attribute extraction")
                continue
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/preprocessor/attri_tree_for_{paper_type}.md",
                paper=paper["md_text"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......", model=PAPER_ANALYSIS_MODEL
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def save_papers(
        self, save_dir: Union[str, Path], file_name_attr: str = "title"
    ) -> None:
        """save every cleaned paper."""
        filter_field = [
            "from",
            "scholar_id",
            "detail_id",
            "title",
            "abstract",
            "arxivid",  # Add arxivid to saved fields
            "bib_name",
            "md_text",
            "paper_type",
            "attri",
            "mount_outline",
            "similarity_score",
            "image",
        ]
        for paper in self.papers:
            try:
                file_name = paper[file_name_attr] + ".json"
                file_name = sanitize_filename(file_name)
                file_path = os.path.join(save_dir, file_name)
                save_dic = {key: paper.get(key, None) for key in filter_field}
                save_result(json.dumps(save_dic, indent=4), file_path)
            except Exception as e:
                logger.error(
                    f"There is an error when saving {file_path}. The error is: {e}"
                )
        return self.papers

    def quick_check(self) -> list[dict]:
        """Used in PaperRecaller for quick check"""
        papers_with_md = [paper for paper in self.papers if "md_text" in paper]
        self.papers = papers_with_md
        self.complete_title()
        self.complete_abstract()
        return self.papers

    def offline_proc(self, task_id: str, max_papers: int = None) -> None:
        # Load papers from EVAL/t{id}/ref.jsonl
        # Extract base task_id (e.g., t1a -> t1, t1b -> t1) since ref.jsonl is stored in base directory
        base_task_id = re.sub(r'^(t\d+)[a-z]?$', r'\1', task_id)
        # Use config default if max_papers not specified
        if max_papers is None:
            max_papers = MAX_PAPERS_TO_PROCESS
        
        ref_jsonl_path = BASE_DIR.parent / "EVAL" / base_task_id / "ref.jsonl"
        if not ref_jsonl_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_jsonl_path}")
        
        # Read jsonl file - each line is a JSON object
        papers = []
        with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper_data = json.loads(line)
                    # Extract title and abstract from jsonl
                    # Ensure all fields are strings (not None) to handle empty values properly
                    title = paper_data.get("title", "") or ""
                    abstract = paper_data.get("abstract", "") or ""
                    arxivid = paper_data.get("arxivid", "") or ""
                    
                    # Filter: only keep papers that have both abstract and arxivid
                    if not abstract or not arxivid:
                        continue
                    
                    paper = {
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "arxivid": arxivid.strip(),
                    }
                    # Create md_text from title and abstract for compatibility
                    md_text_parts = []
                    if paper["title"]:
                        md_text_parts.append(f"# {paper['title']}")
                    if paper["abstract"]:
                        md_text_parts.append("\n\nabstract\n\n" + paper["abstract"])
                    # Ensure md_text is not empty (at least have title)
                    paper["md_text"] = "\n".join(md_text_parts) if md_text_parts else f"# {paper['title']}" if paper['title'] else "# Unknown Paper"
                    papers.append(paper)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {ref_jsonl_path}: {e}")
        
        initial_count = len(papers)
        logger.info(f"Loaded {initial_count} papers with both abstract and arxivid from {ref_jsonl_path}")
        
        # Select top papers if exceeding limit
        if len(papers) > max_papers:
            papers = self._select_top_papers(papers, max_papers)
            logger.info(f"Selected top {len(papers)} papers from {initial_count} total")
        
        self.papers = papers

        # Title and abstract are already extracted from jsonl, no need to call complete_title/abstract
        # But we keep them for compatibility (they should be no-ops now)
        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        self.complete_bib(bib_file_path)

        self.check_md_text_length()
        chat_agent = ChatAgent()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

    def run(self, task_id: str, chat_agent: ChatAgent = None):
        time_monitor = TimeMonitor(task_id)
        time_monitor.start("clean paper")

        self.load_json_dir(Path(OUTPUT_DIR) / task_id / "jsons")
        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        self.complete_bib(bib_file_path)

        self.check_md_text_length()
        if chat_agent is None:
            chat_agent = ChatAgent()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

        time_monitor.end("clean paper")


# python -m src.modules.preprocessor.data_cleaner
if __name__ == "__main__":
    dc = DataCleaner()
    dc.offline_proc("ref1")
    print(len(dc.papers))
