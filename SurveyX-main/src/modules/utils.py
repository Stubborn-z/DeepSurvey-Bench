import os
import json
import logging
import re
import ast
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Union, Dict

from src.configs.logger import get_logger

logger = get_logger("src.modules.utils")


def shut_loggers():
    for logger in logging.Logger.manager.loggerDict:
        logging.getLogger(logger).setLevel(logging.INFO)


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\\/:"*?<>|]', "_", filename)


def save_result(result: str, path: Union[str, Path]) -> None:
    """save a string to a file, if the prefix dir doesn't exit, create them.

    Args:
        result (str): string waiting to be saved.
        path (str): where to save this string.
    """
    if isinstance(path, str):
        path = Path(path)
    directory = path.parent
    # 如果目录不存在，则创建目录
    if not directory.exists():
        directory.mkdir(exist_ok=True, parents=True)
    # 写入文件
    with path.open("w", encoding="utf-8") as fw:
        fw.write(result)


def load_file_as_string(path: Union[str, Path]) -> str:
    if isinstance(path, str):
        with open(path, "r", encoding="utf-8") as fr:
            return fr.read()
    elif isinstance(path, Path):
        with path.open("r", encoding="utf-8") as fr:
            return fr.read()
    else:
        raise ValueError(path)


def update_config(dic: dict, config_path: str):
    """update the config file

    Args:
        dic (dict): new config dict.
    """
    config_path = Path(config_path)
    if config_path.exists():
        config: dict = json.load(open(config_path, "r", encoding="utf-8"))
        config.update(dic)
    else:
        config: dict = dic
    save_result(json.dumps(config, indent=4), config_path)


def save_as_json(result: dict, path: str):
    """
    Save the result as a JSON file.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)


def load_meta_data(dir_path):
    """
    Load all JSON files in the directory.
    """
    data = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                result = json.load(file)  # 将 JSON 文件内容读取为 Python 列表
            data.append(result)
    return data


def load_single_file(file_path):
    """
    Load a single JSON file based on its path.
    """
    # 判断文件路径是否存在
    if not os.path.exists(file_path):
        return ""

    # 如果路径存在，打开并读取文件
    with open(file_path, "r") as file:
        article = json.load(file)
    return article


def load_prompt(filename: str, **kwargs) -> str:
    """
    读取prompt模板
    """
    path = os.path.join("", filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().format(**kwargs)
    else:
        logger.error(f"Prompt template not found at {path}")
        return ""


Clean_patten = re.compile(pattern=r"```(json|latex)?", flags=re.DOTALL)


def clean_chat_agent_format(content: str):
    content = re.sub(Clean_patten, "", content)
    return content


def load_papers(paper_dir_path_or_papers: Union[Path, List[Dict]]) -> list[dict]:
    if isinstance(paper_dir_path_or_papers, Path):
        papers = []
        for file in os.listdir(paper_dir_path_or_papers):
            file_path = paper_dir_path_or_papers / file
            if file_path.is_dir():
                file_path = file_path / os.listdir(file_path)[0]
            if not file_path.is_file():
                logger.error(f"loading paper error: {file_path} is not a file.")
                continue
            paper = json.loads(load_file_as_string(file_path))
            papers.append(paper)
        return papers
    elif isinstance(paper_dir_path_or_papers, list):
        return paper_dir_path_or_papers
    else:
        raise ValueError()


def load_file_as_text(file_path: Path):
    with file_path.open("r", encoding="utf-8") as fr:
        return fr.read()


# ========== JSON Output Generator Functions ==========

def extract_citations_from_latex(latex_content: str) -> Tuple[Dict[str, int], List[Tuple[int, int, str]]]:
    """
    Extract citation keys from LaTeX and build mapping from key to citation number.
    Returns:
        - key_to_number: Dictionary mapping citation key to its number (1-indexed)
        - citation_positions: List of (start_pos, end_pos, citation_numbers_str) tuples in order of appearance
    """
    # Extract content between \begin{document} and \bibliography
    doc_match = re.search(r'\\begin\{document\}(.*?)\\bibliography', latex_content, re.DOTALL)
    if doc_match:
        main_content = doc_match.group(1)
        doc_start_offset = doc_match.start(1)
    else:
        main_content = latex_content
        doc_start_offset = 0
    
    # Find all \cite{} commands with their positions (relative to main_content)
    cite_pattern = r'\\cite\{([^}]+)\}'
    citations = []
    key_to_number = {}
    citation_number = 1
    
    for match in re.finditer(cite_pattern, main_content):
        cite_keys_str = match.group(1)
        # Handle multiple keys: \cite{key1,key2,key3}
        cite_keys = [k.strip() for k in cite_keys_str.split(',')]
        
        citation_numbers = []
        for key in cite_keys:
            if key not in key_to_number:
                key_to_number[key] = citation_number
                citation_number += 1
            citation_numbers.append(str(key_to_number[key]))
        
        # Store positions relative to main_content, and citation numbers string
        citations.append((match.start(), match.end(), ','.join(citation_numbers)))
    
    return key_to_number, citations


def extract_text_from_latex(latex_content: str, include_citations: bool = True) -> str:
    """
    Extract plain text from LaTeX content, removing LaTeX commands.
    If include_citations is True, replace \cite{} with [number] citations.
    
    Returns a string with extracted text.
    
    Strategy: Extract content between \begin{document} and \bibliography, 
    focusing on the main body text.
    """
    # Extract content between \begin{document} and \bibliography
    doc_match = re.search(r'\\begin\{document\}(.*?)\\bibliography', latex_content, re.DOTALL)
    if doc_match:
        main_content = doc_match.group(1)
    else:
        # Fallback: use entire content
        main_content = latex_content
    
    # Remove LaTeX comments
    main_content = re.sub(r'%.*?$', '', main_content, flags=re.MULTILINE)
    
    # Extract abstract if exists
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main_content, re.DOTALL)
    abstract_text = ""
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        main_content = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', main_content, flags=re.DOTALL)
    
    # Replace citations with [number] format if requested
    if include_citations:
        # Extract citations from the same main_content we're working with
        cite_pattern = r'\\cite\{([^}]+)\}'
        key_to_number = {}
        citation_number = 1
        citation_replacements = []  # List of (start, end, replacement) tuples
        
        for match in re.finditer(cite_pattern, main_content):
            cite_keys_str = match.group(1)
            # Handle multiple keys: \cite{key1,key2,key3}
            cite_keys = [k.strip() for k in cite_keys_str.split(',')]
            
            citation_numbers = []
            for key in cite_keys:
                if key not in key_to_number:
                    key_to_number[key] = citation_number
                    citation_number += 1
                citation_numbers.append(str(key_to_number[key]))
            
            # Format: [1,2,3] for multiple citations
            citation_str = f"[{','.join(citation_numbers)}]"
            citation_replacements.append((match.start(), match.end(), citation_str))
        
        # Replace citations in reverse order to preserve positions
        for start, end, replacement in reversed(citation_replacements):
            main_content = main_content[:start] + replacement + main_content[end:]
    else:
        # Remove citation commands
        main_content = re.sub(r'\\cite\{[^}]+\}', '', main_content)
    
    # Remove title/author commands (already processed)
    main_content = re.sub(r'\\maketitle', '', main_content)
    main_content = re.sub(r'\\title\{[^}]*\}', '', main_content)
    main_content = re.sub(r'\\author\{[^}]*\}', '', main_content)
    
    # Extract section titles and content
    # Replace section commands with just the title
    main_content = re.sub(r'\\section\{([^}]+)\}', r'\n\n\1\n\n', main_content)
    main_content = re.sub(r'\\subsection\{([^}]+)\}', r'\n\n\1\n\n', main_content)
    
    # Remove labels
    main_content = re.sub(r'\\label\{[^}]+\}', '', main_content)
    
    # Remove input commands (figures, etc.)
    main_content = re.sub(r'\\input\{[^}]+\}', '', main_content)
    
    # Remove simple formatting commands, keeping content
    main_content = re.sub(r'\\textbf\{([^}]+)\}', r'\1', main_content)
    main_content = re.sub(r'\\textit\{([^}]+)\}', r'\1', main_content)
    main_content = re.sub(r'\\emph\{([^}]+)\}', r'\1', main_content)
    main_content = re.sub(r'\\href\{[^}]+\}\{([^}]+)\}', r'\1', main_content)
    main_content = re.sub(r'\\textcolor\{[^}]+\}\{([^}]+)\}', r'\1', main_content)
    main_content = re.sub(r'\\underline\{([^}]+)\}', r'\1', main_content)
    
    # Remove remaining common LaTeX commands (but be careful with nested braces)
    # Remove commands like \newpage, \vfill, etc.
    main_content = re.sub(r'\\newpage', '\n\n', main_content)
    main_content = re.sub(r'\\vfill', '', main_content)
    main_content = re.sub(r'\\clearpage', '\n\n', main_content)
    
    # Remove remaining LaTeX commands (more aggressive but careful)
    # Pattern: \command[optional]{required} or \command{required}
    # Do this iteratively to handle nested braces
    max_iterations = 10
    for _ in range(max_iterations):
        old_content = main_content
        # Remove commands with optional and required arguments (but preserve citation brackets we just added)
        # Skip citation brackets when removing commands
        main_content = re.sub(r'\\[a-zA-Z@]+(\[[^\]]*\])?\{[^}]*\}', '', main_content)
        if old_content == main_content:
            break
    
    # Remove standalone command names (like \item, \par, etc.)
    main_content = re.sub(r'\\[a-zA-Z@]+\*?\b', '', main_content)
    
    # Clean up whitespace (but preserve citation brackets)
    main_content = re.sub(r'\s+', ' ', main_content)
    main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
    
    # Split into paragraphs
    paragraphs = []
    if abstract_text:
        # Clean abstract
        abstract_text = re.sub(r'\\[a-zA-Z@]+(\[[^\]]*\])?\{[^}]*\}', '', abstract_text)
        abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()
        if abstract_text and len(abstract_text) > 10:
            paragraphs.append(abstract_text)
    
    # Add main content paragraphs
    for para in main_content.split('\n\n'):
        para = para.strip()
        if para and len(para) > 10:
            paragraphs.append(para)
    
    # Join all paragraphs into a single string
    return '\n\n'.join(paragraphs)


def parse_ref_jsonl(task_id: str) -> Dict[str, str]:
    """
    Parse ref.jsonl file and extract references.
    Returns a dictionary mapping reference keys to arxivid (or title if arxivid is empty).
    The format matches the template: {"1": "arxiv_id", "2": "arxiv_id", ...}
    
    Args:
        task_id: Task ID (e.g., "t1", "t1a", "t1b", "t2")
    """
    from src.configs.constants import BASE_DIR
    
    # Extract base task_id (e.g., t1a -> t1, t1b -> t1) since ref.jsonl is stored in base directory
    base_task_id = re.sub(r'^(t\d+)[a-z]?$', r'\1', task_id)
    ref_jsonl_path = BASE_DIR.parent / "EVAL" / base_task_id / "ref.jsonl"
    if not ref_jsonl_path.exists():
        logger.warning(f"Reference jsonl file not found: {ref_jsonl_path}")
        return {}
    
    references = {}
    try:
        with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    paper_data = json.loads(line)
                    # Use arxivid if available, otherwise use title
                    arxivid = paper_data.get("arxivid", "").strip()
                    if arxivid:
                        references[str(idx)] = arxivid
                    else:
                        title = paper_data.get("title", "").strip()
                        if title:
                            references[str(idx)] = title
                        else:
                            references[str(idx)] = f"reference_{idx}"
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {idx} in {ref_jsonl_path}: {e}")
        
    except Exception as e:
        logger.error(f"Error parsing jsonl file {ref_jsonl_path}: {e}")
        return {}
    
    return references


def parse_bib_file(bib_path: Path) -> Dict[str, str]:
    """
    Parse .bib file and extract references.
    Returns a dictionary mapping reference keys to arxiv IDs or titles.
    The format matches the template: {"1": "arxiv_id", "2": "arxiv_id", ...}
    """
    if not bib_path.exists():
        logger.warning(f"Bibliography file not found: {bib_path}")
        return {}
    
    references = {}
    try:
        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_content = f.read()
        
        # Find all BibTeX entries: @article{key, ...}
        # Pattern to match entries and their content
        entry_pattern = r'@\w+\{([^,]+),.*?\}'
        entries = list(re.finditer(entry_pattern, bib_content, re.DOTALL))
        
        # Process each entry
        for idx, entry_match in enumerate(entries, start=1):
            key = entry_match.group(1).strip()
            entry_text = entry_match.group(0)
            
            # Try to extract arxiv ID first
            arxiv_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5}v?\d*)', entry_text, re.IGNORECASE)
            if arxiv_match:
                references[str(idx)] = arxiv_match.group(1)
            else:
                # Try to extract title
                title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry_text, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    # Remove LaTeX formatting from title
                    title = re.sub(r'\\[a-zA-Z@]+(\[[^\]]*\])?\{[^}]*\}', '', title)
                    title = re.sub(r'\{|\}', '', title)
                    references[str(idx)] = title
                else:
                    # Use key as fallback
                    references[str(idx)] = key
        
        # If no entries found with the pattern, try simpler extraction
        if not references:
            for idx, match in enumerate(re.finditer(r'@\w+\{([^,]+)', bib_content), start=1):
                key = match.group(1).strip()
                references[str(idx)] = key
        
    except Exception as e:
        logger.error(f"Error parsing bib file {bib_path}: {e}")
        return {}
    
    return references


def parse_bib_file_by_key(bib_path: Path, task_id: str = None) -> Dict[str, Dict[str, str]]:
    """
    Parse .bib file and return a dictionary mapping citation key to its information.
    If task_id is provided, also try to get arxivid from ref.jsonl or papers directory.
    Returns: {citation_key: {"title": "...", "arxivid": "..." (if available)}}
    """
    if not bib_path.exists():
        logger.warning(f"Bibliography file not found: {bib_path}")
        return {}
    
    references = {}
    
    # Get arxivid mapping from ref.jsonl if task_id is provided
    arxivid_mapping = {}
    if task_id:
        arxivid_mapping = get_arxivid_mapping_from_ref_jsonl(task_id)
        
        # Also try to get from papers directory (bib_name -> arxivid mapping)
        from src.configs.constants import OUTPUT_DIR
        papers_dir = Path(OUTPUT_DIR) / task_id / "papers"
        if papers_dir.exists():
            try:
                for filename in os.listdir(papers_dir):
                    if not filename.endswith('.json'):
                        continue
                    paper_path = papers_dir / filename
                    try:
                        with open(paper_path, 'r', encoding='utf-8') as f:
                            paper_data = json.load(f)
                            bib_name = paper_data.get("bib_name", "").strip()
                            title = paper_data.get("title", "").strip()
                            arxivid = paper_data.get("arxivid", "").strip()
                            
                            # If we have bib_name and title, create mapping
                            if bib_name and title:
                                if arxivid:
                                    arxivid_mapping[bib_name] = arxivid  # bib_name -> arxivid
                                elif title in arxivid_mapping:
                                    arxivid_mapping[bib_name] = arxivid_mapping[title]  # bib_name -> arxivid (via title)
                    except Exception:
                        continue
            except Exception:
                pass
    
    try:
        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_content = f.read()
        
        # Find all BibTeX entries: @article{key, ...}
        # Use a better pattern to handle multi-line entries
        entry_pattern = r'@\w+\{([^,]+),(.*?)(?=@|\Z)'
        entries = list(re.finditer(entry_pattern, bib_content, re.DOTALL))
        
        for entry_match in entries:
            key = entry_match.group(1).strip()
            entry_text = entry_match.group(2)
            
            ref_info = {}
            
            # Try to extract arxiv ID from bib entry
            arxiv_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5}v?\d*)', entry_text, re.IGNORECASE)
            if arxiv_match:
                ref_info["arxivid"] = arxiv_match.group(1)
            
            # Extract title
            title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry_text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                # Remove LaTeX formatting from title
                title = re.sub(r'\\[a-zA-Z@]+(\[[^\]]*\])?\{[^}]*\}', '', title)
                title = re.sub(r'\{|\}', '', title)
                ref_info["title"] = title
            else:
                ref_info["title"] = key  # Use key as fallback
            
            # Try to get arxivid from mapping (by bib_name/key or by title)
            if "arxivid" not in ref_info or not ref_info["arxivid"]:
                if key in arxivid_mapping:
                    ref_info["arxivid"] = arxivid_mapping[key]
                elif ref_info.get("title") and ref_info["title"] in arxivid_mapping:
                    ref_info["arxivid"] = arxivid_mapping[ref_info["title"]]
            
            references[key] = ref_info
        
    except Exception as e:
        logger.error(f"Error parsing bib file {bib_path}: {e}")
        return {}
    
    return references


def get_arxivid_mapping_from_ref_jsonl(task_id: str) -> Dict[str, str]:
    """
    Build a mapping from title to arxivid from ref.jsonl file.
    Returns: {title: arxivid}
    
    Args:
        task_id: Task ID (e.g., "t1", "t1a", "t1b", "t2")
    """
    from src.configs.constants import BASE_DIR
    
    # Extract base task_id (e.g., t1a -> t1, t1b -> t1) since ref.jsonl is stored in base directory
    base_task_id = re.sub(r'^(t\d+)[a-z]?$', r'\1', task_id)
    ref_jsonl_path = BASE_DIR.parent / "EVAL" / base_task_id / "ref.jsonl"
    if not ref_jsonl_path.exists():
        return {}
    
    mapping = {}
    try:
        with open(ref_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper_data = json.loads(line)
                    title = paper_data.get("title", "").strip()
                    arxivid = paper_data.get("arxivid", "").strip()
                    if title and arxivid:
                        mapping[title] = arxivid
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error reading ref.jsonl for arxivid mapping: {e}")
    
    return mapping


def parse_selected_papers(task_id: str) -> Dict[str, str]:
    """
    Parse selected papers from outputs/{task_id}/papers/ directory.
    Returns a dictionary mapping reference keys to arxivid (or title if arxivid is empty).
    The format matches the template: {"1": "arxiv_id_or_title", "2": "arxiv_id_or_title", ...}
    
    Args:
        task_id: Task ID (e.g., "t1", "t2")
    """
    from src.configs.constants import OUTPUT_DIR
    
    papers_dir = Path(OUTPUT_DIR) / task_id / "papers"
    if not papers_dir.exists():
        logger.warning(f"Selected papers directory not found: {papers_dir}")
        return {}
    
    # Get arxivid mapping from ref.jsonl (title -> arxivid)
    arxivid_mapping = get_arxivid_mapping_from_ref_jsonl(task_id)
    
    chooseref = {}
    try:
        # Get all JSON files in papers directory, sort by filename for consistent ordering
        paper_files = sorted([f for f in os.listdir(papers_dir) if f.endswith('.json')])
        
        for idx, filename in enumerate(paper_files, start=1):
            paper_path = papers_dir / filename
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                    
                    # Try to get arxivid: first from paper_data, then from ref.jsonl mapping
                    arxivid = paper_data.get("arxivid", "").strip()
                    if not arxivid:
                        title = paper_data.get("title", "").strip()
                        if title and title in arxivid_mapping:
                            arxivid = arxivid_mapping[title]
                    
                    # Use arxivid if available, otherwise use title
                    if arxivid:
                        chooseref[str(idx)] = arxivid
                    else:
                        title = paper_data.get("title", "").strip()
                        if title:
                            chooseref[str(idx)] = title
                        else:
                            chooseref[str(idx)] = f"paper_{idx}"
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to parse paper file {paper_path}: {e}")
        
    except Exception as e:
        logger.error(f"Error parsing selected papers from {papers_dir}: {e}")
        return {}
    
    return chooseref


def generate_x_json(task_id: str, survey_tex_path: Path, references_bib_path: Path, output_path: Path):
    """
    Generate x.json file containing survey text and references.
    
    Args:
        task_id: Task ID (e.g., "t1", "t2")
        survey_tex_path: Path to survey.tex file
        references_bib_path: Path to references.bib file
        output_path: Path where x.json should be saved
    """
    try:
        # Read survey.tex
        with open(survey_tex_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        # Extract citation keys and build key-to-number mapping
        key_to_number, _ = extract_citations_from_latex(latex_content)
        
        # Extract text from LaTeX with citation numbers included
        survey_text = extract_text_from_latex(latex_content, include_citations=True)
        
        # Parse bib file to get information for each citation key
        # Pass task_id to enable arxivid lookup from ref.jsonl and papers directory
        bib_info_by_key = parse_bib_file_by_key(references_bib_path, task_id=task_id)
        
        # Build reference dictionary: map citation number to arxivid or title
        # Numbers are assigned in order of first appearance in the document
        references = {}
        # Sort keys by their citation number
        sorted_keys = sorted(key_to_number.items(), key=lambda x: x[1])
        
        for citation_key, citation_number in sorted_keys:
            ref_value = None
            if citation_key in bib_info_by_key:
                bib_info = bib_info_by_key[citation_key]
                # Prefer arxivid if available, otherwise use title
                ref_value = bib_info.get("arxivid") or bib_info.get("title", citation_key)
            else:
                # Fallback: use the key itself
                ref_value = citation_key
                logger.warning(f"Citation key '{citation_key}' not found in bib file")
            
            references[str(citation_number)] = ref_value
        
        # Parse selected papers (actually processed papers from outputs/{task_id}/papers/)
        chooseref = parse_selected_papers(task_id)
        
        # Create output structure similar to template
        # survey is a string with citation numbers like [1], [2,3], etc.
        # reference is a dict mapping citation number to arxivid/title (only cited references)
        # chooseref is a dict mapping index to arxivid/title (all selected papers, up to 200)
        output_data = {
            "survey": survey_text,  # String with citation numbers included
            "reference": references,  # Only references actually cited in the survey
            "chooseref": chooseref    # All selected and processed papers (up to 200)
        }
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated x.json at {output_path}")
        logger.info(f"Cited references: {len(references)}, Selected references: {len(chooseref)}")
        
    except Exception as e:
        logger.error(f"Error generating x.json: {e}")
        raise


def generate_xc_json(task_id: str, output_path: Path):
    """
    Merge time_monitor.json and token_monitor.json into xc.json.
    
    Args:
        task_id: Task ID
        output_path: Path where xc.json should be saved
    """
    from src.configs.constants import OUTPUT_DIR
    
    try:
        task_dir = OUTPUT_DIR / task_id
        time_monitor_path = task_dir / "metrics" / "time_monitor.json"
        token_monitor_path = task_dir / "metrics" / "token_monitor.json"
        
        # Load time monitor data
        time_data = {}
        if time_monitor_path.exists():
            with open(time_monitor_path, 'r', encoding='utf-8') as f:
                time_data = json.load(f)
        else:
            logger.warning(f"Time monitor file not found: {time_monitor_path}")
        
        # Load token monitor data
        token_data = {}
        if token_monitor_path.exists():
            with open(token_monitor_path, 'r', encoding='utf-8') as f:
                token_data = json.load(f)
        else:
            logger.warning(f"Token monitor file not found: {token_monitor_path}")
        
        # Create merged structure
        output_data = {
            "time": time_data,
            "token": token_data
        }
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Generated xc.json at {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating xc.json: {e}")
        raise
