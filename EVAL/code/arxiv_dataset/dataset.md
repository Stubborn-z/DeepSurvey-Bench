# 数据集构建说明
## 0. 数据集文件
<1> arxiv_433.jsonl --- 元数据
<br>
<2> reviews_433.jsonl --- GPT生成的综述
<br>
<3> title_433.jsonl ---大纲
## 1. 采样
### ArXiv侧
选取2022-01-01至2025-11-30时间区间内，发表在ArXiv上的综述论文。所有ArXiv论文的元数据来自于自己收集的ArXiv的历史元数据。<br>

仅保存上一步title中，明确包含"survey", "literature review", "review"单词的那些标题。<br>

_ArXiv侧负责提供我们数据集中的title，Abstract，category_

### Semantic scholar侧
通过上一步中获取到的那一批带有ArXiv id（唯一）的数据，和自己收集的semantic schloar提供的元数据做匹配，靠semantic scholar历史版本（2025-12-02），从中匹配externalids.ArXiv和我们ArXiv id相匹配的那些数据，同时要求发表于2022-2024年的论文，被引数必须大于40，发表于2025年的论文，被引数必须大于20，来收集authors，year，date，cite_counts（这篇论文的引用列表的长度），Conference_journal_name，influentialcitationcount（semantic scholar定义）以及第一作者的信息
```bash
"Author_info": {
      "Publicationsh": 45, #作者发布的论文数量
      "h_index": 13, #作者的h_index
      "Citations": 1637, #作者的被引用量
      "Highly Influential Citations": 0 #这个靠遍历semantic scholar数据库，累加该作者发布的论文的 influentialcitationcount
    }
```
_我们使用semantic scholar中的corpusid作为我们的literature_review_id作为我们数据集中每一条数据的唯一标识符_
<br>
_采样共获取1101条数据_
## 2. 富化
从ArXiv上收集有TeX Source的论文（仅有896篇有Tex Source），和论文的PDF。
<br>
对于Tex Source，我们采取"\section","\subsection","\subsubsection"来解析论文中的一级标题，二级标题，三级标题，通过"\cite"来解析引用数量。
```bash
{"section_title": 
    "Main Body",
    "level": "1",
    "content":"~",
    "origin_cites_number": 48
}
```
对于引用列表，我们采用grobid来解析完整的引用列表。

## 3. 筛选
第二步中解析成功，长度合格的那些论文中，我们采样出了457篇，然后从另一个数据集中通过去重，又补充了11篇<br>
然后靠categories中进行采选，保留所有cs类别的，丢弃q-bio.BM 、econ.EM ，eess.开头的随机保留15条，物理学的随机保留15条。
<br>
最后得到了一个433篇文献的数据集

## 4. 结构
```bash
{
  "authors": [
    "string"
  ],
  "literature_review_title": "string",
  "year": "string",
  "date": "string",
  "category": "string",
  "abstract": "string",
  "structure": [
    {
      "section_title": "string",
      "level": "string",
      "content": "string",
      "origin_cites_number": "number"
    }
  ],
  "literature_review_id": "number",
  "meta_info": {
    "cite_counts": "number",
    "Conference_journal_name": "string",
    "influentialcitationcount": "number",
    "Author_info": {
      "Publicationsh": "number",
      "h_index": "number",
      "Citations": "number",
      "Highly Influential Citations": "number"
    },
    "all_cites_title": [
      "string"
    ]
  }
}
```

## 5. GPT生成综述
prompt:
```bash
ROUTER_PROMPT = """You are an academic paper structure analyzer.
I will provide you with a list of section titles from a literature review paper.
Your task is to classify EVERY section index into one of these four categories:
1. "abs_intro", 2. "method", 3. "experiment", 4. "gap".
Output STRICTLY JSON with keys: "abs_intro", "method", "experiment", "gap"."""

PROMPTS = {
    "abs_intro": """You are acting as an expert in domain literature review papers. Based ONLY on the Abstract and Introduction sections, extract: 1. Research Background and Motivation, 2. Core Research Questions and Task Positioning, 3. Fundamental Terminology and Core Concepts, 4. Field Status and Development Trends, 5. Structured and Visual Contributions. Output strictly in JSON format.""",
    "method": """You are acting as an expert in methodological taxonomy. Based on Method/Related Work sections, extract: 1. Method Taxonomy, 2. Method Evolution, 3. Commonalities/Relationships, 4. Method Comparison/Insights, 5. Systematic Presentation. Output strictly in JSON format.""",
    "experiment": """You are acting as an expert in experimental design. Based on Data/Evaluation/Experiments sections, extract: 1. Dataset Systematic Review, 2. Evaluation Metric Framework, 3. Experimental Results, 4. Result Visualization. Output strictly in JSON format.""",
    "gap": """You are acting as an expert in identifying research gaps. Based on Gap/Limitations/Future Work/Conclusion, extract: 1. Limitations and Bottlenecks, 2. Overall Assessment of Field, 3. Future Research Directions. Output strictly in JSON format."""
}
```
结构：
```bash
{
  "literature_review_id": "number",
  "Research Background and Motivation": "string",
  "Core Research Questions and Task Positioning": "string",
  "Fundamental Terminology and Core Concepts": "array",
  "Field Status and Development Trends": "string",
  "Structured and Visual Contributions": "array",
  "Method Taxonomy": "object",
  "Method Evolution": "object",
  "Commonalities/Relationships": "object",
  "Method Comparison/Insights": "object",
  "Systematic Presentation": "object",
  "Dataset Systematic Review": "object",
  "Evaluation Metric Framework": "object",
  "Experimental Results": "object",
  "Result Visualization": "object",
  "Limitations and Bottlenecks": "array",
  "Overall Assessment of Field": "array",
  "Future Research Directions": "array"
}
```
首先，用上一步收集到的section_title进行一次路由分桶，由GPT-4o判断每一个title应该属于哪一个部分（temperature=0.2），然后分四个问题，分别去问GPT-4o并进行组装。
<br>
同时，由于某些部分的正文可能非常的长，超出了128k的上限，所以我们对于超长的正文，进行自动的切块（100k），由GPT总结后再进行组装。<br>
_最后产生了431条数据_

## 6. 大纲结构
```bash
{
  "literature_review_id": "number",
  "structure": "array"
}
```