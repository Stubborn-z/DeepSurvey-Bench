import re
from typing import List, Optional

from flair.data import Sentence
from flair.nn import Classifier
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 延迟加载模型，避免导入时网络问题导致失败
_tagger = None
_encoder = None

def get_tagger():
    """延迟加载 NER tagger"""
    global _tagger
    if _tagger is None:
        try:
            _tagger = Classifier.load('ner')
        except Exception as e:
            error_msg = (
                f"无法加载 Flair NER 模型。错误: {e}\n"
                f"可能的原因：\n"
                f"1. 网络连接问题（无法从 HuggingFace 下载模型）\n"
                f"2. 本地没有缓存的模型文件\n"
                f"解决方案：\n"
                f"1. 检查网络连接和代理设置\n"
                f"2. 手动下载模型：python -c \"from flair.nn import Classifier; Classifier.load('ner')\"\n"
                f"3. 或者在有网络的环境下运行一次以缓存模型"
            )
            raise RuntimeError(error_msg) from e
    return _tagger

def get_encoder():
    """延迟加载 sentence encoder"""
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return _encoder


def card(l):
    encoder = get_encoder()  # 延迟加载
    encoded_l = encoder.encode(list(l))
    cosine_sim = cosine_similarity(encoded_l)
    soft_count = 1 / cosine_sim.sum(axis=1)

    return soft_count.sum()


def heading_soft_recall(golden_headings: List[str], predicted_headings: List[str], similarity_threshold: float = 0.7):
    """
    Given golden headings and predicted headings, compute soft recall.
    Uses cosine similarity > threshold to determine heading matching.
        -  golden_headings: list of strings
        -  predicted_headings: list of strings
        -  similarity_threshold: cosine similarity threshold for heading matching (default: 0.7)

    Ref: https://www.sciencedirect.com/science/article/pii/S0167865523000296
    """
    # 如果真实大纲为空，返回 -1
    if len(golden_headings) == 0:
        return -1.0

    g = set(golden_headings)
    p = set(predicted_headings)
    if len(p) == 0:
        return 0.0
    
    card_g = card(g)
    
    # 使用相似度匹配而不是完全匹配
    # 首先尝试精确匹配
    exact_intersection = g.intersection(p)
    matched_golden = set(exact_intersection)
    unmatched_predicted = p - exact_intersection
    
    # 对于未精确匹配的标题，使用向量相似度匹配
    unmatched_golden = g - matched_golden
    
    if len(unmatched_golden) == 0:
        # 所有真实标题都精确匹配
        if len(exact_intersection) == 0:
            card_intersection = 0.0
        else:
            card_intersection = card(exact_intersection)
    else:
        # 使用向量相似度匹配未精确匹配的标题
        if len(unmatched_predicted) == 0:
            # 没有可匹配的预测标题，只计算精确匹配的部分
            if len(exact_intersection) == 0:
                card_intersection = 0.0
            else:
                card_intersection = card(exact_intersection)
        else:
            # 编码未匹配的标题
            encoder = get_encoder()  # 延迟加载
            all_unmatched = list(unmatched_golden | unmatched_predicted)
            heading_vectors = encoder.encode(all_unmatched)
            heading_to_idx = {heading: idx for idx, heading in enumerate(all_unmatched)}
            
            # 计算相似度匹配
            similarity_matched_golden = set()
            for golden_heading in unmatched_golden:
                golden_idx = heading_to_idx[golden_heading]
                golden_vec = heading_vectors[golden_idx:golden_idx+1]
                
                max_similarity = 0
                for predicted_heading in unmatched_predicted:
                    pred_idx = heading_to_idx[predicted_heading]
                    pred_vec = heading_vectors[pred_idx:pred_idx+1]
                    similarity = cosine_similarity(golden_vec, pred_vec)[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity >= similarity_threshold:
                    similarity_matched_golden.add(golden_heading)
            
            # 合并精确匹配和相似度匹配的标题
            all_matched = matched_golden | similarity_matched_golden
            
            if len(all_matched) == 0:
                card_intersection = 0.0
            else:
                card_intersection = card(all_matched)
    
    hsr = card_intersection / card_g
    # 由于浮点数精度问题，HSR 可能略微大于 1，将其限制在 [0, 1] 范围内
    return min(max(hsr, 0.0), 1.0)


def extract_entities_from_list(l):
    """
    提取实体列表（与原始实现保持一致）
    只转小写，不进行其他规范化，保持与原始代码一致
    """
    tagger = get_tagger()  # 延迟加载
    entities = []
    for sent in l:
        if len(sent) == 0:
            continue
        sent = Sentence(sent)
        tagger.predict(sent)
        entities.extend([e.text for e in sent.get_spans('ner')])

    # 只转小写，与原始实现保持一致
    entities = list(set([e.lower() for e in entities]))

    return entities


def heading_entity_recall(golden_entities: Optional[List[str]] = None,
                          predicted_entities: Optional[List[str]] = None,
                          golden_headings: Optional[List[str]] = None,
                          predicted_headings: Optional[List[str]] = None,
                          similarity_threshold: float = 0.6):
    """
    Given golden entities and predicted entities, compute entity recall.
    Uses cosine similarity > threshold to determine entity matching.
    Entity extraction is consistent with original implementation (only lowercase).
        -  golden_entities: list of strings or None; if None, extract from golden_headings
        -  predicted_entities: list of strings or None; if None, extract from predicted_headings
        -  golden_headings: list of strings or None
        -  predicted_headings: list of strings or None
        -  similarity_threshold: cosine similarity threshold for entity matching (default: 0.7)
    """
    if golden_entities is None:
        assert golden_headings is not None, "golden_headings and golden_entities cannot both be None."
        golden_entities = extract_entities_from_list(golden_headings)
    if predicted_entities is None:
        assert predicted_headings is not None, "predicted_headings and predicted_entities cannot both be None."
        predicted_entities = extract_entities_from_list(predicted_headings)
    
    if len(golden_entities) == 0:
        # 如果真实大纲没有实体，检查预测实体
        # 如果预测也没有实体，认为完美匹配（返回1）
        # 如果预测有实体，应该如何处理？这里返回1，因为无法计算recall
        # 另一种方案是返回None，但为了兼容性，返回1
        return 1.0 if len(predicted_entities) == 0 else 0.0
    
    if len(predicted_entities) == 0:
        return 0.0
    
    # 首先尝试精确匹配（与原始实现兼容）
    g_set = set(golden_entities)
    p_set = set(predicted_entities)
    exact_matches = g_set.intersection(p_set)
    
    # 对于未精确匹配的实体，使用向量相似度匹配
    unmatched_golden = g_set - exact_matches
    unmatched_predicted = p_set - exact_matches
    
    if len(unmatched_golden) == 0:
        # 所有实体都精确匹配
        return float(len(exact_matches)) / len(golden_entities)
    
    # 使用向量相似度匹配未精确匹配的实体
    if len(unmatched_predicted) == 0:
        # 没有可匹配的预测实体
        return float(len(exact_matches)) / len(golden_entities)
    
    # 编码未匹配的实体
    encoder = get_encoder()  # 延迟加载
    all_unmatched = list(unmatched_golden | unmatched_predicted)
    entity_vectors = encoder.encode(all_unmatched)
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_unmatched)}
    
    # 计算相似度匹配
    similarity_matches = 0
    for golden_entity in unmatched_golden:
        golden_idx = entity_to_idx[golden_entity]
        golden_vec = entity_vectors[golden_idx:golden_idx+1]
        
        max_similarity = 0
        for predicted_entity in unmatched_predicted:
            pred_idx = entity_to_idx[predicted_entity]
            pred_vec = entity_vectors[pred_idx:pred_idx+1]
            similarity = cosine_similarity(golden_vec, pred_vec)[0][0]
            max_similarity = max(max_similarity, similarity)
        
        if max_similarity >= similarity_threshold:
            similarity_matches += 1
    
    # 总匹配数 = 精确匹配 + 相似度匹配
    total_matches = len(exact_matches) + similarity_matches
    return float(total_matches) / len(golden_entities)


def article_entity_recall(golden_entities: Optional[List[str]] = None,
                          predicted_entities: Optional[List[str]] = None,
                          golden_article: Optional[str] = None,
                          predicted_article: Optional[str] = None):
    """
    Given golden entities and predicted entities, compute entity recall.
        -  golden_entities: list of strings or None; if None, extract from golden_article
        -  predicted_entities: list of strings or None; if None, extract from predicted_article
        -  golden_article: string or None
        -  predicted_article: string or None
    """
    if golden_entities is None:
        assert golden_article is not None, "golden_article and golden_entities cannot both be None."
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', golden_article)
        golden_entities = extract_entities_from_list(sentences)
    if predicted_entities is None:
        assert predicted_article is not None, "predicted_article and predicted_entities cannot both be None."
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', predicted_article)
        predicted_entities = extract_entities_from_list(sentences)
    g = set(golden_entities)
    p = set(predicted_entities)
    if len(g) == 0:
        return 1
    else:
        return len(g.intersection(p)) / len(g)


def compute_rouge_scores(golden_answer: str, predicted_answer: str):
    """
    Compute rouge score for given output and golden answer to compare text overlap.
        - golden_answer: plain text of golden answer
        - predicted_answer: plain text of predicted answer
    """

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(golden_answer, predicted_answer)
    score_dict = {}
    for metric, metric_score in scores.items():
        score_dict[f'{metric.upper()}_precision'] = metric_score.precision
        score_dict[f'{metric.upper()}_recall'] = metric_score.recall
        score_dict[f'{metric.upper()}_f1'] = metric_score.fmeasure
    return score_dict

