# eval.jsonl 数据结构说明

## 行类型结构

eval.jsonl 文件中每行都是一个 JSON 对象，包含 `name` 字段和另一个评估指标字段。根据除 `name` 外的键不同，共有以下类型：

### 类型 1: hsr // name取值中:a f x对应autoservey serveyforge surveyx。null 1 2对应gpt-4o claude-3-5-haiku-20241022 deepseek-v3
```bash
{
  "name": "string",
  "hsr": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **hsr**: heading soft recall (hsr)

### 类型 2: her // -1.0值表示黄金大纲抽出实体为0
```bash
{
  "name": "string",
  "her": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **her**: heading entity recall (her)

### 类型 3: outline 
```bash
{
  "name": "string",
  "outline": [
    "number",
    "number",
    "number"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **outline**: 大纲评估分数列表，包含3个整数，格式为 [int, int, int]
      对应Guidance for Content Generation 和 Hierarchical Clarity 和 Logical Coherence

### 类型 4: citationrecall 
```bash
{
  "name": "string",
  "citationrecall": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **citationrecall**: Citation Recall

### 类型 5: citationprecision 
```bash
{
  "name": "string",
  "citationprecision": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **citationprecision**: Citation Precision

### 类型 6: paperold 
```bash
{
  "name": "string",
  "paperold": [
    "number",
    "number",
    "number",
    "number"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **paperold**: 评估分数列表，包含4个整数，格式为 [int, int, int, int]
对应Coverage、 Structure、 Relevance、Language

### 类型 7: paperour 和 reason  //our给的提示词的分析结果
```bash
{
  "name": "string",
  "paperour": [
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number"
  ],
  "reason": [
    "string",
    "string",
    "string",
    "string",
    "string",
    "string",
    "string"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **paperour**: 评估分数列表，包含7个整数，格式为 [int, int, int, int, int, int, int]，对应7个维度
- **reason**: 评估理由列表，包含7个字符串，每个字符串是对应评估维度的详细说明

### 类型 8: rouge 
```bash
{
  "name": "string",
  "rouge": [
    "number",
    "number",
    "number"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **rouge**: ROUGE 评估指标列表，ROUGE-1、ROUGE-2、ROUGE-L，格式为 [number, number, number]

### 类型 9: bleu 
```bash
{
  "name": "string",
  "bleu": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **bleu**: BLEU 

### 类型 10: recallak  //x无检索，无该值
```bash
{
  "name": "string",
  "recallak": [
    "number",
    "number",
    "number",
    "number",
    "number",
    "number"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2"
- **recallak**: Recall@K值列表，k=20,30,100,200,500,1000 [number, number, number, number, number, number]

### 类型 11: recallpref 
```bash
{
  "name": "string",
  "recallpref": [
    "number",
    "number",
    "number"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **recallpref**: 列表[number, number, number],
                   recall、precision和F1

### 类型 12: lourele //不一定有该数据
```bash
{
  "name": "string",
  "lourele": [
    "number",
    "number",
    "number"
  ]
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **lourele**:列表[number, number, number]
      IoU、Relevancesemantic、RelevanceLLM

