# eval.jsonl 数据结构说明

## 行类型结构

eval.jsonl 文件中每行都是一个 JSON 对象，包含 `name` 字段和另一个评估指标字段。根据除 `name` 外的键不同，共有以下类型：

### 类型 1: hsr (Hit Success Rate)
```bash
{
  "name": "string",
  "hsr": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **hsr**: 命中成功率，数值类型

### 类型 2: her (Hit Error Rate)
```bash
{
  "name": "string",
  "her": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **her**: 命中错误率，数值类型

### 类型 3: outline (大纲评估)
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

### 类型 4: citationrecall (引用召回率)
```bash
{
  "name": "string",
  "citationrecall": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **citationrecall**: 引用召回率，数值类型

### 类型 5: citationprecision (引用精确率)
```bash
{
  "name": "string",
  "citationprecision": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **citationprecision**: 引用精确率，数值类型

### 类型 6: paperold (旧论文评估)
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
- **paperold**: 旧论文评估分数列表，包含4个整数，格式为 [int, int, int, int]

### 类型 7: paperour (新论文评估) 和 reason (评估理由)
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
- **paperour**: 新论文评估分数列表，包含7个整数，格式为 [int, int, int, int, int, int, int]
- **reason**: 评估理由列表，包含7个字符串，每个字符串是对应评估维度的详细说明

### 类型 8: rouge (ROUGE 评估指标)
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
- **rouge**: ROUGE 评估指标列表，包含3个数值，通常对应 ROUGE-1、ROUGE-2、ROUGE-L，格式为 [number, number, number]

### 类型 9: bleu (BLEU 评估指标)
```bash
{
  "name": "string",
  "bleu": "number"
}
```
- **name 可能的取值**: "a", "a1", "a2", "f", "f1", "f2", "x", "x1", "x2"
- **bleu**: BLEU 评估分数，数值类型

### 类型 10: recallak (Recall@K 评估)
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
- **recallak**: Recall@K 评估指标列表，包含6个数值，通常对应不同 K 值下的召回率，格式为 [number, number, number, number, number, number]

### 类型 11: recallpref (偏好召回率)
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
- **recallpref**: 偏好召回率列表，包含3个数值，格式为 [number, number, number]

### 类型 12: lourele (Lourele 评估指标)
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
- **lourele**: Lourele 评估指标列表，包含3个数值，格式为 [number, number, number]

