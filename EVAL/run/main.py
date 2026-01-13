import json
import os

# 读取 topic.json 文件
topic_json_path = os.path.join(os.path.dirname(__file__), '..', 'code', 'topic.json')
with open(topic_json_path, 'r', encoding='utf-8') as f:
    topics = json.load(f)

# 创建字典，以 id 为键
topic_dict = {item['id']: item['topic'] for item in topics}

# 获取 id 1-20 的 topic 值
topics_list = [topic_dict[i] for i in range(1, 21)]

# 生成三个 txt 文件
output_dir = os.path.dirname(__file__)

# 生成 x.txt (格式参考 runtool 39-40 行)
x_lines = []
for topic in topics_list:
    line = f'python tasks/offline_run.py --title "{topic}" --key_words "{topic}"'
    x_lines.append(line)

with open(os.path.join(output_dir, 'x.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(x_lines) + '\n')

# 生成 a.txt (格式参考 runtool 8-9 行)
a_lines = []
for topic in topics_list:
    line = f'export HF_ENDPOINT=https://hf-mirror.com && python main.py --topic "{topic}"'
    a_lines.append(line)

with open(os.path.join(output_dir, 'a.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(a_lines) + '\n')

# 生成 f.txt (格式参考 runtool 23-24 行)
f_lines = []
for topic in topics_list:
    line = f'python run_demo.py --topic "{topic}"'
    f_lines.append(line)

with open(os.path.join(output_dir, 'f.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(f_lines) + '\n')

# 生成 x1.txt, x2.txt (在 x.txt 基础上每行末尾加上 --mid 参数)
with open(os.path.join(output_dir, 'x1.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([line + ' --mid 1' for line in x_lines]) + '\n')

with open(os.path.join(output_dir, 'x2.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([line + ' --mid 2' for line in x_lines]) + '\n')

# 生成 a1.txt, a2.txt (在 a.txt 基础上每行末尾加上 --mid 参数)
with open(os.path.join(output_dir, 'a1.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([line + ' --mid 1' for line in a_lines]) + '\n')

with open(os.path.join(output_dir, 'a2.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([line + ' --mid 2' for line in a_lines]) + '\n')

# 生成 f1.txt, f2.txt (在 f.txt 基础上每行末尾加上 --mid 参数)
with open(os.path.join(output_dir, 'f1.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([line + ' --mid 1' for line in f_lines]) + '\n')

with open(os.path.join(output_dir, 'f2.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join([line + ' --mid 2' for line in f_lines]) + '\n')

print(f"已成功生成 x.txt, a.txt, f.txt, x1.txt, x2.txt, a1.txt, a2.txt, f1.txt, f2.txt 文件，每个文件包含 {len(topics_list)} 行")
