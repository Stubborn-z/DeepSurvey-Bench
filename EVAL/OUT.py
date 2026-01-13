#!/usr/bin/env python3
import json
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

# 解析 eval_structure.md 提取类型和 name 取值
def parse_eval_structure(md_file):
    """解析 eval_structure.md 文件，提取所有类型和对应的 name 可能取值"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    types_info = {}
    
    # 匹配每个类型块
    pattern = r'### 类型 \d+: ([^(]+)\([^)]*\)\s*```bash\s*\{[^}]*"([^"]+)":\s*"[^"]+"[^}]*\}\s*```\s*-\s*\*\*name 可能的取值\*\*:\s*"([^"]+)"'
    
    # 更灵活的匹配方式
    type_blocks = re.split(r'### 类型 \d+:', content)
    
    for block in type_blocks[1:]:  # 跳过第一个空块
        # 提取类型名称（第一个非空行）
        lines = block.strip().split('\n')
        if not lines:
            continue
            
        # 查找类型键名（在代码块中）
        code_block_match = re.search(r'```bash\s*\{[^}]*"([^"]+)":', block)
        if not code_block_match:
            continue
        type_key = code_block_match.group(1)
        
        # 查找 name 可能取值
        name_match = re.search(r'\*\*name 可能的取值\*\*:\s*"([^"]+)"', block)
        if not name_match:
            continue
        
        # 解析 name 取值列表（可能是多个用引号包围的值）
        name_str = name_match.group(1)
        # 提取所有引号内的值
        name_values = re.findall(r'"([^"]+)"', name_str)
        if not name_values:
            # 如果没有引号，尝试按逗号分割
            name_values = [v.strip() for v in name_str.split(',')]
        
        types_info[type_key] = set(name_values)
    
    return types_info

# 从文档中手动提取类型信息（更可靠的方法）
def get_types_info():
    """从 eval_structure.md 中提取类型信息"""
    # 使用有序列表，保持 name 的排序顺序
    types_info = {
        'hsr': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'her': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'outline': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'citationrecall': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'citationprecision': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'paperold': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'paperour': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],  # paperour 和 reason 一起
        'rouge': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'bleu': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'recallak': ['a', 'a1', 'a2', 'f', 'f1', 'f2'],
        'recallpref': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
        'lourele': ['a', 'a1', 'a2', 'f', 'f1', 'f2', 'x', 'x1', 'x2'],
    }
    return types_info

def get_type_key(data):
    """获取数据行的类型键（排除 name）"""
    keys = [k for k in data.keys() if k != 'name']
    # 对于 paperour 类型，需要同时包含 paperour 和 reason
    if 'paperour' in keys:
        return ('paperour', 'reason')
    return tuple(sorted(keys))

def check_eval_file(eval_file, types_info):
    """检查 eval.jsonl 文件是否包含所有类型，且每个类型的所有 name 值都出现
    返回: (is_valid, message, missing_lourele)
    missing_lourele: True 表示缺失 lourele 类型
    """
    if not os.path.exists(eval_file):
        return False, "文件不存在", False
    
    found_types = defaultdict(set)  # type_key -> set of names
    
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'name' not in data:
                        continue
                    
                    name = data.get('name')
                    type_key = get_type_key(data)
                    
                    # 转换为字符串键以便比较
                    if isinstance(type_key, tuple):
                        type_str = ','.join(sorted(type_key))
                    else:
                        type_str = str(type_key)
                    
                    # 检查是否匹配已知类型
                    for type_name, expected_names in types_info.items():
                        if type_name in type_str or (type_name == 'paperour' and 'paperour' in type_str and 'reason' in type_str):
                            found_types[type_name].add(name)
                            break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return False, f"读取文件出错: {e}", False
    
    # 检查是否包含所有类型（lourele 除外）
    required_types = set(types_info.keys()) - {'lourele'}
    missing_types = required_types - set(found_types.keys())
    if missing_types:
        return False, f"缺少类型: {missing_types}", False
    
    # 检查每个类型是否包含所有 name 值（lourele 除外）
    for type_name, expected_names in types_info.items():
        if type_name == 'lourele':
            continue  # 跳过 lourele 的检查
        found_names = found_types.get(type_name, set())
        expected_set = set(expected_names)
        missing_names = expected_set - found_names
        if missing_names:
            return False, f"类型 {type_name} 缺少 name 值: {missing_names}", False
    
    # 检查是否缺失 lourele（完全缺失或缺少部分 name 值）
    missing_lourele = False
    if 'lourele' not in found_types:
        missing_lourele = True  # 完全没有 lourele 类型
    else:
        found_lourele_names = found_types.get('lourele', set())
        expected_lourele_set = set(types_info['lourele'])
        missing_lourele_names = expected_lourele_set - found_lourele_names
        if missing_lourele_names:
            missing_lourele = True  # 有 lourele 但缺少部分 name 值
    
    return True, "通过检查", missing_lourele

def sort_lines(lines, types_info):
    """对行进行排序：先按类型，再按 name 值"""
    def get_sort_key(line):
        try:
            data = json.loads(line)
            name = data.get('name', '')
            
            # 确定类型
            type_key = get_type_key(data)
            if isinstance(type_key, tuple):
                type_str = ','.join(sorted(type_key))
            else:
                type_str = str(type_key)
            
            # 找到对应的类型名称
            type_name = None
            for tn in types_info.keys():
                if tn in type_str or (tn == 'paperour' and 'paperour' in type_str and 'reason' in type_str):
                    type_name = tn
                    break
            
            if type_name is None:
                type_name = type_str
            
            # 获取类型在 types_info 中的顺序
            type_order = list(types_info.keys()).index(type_name) if type_name in types_info else 999
            
            # 获取 name 在对应类型中的顺序
            if type_name in types_info:
                name_list = types_info[type_name]
                name_order = name_list.index(name) if name in name_list else 999
            else:
                name_order = 999
            
            return (type_order, name_order)
        except:
            return (999, 999)
    
    return sorted(lines, key=get_sort_key)

def process_eval_files():
    """处理所有 eval.jsonl 文件"""
    base_dir = Path('/home/liudingyuan/code/SurveyX/code/EVAL')
    structure_file = base_dir / 'code' / 'eval_structure.md'
    out_base = base_dir / 'OUT'
    
    # 获取类型信息
    types_info = get_types_info()
    
    # 创建输出目录
    for n in range(1, 5):
        out_dir = out_base / f'p{n}'
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # 映射 M 到 p<n>
    def get_p_dir(M):
        if 1 <= M <= 5:
            return out_base / 'p1'
        elif 6 <= M <= 10:
            return out_base / 'p2'
        elif 11 <= M <= 15:
            return out_base / 'p3'
        elif 16 <= M <= 20:
            return out_base / 'p4'
        return None
    
    # 处理每个 tM
    for M in range(1, 21):
        eval_file = base_dir / f't{M}' / 'eval.jsonl'
        p_dir = get_p_dir(M)
        output_file = p_dir / f'eval{M}.jsonl'
        
        print(f"\n处理 t{M}/eval.jsonl...")
        
        # 检查文件是否存在
        if not eval_file.exists():
            print(f"  跳过: 文件不存在")
            continue
        
        # 检查是否满足条件
        is_valid, message, missing_lourele = check_eval_file(eval_file, types_info)
        if not is_valid:
            print(f"  跳过: {message}")
            continue
        
        # 检查目标文件是否已存在，如果存在则提示覆盖
        file_exists = output_file.exists()
        if file_exists:
            print(f"  ⚠ 目标文件已存在，将覆盖: {output_file}")
        
        # 读取并处理文件
        try:
            lines = []
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 删除空行
                        # 验证是否为有效 JSON
                        try:
                            json.loads(line)
                            lines.append(line)
                        except json.JSONDecodeError:
                            continue  # 跳过无效的 JSON 行
            
            # 排序
            lines = sort_lines(lines, types_info)
            
            # 写入输出文件（覆盖模式）
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            
            # 输出复制信息，如果缺失 lourele 则注明
            action = "已覆盖" if file_exists else "已复制到"
            if missing_lourele:
                print(f"  ✓ {action} {output_file} ({len(lines)} 行) [缺失 lourele]")
            else:
                print(f"  ✓ {action} {output_file} ({len(lines)} 行)")
        except Exception as e:
            print(f"  错误: 处理文件时出错 - {e}")

if __name__ == '__main__':
    process_eval_files()

