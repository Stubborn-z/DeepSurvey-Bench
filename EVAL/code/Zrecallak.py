#!/usr/bin/env python3
"""
批量评估所有 t<a>/ 目录下的 a<b>.json 和 f<b>.json 文件的 Recall@K 指标
"""

import os
import json
import subprocess
import sys

def load_existing_results(eval_jsonl_path):
    """
    加载 eval.jsonl 中已存在的结果名称集合和结果字典
    
    跳过条件：某行name键的值和该实例想要得到的name键的值相同，且该行有recallak键
    例如：如果某行 name='a2' 且该行有 recallak 键，则实例 a2.json 会被跳过
    
    Returns:
        (existing_names, results_dict)
        existing_names: 已存在的结果名称集合
        results_dict: {name: result_dict} 字典，存储每个name对应的完整结果
    """
    existing_names = set()
    results_dict = {}
    if os.path.exists(eval_jsonl_path):
        try:
            with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        name = item.get('name', '')
                        # 只有当name存在且recallak键也存在时才认为该实例已处理
                        # 符合跳过条件：name键的值和实例的name相同，且该行有recallak键
                        if name and 'recallak' in item:
                            existing_names.add(name)
                            results_dict[name] = item
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"读取 {eval_jsonl_path} 时出错: {e}")
    return existing_names, results_dict

def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath) and os.path.isfile(filepath)

def evaluate_instance(t_num, method_name, code_dir, base_dir):
    """
    评估单个实例
    
    Args:
        t_num: t目录编号（如 1, 2, ..., 20）
        method_name: 方法名称（如 'a', 'a1', 'f', 'f2' 等）
        code_dir: recallak.py 所在的目录
        base_dir: EVAL 目录
    
    Returns:
        (success, result_dict) 如果成功返回 (True, result_dict)，失败返回 (False, None)
    """
    # 构建 recallak.py 的调用命令
    recallak_script = os.path.join(code_dir, 'recallak.py')
    cmd = [sys.executable, recallak_script, '--t', str(t_num), '--method', method_name]
    
    # 验证要评估的文件路径
    json_file = os.path.join(base_dir, f't{t_num}', f'{method_name}.json')
    if not os.path.exists(json_file):
        print(f"错误: 文件不存在 {json_file}")
        return False, None
    
    try:
        # 调用 recallak.py
        result = subprocess.run(
            cmd,
            cwd=code_dir,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            # 读取评估结果
            eval_jsonl_path = os.path.join(base_dir, f't{t_num}', 'eval.jsonl')
            if os.path.exists(eval_jsonl_path):
                # 读取最后一行（刚写入的结果）
                with open(eval_jsonl_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        try:
                            last_line = lines[-1].strip()
                            if last_line:
                                result_dict = json.loads(last_line)
                                if result_dict.get('name') == method_name and 'recallak' in result_dict:
                                    return True, result_dict
                        except json.JSONDecodeError:
                            pass
            return True, None
        else:
            print(f"处理 t{t_num}/{method_name}.json 时出错:")
            print(result.stderr)
            return False, None
    
    except Exception as e:
        print(f"调用 recallak.py 处理 t{t_num}/{method_name}.json 时出错: {e}")
        return False, None

def main():
    # 获取脚本所在目录
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(code_dir, '..')
    
    # t 目录范围：1 到 20
    t_range = range(1, 21)
    
    # 方法前缀：a 和 f
    method_prefixes = ['a', 'f']
    
    # 方法后缀：空字符串和 1-9
    method_suffixes = [''] + [str(i) for i in range(1, 10)]
    
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    
    # 遍历所有 t 目录
    for t_num in t_range:
        t_dir = os.path.join(base_dir, f't{t_num}')
        
        # 检查 t 目录是否存在
        if not os.path.exists(t_dir) or not os.path.isdir(t_dir):
            continue
        
        print(f"\n{'='*60}")
        print(f"处理目录: t{t_num}")
        print(f"{'='*60}")
        
        # 加载该目录下已存在的结果
        eval_jsonl_path = os.path.join(t_dir, 'eval.jsonl')
        existing_names, results_dict = load_existing_results(eval_jsonl_path)
        
        # 遍历所有可能的方法组合
        # 重要：确保先处理基础方法（a 和 f），再处理变体（a1, a2, f1, f2 等）
        # 这样变体才能复制基础方法的结果
        for prefix in method_prefixes:
            # 首先处理基础方法（suffix=''，即 a 或 f）
            # 确保在检查 a<b> 或 f<b> 之前，a 或 f 已经被评估和写入
            base_method_name = prefix  # a 或 f
            base_json_file = os.path.join(t_dir, f'{base_method_name}.json')
            
            # 处理基础方法（如果文件存在）
            if check_file_exists(base_json_file):
                # 检查结果是否已存在
                if base_method_name not in existing_names:
                    # 进行实际评估
                    print(f"\n处理实例: t{t_num}/{base_method_name}.json")
                    success, result_dict = evaluate_instance(t_num, base_method_name, code_dir, base_dir)
                    
                    if success:
                        total_processed += 1
                        # 更新已存在的结果集合
                        existing_names.add(base_method_name)
                        
                        # 如果 result_dict 为 None，尝试从文件读取
                        if result_dict is None:
                            # 重新加载结果，获取刚写入的基础方法结果
                            _, updated_results_dict = load_existing_results(eval_jsonl_path)
                            if base_method_name in updated_results_dict:
                                result_dict = updated_results_dict[base_method_name]
                                results_dict[base_method_name] = result_dict
                        
                        if result_dict:
                            results_dict[base_method_name] = result_dict
                            recallak_values = result_dict.get('recallak', [])
                            print(f"✓ 完成: t{t_num}/{base_method_name}.json")
                            print(f"  评估结果: {base_method_name}")
                            print(f"  Recall@K: {recallak_values}")
                        else:
                            print(f"✓ 完成: t{t_num}/{base_method_name}.json (结果已写入，但无法读取)")
                    else:
                        total_failed += 1
                        print(f"✗ 失败: t{t_num}/{base_method_name}.json")
                else:
                    print(f"跳过 t{t_num}/{base_method_name}.json (结果已存在: name={base_method_name} 且存在recallak键)")
                    total_skipped += 1
            
            # 然后处理变体方法（suffix='1', '2', ..., '9'，即 a1, a2, f1, f2 等）
            for suffix in method_suffixes[1:]:  # 跳过空字符串，因为基础方法已经处理过了
                method_name = prefix + suffix
                json_file = os.path.join(t_dir, f'{method_name}.json')
                
                # 检查文件是否存在
                if not check_file_exists(json_file):
                    continue
                
                # 检查结果是否已存在
                if method_name in existing_names:
                    print(f"跳过 t{t_num}/{method_name}.json (结果已存在: name={method_name} 且存在recallak键)")
                    total_skipped += 1
                    continue
                
                # 直接复制基础方法的结果
                base_name = prefix  # 对应的基础名称（a 或 f）
                
                # 检查基础名称（a 或 f）是否有结果
                if base_name in results_dict:
                    base_result = results_dict[base_name]
                    # 复制结果，只修改 name 键
                    new_result = base_result.copy()
                    new_result['name'] = method_name
                    
                    # 写入结果
                    with open(eval_jsonl_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(new_result, ensure_ascii=False) + '\n')
                    
                    # 更新已存在的结果集合
                    existing_names.add(method_name)
                    results_dict[method_name] = new_result
                    
                    recallak_values = new_result.get('recallak', [])
                    print(f"✓ 完成: t{t_num}/{method_name}.json (复制自 {base_name})")
                    print(f"  评估结果: {method_name}")
                    print(f"  Recall@K: {recallak_values}")
                    total_processed += 1
                else:
                    print(f"跳过 t{t_num}/{method_name}.json (基础结果 {base_name} 不存在，无法复制)")
                    total_skipped += 1
    
    # 打印总结
    print(f"\n{'='*60}")
    print("处理完成！")
    print(f"{'='*60}")
    print(f"成功处理: {total_processed} 个实例")
    print(f"跳过（已存在）: {total_skipped} 个实例")
    print(f"处理失败: {total_failed} 个实例")
    print(f"总计: {total_processed + total_skipped + total_failed} 个实例")

if __name__ == '__main__':
    main()

