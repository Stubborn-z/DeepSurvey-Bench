#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估所有 t<a>/ 下的 a<b>.json, f<b>.json, x<b>.json
通过调用 recallpref.py 实现
"""

import os
import sys

# 添加当前目录到路径，以便导入 recallpref
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recallpref


def find_all_instances(base_dir: str):
    """
    查找所有存在的实例文件
    
    Returns:
        List[Tuple[t_num, method_type, method_suffix]]
    """
    instances = []
    
    # 遍历 t1 到 t20
    for t_num in range(1, 21):
        t_dir = os.path.join(base_dir, f"t{t_num}")
        
        if not os.path.exists(t_dir) or not os.path.isdir(t_dir):
            continue
        
        # 查找所有可能的文件
        for method_type in ["a", "f", "x"]:
            # 检查无后缀的文件（如 a.json, f.json, x.json）
            json_path = os.path.join(t_dir, f"{method_type}.json")
            if os.path.exists(json_path):
                instances.append((str(t_num), method_type, ""))
            
            # 检查有后缀的文件（如 a1.json, a2.json, ...）
            for suffix in range(1, 10):
                method_name = f"{method_type}{suffix}"
                json_path = os.path.join(t_dir, f"{method_name}.json")
                if os.path.exists(json_path):
                    instances.append((str(t_num), method_type, str(suffix)))
    
    return instances


def main():
    """主函数"""
    # 获取 EVAL 目录路径
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(code_dir)
    
    print("=" * 60)
    print("批量评估 recall, precision, F1 指标")
    print("=" * 60)
    print(f"基础目录: {base_dir}")
    
    # 查找所有实例
    print("\n查找所有实例...")
    instances = find_all_instances(base_dir)
    print(f"找到 {len(instances)} 个实例")
    
    # 统计信息
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 处理每个实例
    for idx, (t_num, method_type, suffix) in enumerate(instances, 1):
        method_name = f"{method_type}{suffix}" if suffix else method_type
        print(f"\n[{idx}/{len(instances)}] 处理 t{t_num}/{method_name}")
        print("-" * 60)
        
        try:
            # 先检查是否已评估
            t_dir = os.path.join(base_dir, f"t{t_num}")
            eval_jsonl_path = os.path.join(t_dir, "eval.jsonl")
            
            if recallpref.is_already_evaluated(method_name, eval_jsonl_path):
                skip_count += 1
                print(f"  跳过: {method_name} (已评估)")
                continue
            
            result = recallpref.evaluate_single_instance(base_dir, t_num, method_type, suffix)
            
            if result:
                success_count += 1
                print(f"✓ 成功评估: t{t_num}/{method_name}")
            else:
                fail_count += 1
                print(f"✗ 评估失败: t{t_num}/{method_name}")
                
        except Exception as e:
            fail_count += 1
            print(f"✗ 处理 {method_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总计: {len(instances)} 个实例")
    print(f"  成功: {success_count} 个")
    print(f"  跳过: {skip_count} 个（已评估）")
    print(f"  失败: {fail_count} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()

