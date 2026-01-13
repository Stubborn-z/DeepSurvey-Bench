#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

def find_token_stage(token_data, time_stage_name):
    """在 token 数据中查找对应的阶段名称（处理拼写差异）"""
    # 精确匹配
    if time_stage_name in token_data:
        return time_stage_name
    
    # 模糊匹配：处理常见的拼写差异
    # 例如 "generate tabel" vs "generate table"
    normalized_time = time_stage_name.lower().replace('tabel', 'table')
    
    for token_stage in token_data.keys():
        normalized_token = token_stage.lower().replace('tabel', 'table')
        if normalized_time == normalized_token:
            return token_stage
    
    # 如果找不到，返回 None
    return None

def extract_stage_data(data):
    """从 JSON 数据中提取各阶段的数据"""
    stage_data = {}
    
    # 获取所有阶段名称（从 time 字段）
    if 'time' not in data:
        return {}
    
    stages = list(data['time'].keys())
    token_data = data.get('token', {})
    
    for stage in stages:
        # 获取 duration
        duration = None
        if stage in data.get('time', {}):
            duration = data['time'][stage].get('duration')
        
        # 获取 input_tokens 和 output_tokens
        input_tokens = 0
        output_tokens = 0
        
        # 查找对应的 token 阶段（处理拼写差异）
        token_stage = find_token_stage(token_data, stage)
        
        if token_stage and token_stage in token_data:
            # 可能有多个模型，累加所有模型的 tokens
            for model_name, model_data in token_data[token_stage].items():
                input_tokens += model_data.get('input_tokens', 0)
                output_tokens += model_data.get('output_tokens', 0)
        
        if duration is not None:
            stage_data[stage] = {
                'duration': duration,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
    
    return stage_data

def get_pricing(json_type):
    """根据 JSON 类型获取价目表
    价目表：
        null    1     2
    In  0.0025  0.0008 0.00028
    out 0.01    0.004  0.00111
    
    映射关系：
    - ac, fc, xc -> null
    - ac1, fc1, xc1 -> 1
    - ac2, fc2, xc2 -> 2
    """
    if json_type in ['ac', 'fc', 'xc']:
        return {'in': 0.0025, 'out': 0.01}  # null
    elif json_type in ['ac1', 'fc1', 'xc1']:
        return {'in': 0.0008, 'out': 0.004}  # 1
    elif json_type in ['ac2', 'fc2', 'xc2']:
        return {'in': 0.00028, 'out': 0.00111}  # 2
    else:
        # 默认使用 null
        return {'in': 0.0025, 'out': 0.01}

def process_cost_averages():
    """处理所有 tM 目录下的 9 种 JSON 文件，计算平均值"""
    # 获取脚本所在目录的父目录（EVAL）
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # EVAL 目录
    out_dir = base_dir / 'OUT'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = out_dir / 'AVGC.jsonl'
    
    # 删除已存在的文件，重新生成
    if output_file.exists():
        output_file.unlink()
        print(f"已删除旧文件: {output_file}")
    
    # 9 种 JSON 文件类型
    json_types = ['ac', 'ac1', 'ac2', 'fc', 'fc1', 'fc2', 'xc', 'xc1', 'xc2']
    
    print("=" * 60)
    print("处理成本统计...")
    print("=" * 60)
    
    # 收集所有数据：{json_type: {stage: {metric: [values]}}}
    all_data = {json_type: defaultdict(lambda: defaultdict(list)) for json_type in json_types}
    
    # 遍历所有 tM 目录
    for M in range(1, 21):
        t_dir = base_dir / f't{M}'
        
        if not t_dir.exists():
            print(f"  t{M}: 目录不存在")
            continue
        
        print(f"\n处理 t{M}...")
        
        for json_type in json_types:
            json_file = t_dir / f'{json_type}.json'
            
            if not json_file.exists():
                print(f"    {json_type}.json: 文件不存在")
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stage_data = extract_stage_data(data)
                
                if stage_data:
                    for stage, metrics in stage_data.items():
                        all_data[json_type][stage]['duration'].append(metrics['duration'])
                        all_data[json_type][stage]['input_tokens'].append(metrics['input_tokens'])
                        all_data[json_type][stage]['output_tokens'].append(metrics['output_tokens'])
                    
                    print(f"    ✓ {json_type}.json: 找到 {len(stage_data)} 个阶段")
                else:
                    print(f"    ✗ {json_type}.json: 未找到有效数据")
            
            except json.JSONDecodeError as e:
                print(f"    ✗ {json_type}.json: JSON 解析错误 - {e}")
            except Exception as e:
                print(f"    ✗ {json_type}.json: 读取错误 - {e}")
    
    # 计算平均值并生成结果
    print("\n" + "=" * 60)
    print("计算平均值...")
    print("=" * 60)
    
    results = []
    
    for json_type in json_types:
        print(f"\n处理 {json_type}...")
        
        if not all_data[json_type]:
            print(f"  跳过: 无数据")
            continue
        
        cost_data = {}
        
        # 获取所有阶段（按顺序）
        stages = sorted(all_data[json_type].keys())
        
        # 获取价目表
        pricing = get_pricing(json_type)
        
        for stage in stages:
            duration_values = all_data[json_type][stage]['duration']
            input_tokens_values = all_data[json_type][stage]['input_tokens']
            output_tokens_values = all_data[json_type][stage]['output_tokens']
            
            if not duration_values:
                continue
            
            # 计算平均值
            avg_duration = sum(duration_values) / len(duration_values)
            avg_input_tokens = sum(input_tokens_values) / len(input_tokens_values) if input_tokens_values else 0
            avg_output_tokens = sum(output_tokens_values) / len(output_tokens_values) if output_tokens_values else 0
            
            # 计算成本：input_tokens * in_price / 1000, output_tokens * out_price / 1000
            input_cost = avg_input_tokens * pricing['in'] / 1000
            output_cost = avg_output_tokens * pricing['out'] / 1000
            
            # 格式：[duration, input_tokens, output_tokens, input_cost, output_cost]
            cost_data[stage] = [avg_duration, avg_input_tokens, avg_output_tokens, input_cost, output_cost]
            
            print(f"  {stage}:")
            print(f"    文件数: {len(duration_values)}")
            print(f"    平均值: [duration={avg_duration:.2f}, input_tokens={avg_input_tokens:.2f}, output_tokens={avg_output_tokens:.2f}]")
            print(f"    成本: [input_cost={input_cost:.4f}, output_cost={output_cost:.4f}] (价目: in={pricing['in']}, out={pricing['out']})")
        
        if cost_data:
            result = {
                "name": json_type,
                "cost": cost_data
            }
            results.append(result)
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果...")
    print("=" * 60)
    
    if results:
        # 写入模式（重新生成）
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 追加价目表数据
            pricing_table = [
                {"name": "null", "in": 0.0025, "out": 0.01},
                {"name": "1", "in": 0.0008, "out": 0.004},
                {"name": "2", "in": 0.00028, "out": 0.00111}
            ]
            for pricing in pricing_table:
                f.write(json.dumps(pricing, ensure_ascii=False) + '\n')
        
        print(f"✓ 已生成 {len(results)} 条结果到 {output_file}")
        for result in results:
            print(f"  {result['name']}: {len(result['cost'])} 个阶段")
        print(f"✓ 已追加 3 条价目表数据")
    else:
        print("✗ 没有可保存的结果")

if __name__ == '__main__':
    process_cost_averages()

