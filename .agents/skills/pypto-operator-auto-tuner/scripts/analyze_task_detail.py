#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyPTO 算子任务详细分析脚本

功能：
1. 查找任务对应的代码行（通过 leafHash 或 opmagic）
2. 分析 AIV/AIC 核上的任务
3. 分析 COPY_IN/COPY_OUT tensor 流动
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_tid_to_core_name(swimlane):
    """获取 tid 到核名称的映射"""
    tid_to_name = {}
    for event in swimlane.get('traceEvents', []):
        if event.get('name') == 'thread_name':
            tid = event.get('tid')
            args = event.get('args', {})
            thread_name = args.get('name', '')
            tid_to_name[tid] = thread_name
    return tid_to_name


def get_core_tids(swimlane):
    """获取 AIC 和 AIV 核的 tid"""
    tid_to_name = get_tid_to_core_name(swimlane)
    aic_tids = {tid for tid, name in tid_to_name.items() if 'AIC' in name}
    aiv_tids = {tid for tid, name in tid_to_name.items() if 'AIV' in name}
    return aic_tids, aiv_tids, tid_to_name


def build_opmagic_to_op(program):
    """构建 opmagic -> operation 的映射"""
    opmagic_to_op = {}
    for func in program.get('functions', []):
        for op in func.get('operations', []):
            opmagic = op.get('opmagic')
            if opmagic:
                opmagic_to_op[opmagic] = {
                    'opcode': op.get('opcode', 'UNKNOWN'),
                    'file': op.get('file', 'N/A'),
                    'line': op.get('line', 'N/A'),
                    'ioperands': op.get('ioperands', []),
                    'ooperands': op.get('ooperands', []),
                    'func_name': func.get('rawname', 'N/A')
                }
    return opmagic_to_op


def find_task_code_location(swimlane, program, task_name):
    """
    根据任务名称找到对应的代码行

    步骤：
    1. 在 merged_swimlane.json 中搜索节点 name
    2. 在该节点所在对象中，查找键 "args"，在 "event-hint" 中查找 leafHash 或 callOpMagic 值
    3. 在 program.json 中检索该值，找到对应的 operation
    4. 查看 "opcode"、"file"、"line" 等键的值

    Args:
        swimlane: merged_swimlane.json 数据
        program: program.json 数据
        task_name: 任务名称，如 "0-1-38-23-4()"

    Returns:
        dict: 包含 opcode, file, line 等信息
    """
    # 查找任务
    task_event = None
    for event in swimlane.get('traceEvents', []):
        if event.get('name') == task_name:
            task_event = event
            break

    if not task_event:
        return {'error': f'未找到任务: {task_name}'}

    # 提取 opmagic
    args = task_event.get('args', {})
    event_hint = args.get('event-hint', '')

    # 优先使用 callOpMagic
    match = re.search(r'callOpMagic:(\d+)', event_hint)
    if match:
        opmagic = int(match.group(1))
    else:
        # 尝试 leafHash
        match = re.search(r'leafHash:(\d+)', event_hint)
        if match:
            opmagic = int(match.group(1))
        else:
            return {'error': '未找到 opmagic 或 leafHash'}

    # 在 program.json 中查找
    opmagic_to_op = build_opmagic_to_op(program)
    op_info = opmagic_to_op.get(opmagic, {})

    if not op_info:
        return {'error': f'未找到 opmagic: {opmagic}'}

    return {
        'task_name': task_name,
        'opmagic': opmagic,
        'opcode': op_info.get('opcode', 'UNKNOWN'),
        'file': op_info.get('file', 'N/A'),
        'line': op_info.get('line', 'N/A'),
        'func_name': op_info.get('func_name', 'N/A')
    }


def analyze_core_tasks(swimlane, program, core_type='AIV'):
    """
    分析指定类型核上的任务

    步骤：
    1. 在 merged_swimlane.json 查找 "name": "AIC_X" 或 "name": "AIV_X"
    2. 找到该 json 对象下的 "tid" 及对应的标号
    3. 查找 "tid": XXX 即可找到对应的任务对象

    Args:
        swimlane: merged_swimlane.json 数据
        program: program.json 数据
        core_type: 核类型，'AIC' 或 'AIV'

    Returns:
        dict: 包含任务统计信息
    """
    aic_tids, aiv_tids, tid_to_name = get_core_tids(swimlane)
    target_tids = aiv_tids if core_type == 'AIV' else aic_tids

    opmagic_to_op = build_opmagic_to_op(program)

    # 收集任务
    tasks = []
    opcodes = []

    for event in swimlane.get('traceEvents', []):
        tid = event.get('tid')
        if tid in target_tids:
            name = event.get('name', '')
            if '()' in name:
                args = event.get('args', {})
                event_hint = args.get('event-hint', '')

                # 提取 opmagic
                match = re.search(r'callOpMagic:(\d+)', event_hint)
                if match:
                    opmagic = int(match.group(1))
                    op_info = opmagic_to_op.get(opmagic, {})
                    opcode = op_info.get('opcode', 'UNKNOWN')

                    opcodes.append(opcode)
                    tasks.append({
                        'task_name': name,
                        'core': tid_to_name.get(tid, 'UNKNOWN'),
                        'opmagic': opmagic,
                        'opcode': opcode,
                        'file': op_info.get('file', 'N/A'),
                        'line': op_info.get('line', 'N/A')
                    })

    # 统计
    opcode_counter = Counter(opcodes)

    return {
        'core_type': core_type,
        'core_count': len(target_tids),
        'task_count': len(tasks),
        'opcode_distribution': dict(opcode_counter.most_common(20)),
        'tasks': tasks
    }


def analyze_copy_flow(program):
    """
    分析 COPY_IN/COPY_OUT tensor 流动

    检查是否存在 tensor 被 COPY_OUT 后又被 COPY_IN，
    这表示数据被拷贝出后再拷贝入，可能存在冗余搬运。

    Args:
        program: program.json 数据

    Returns:
        dict: 包含 COPY 流动分析结果
    """
    # 收集 COPY_IN 和 COPY_OUT 操作
    copy_in_ops = []
    copy_out_ops = []

    for func in program.get('functions', []):
        for op in func.get('operations', []):
            opcode = op.get('opcode', '')
            if opcode == 'COPY_IN':
                copy_in_ops.append({
                    'func': func.get('rawname', 'N/A'),
                    'opmagic': op.get('opmagic'),
                    'ioperands': op.get('ioperands', []),
                    'ooperands': op.get('ooperands', [])
                })
            elif opcode == 'COPY_OUT':
                copy_out_ops.append({
                    'func': func.get('rawname', 'N/A'),
                    'opmagic': op.get('opmagic'),
                    'ioperands': op.get('ioperands', []),
                    'ooperands': op.get('ooperands', [])
                })

    # 构建 tensor -> COPY_OUT 映射
    tensor_to_copy_out = defaultdict(list)
    for op in copy_out_ops:
        for tensor_id in op['ooperands']:
            tensor_to_copy_out[tensor_id].append(op)

    # 查找 COPY_IN 的输入是否来自 COPY_OUT
    copy_in_from_copy_out = []
    for op in copy_in_ops:
        for tensor_id in op['ioperands']:
            if tensor_id in tensor_to_copy_out:
                for copy_out_op in tensor_to_copy_out[tensor_id]:
                    copy_in_from_copy_out.append({
                        'tensor_id': tensor_id,
                        'copy_out': copy_out_op,
                        'copy_in': op
                    })

    # 统计涉及的 tensor
    tensors_involved = set(item['tensor_id'] for item in copy_in_from_copy_out)

    return {
        'copy_in_count': len(copy_in_ops),
        'copy_out_count': len(copy_out_ops),
        'copy_in_from_copy_out_count': len(copy_in_from_copy_out),
        'tensors_involved': len(tensors_involved),
        'details': copy_in_from_copy_out[:10],  # 只返回前 10 个
        'recommendation': '如果 COPY_IN_FROM_COPY_OUT 数量较多，建议：\n'
                         '1. 开启合图优化 (sg_set_scope)\n'
                         '2. 调整 TileShape 减少子图数量\n'
                         '3. 重新设计算子结构'
    }


def analyze_opcode_sequence(swimlane, program, core_name='AIV_21'):
    """
    分析某个核上的操作序列，查找连续的 COPY_OUT/COPY_IN

    Args:
        swimlane: merged_swimlane.json 数据
        program: program.json 数据
        core_name: 核名称，如 'AIV_21'

    Returns:
        dict: 包含操作序列分析
    """
    # 获取核的 tid
    tid_to_name = get_tid_to_core_name(swimlane)
    name_to_tid = {v: k for k, v in tid_to_name.items()}
    target_tid = name_to_tid.get(core_name)

    if not target_tid:
        return {'error': f'未找到核: {core_name}'}

    opmagic_to_op = build_opmagic_to_op(program)

    # 收集该核上的任务（按时间顺序）
    tasks = []
    for event in swimlane.get('traceEvents', []):
        if event.get('tid') == target_tid:
            name = event.get('name', '')
            if '()' in name:
                ts = event.get('ts', 0)
                args = event.get('args', {})
                event_hint = args.get('event-hint', '')

                match = re.search(r'callOpMagic:(\d+)', event_hint)
                if match:
                    opmagic = int(match.group(1))
                    op_info = opmagic_to_op.get(opmagic, {})

                    tasks.append({
                        'ts': ts,
                        'task_name': name,
                        'opmagic': opmagic,
                        'opcode': op_info.get('opcode', 'UNKNOWN')
                    })

    # 按时间排序
    tasks.sort(key=lambda x: x['ts'])

    # 查找连续的 COPY_OUT -> COPY_IN
    sequences = []
    for i in range(len(tasks) - 1):
        curr_opcode = tasks[i]['opcode']
        next_opcode = tasks[i + 1]['opcode']

        if curr_opcode == 'COPY_OUT' and next_opcode == 'COPY_IN':
            sequences.append({
                'copy_out_task': tasks[i]['task_name'],
                'copy_in_task': tasks[i + 1]['task_name'],
                'gap': tasks[i + 1]['ts'] - tasks[i]['ts'] if tasks[i + 1]['ts'] and tasks[i]['ts'] else 0
            })

    return {
        'core_name': core_name,
        'task_count': len(tasks),
        'copy_out_copy_in_sequence_count': len(sequences),
        'sequences': sequences[:10],
        'all_tasks': [{'task_name': t['task_name'], 'opcode': t['opcode']} for t in tasks]
    }


def main(output_dir):
    """主函数"""
    output_path = Path(output_dir)

    # 加载文件
    swimlane_path = output_path / 'merged_swimlane.json'
    program_path = output_path / 'program.json'

    if not swimlane_path.exists() or not program_path.exists():
        print(f"错误: 找不到文件 {swimlane_path} 或 {program_path}")
        return

    swimlane = load_json(swimlane_path)
    program = load_json(program_path)

    print("=" * 60)
    print("PyPTO 算子任务详细分析")
    print("=" * 60)

    # 1. 分析 AIV 核任务
    print("\n【1. AIV 核任务分析】")
    aiv_result = analyze_core_tasks(swimlane, program, 'AIV')
    print(f"AIV 核数量: {aiv_result['core_count']}")
    print(f"AIV 核任务总数: {aiv_result['task_count']}")
    print(f"\nOpcode 分布:")
    for opcode, count in aiv_result['opcode_distribution'].items():
        print(f"  {opcode}: {count}")

    # 2. 分析 AIC 核任务
    print("\n【2. AIC 核任务分析】")
    aic_result = analyze_core_tasks(swimlane, program, 'AIC')
    print(f"AIC 核数量: {aic_result['core_count']}")
    print(f"AIC 核任务总数: {aic_result['task_count']}")
    print(f"\nOpcode 分布:")
    for opcode, count in aic_result['opcode_distribution'].items():
        print(f"  {opcode}: {count}")

    # 3. 分析 COPY 流动
    print("\n【3. COPY_IN/COPY_OUT 分析】")
    copy_result = analyze_copy_flow(program)
    print(f"COPY_IN 数量: {copy_result['copy_in_count']}")
    print(f"COPY_OUT 数量: {copy_result['copy_out_count']}")
    print(f"从 COPY_OUT 到 COPY_IN 的流动: {copy_result['copy_in_from_copy_out_count']}")
    print(f"涉及的 tensor 数量: {copy_result['tensors_involved']}")

    if copy_result['copy_in_from_copy_out_count'] > 100:
        print(f"\n⚠️ 警告: 存在大量冗余搬运!")
        print(copy_result['recommendation'])

    # 4. 分析某个核的操作序列
    print("\n【4. AIV_21 核操作序列分析】")
    seq_result = analyze_opcode_sequence(swimlane, program, 'AIV_21')
    if 'error' not in seq_result:
        print(f"任务数: {seq_result['task_count']}")
        print(f"连续 COPY_OUT -> COPY_IN 次数: {seq_result['copy_out_copy_in_sequence_count']}")

        if seq_result['all_tasks']:
            print(f"\n操作序列:")
            for t in seq_result['all_tasks'][:10]:
                print(f"  {t['task_name']}: {t['opcode']}")

    # 5. 示例：查找某个任务的代码位置
    print("\n【5. 任务代码位置查找示例】")
    if aiv_result['tasks']:
        example_task = aiv_result['tasks'][0]['task_name']
        location = find_task_code_location(swimlane, program, example_task)
        if 'error' not in location:
            print(f"任务: {location['task_name']}")
            print(f"Opcode: {location['opcode']}")
            print(f"文件: {location['file']}")
            print(f"行号: {location['line']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_task_detail.py <output_dir>")
        print("示例: python analyze_task_detail.py output/output_20260328_xxx")
        sys.exit(1)

    main(sys.argv[1])