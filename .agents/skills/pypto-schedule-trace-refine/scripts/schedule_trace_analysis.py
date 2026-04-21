#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

def analyze_schedule_trace(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    freq = 50  # 50 MHz, each tick = 0.02 us
    tick_to_us = 1.0 / freq
    
    # Group by coreType
    core_stats = defaultdict(lambda: defaultdict(list))
    
    for entry in data:
        core_type = entry['coreType']
        tasks = entry['tasks']
        
        # Calculate task durations (each task's duration is from its start to its end)
        # In the JSON, 'end' is the end timestamp, we need to calculate duration between consecutive tasks
        for i in range(len(tasks) - 1):
            current_task = tasks[i]
            next_task = tasks[i + 1]
            
            start_time = current_task['end']
            end_time = next_task['end']
            duration = (end_time - start_time) * tick_to_us
            
            # Use next_task's name as the stage name
            name = next_task['name']
            # Extract stage name (remove suffix like _0, _0(0))
            stage_name = name.rsplit('_', 1)[0] if '_' in name else name
            if '(' in stage_name:
                stage_name = stage_name.rsplit('(', 1)[0].rstrip('_')
            
            core_stats[core_type][stage_name].append(duration)
    
    # Calculate statistics
    print("=" * 100)
    print("Machine 调度性能细化打点分析报告")
    print("=" * 100)
    print(f"\n数据文件: {json_file}")
    print(f"时间单位: 微秒 (us), 频率: {freq} MHz")
    print(f"说明: 每个阶段的耗时 = 该阶段结束时间 - 前一阶段结束时间")
    print("\n")
    
    # Focus on AICPU cores for schedule analysis
    sched_cores = ['AICPU-SCHED', 'AICPU-CTRL']
    
    print("=" * 100)
    print("AICPU 调度核心阶段耗时分析 (Level 0: 粗粒度)")
    print("=" * 100)
    
    total_schedule_time = 0
    
    for core_type in sched_cores:
        if core_type not in core_stats:
            continue
        
        # Get entries for this core_type
        core_entries = [e for e in data if e['coreType'] == core_type]
        
        print(f"\n{core_type} (共 {len(core_entries)} 个实例):")
        print("-" * 90)
        print(f"{'阶段名称':<45} {'平均(us)':<12} {'最大(us)':<12} {'最小(us)':<12} {'次数':<8}")
        print("-" * 90)
        
        core_total = 0
        stage_details = []
        for stage, durations in sorted(core_stats[core_type].items()):
            durations_pos = [d for d in durations if d >= 0]  # Filter out negative values
            if not durations_pos:
                continue
            avg = sum(durations_pos) / len(durations_pos)
            max_d = max(durations_pos)
            min_d = min(durations_pos)
            count = len(durations_pos)
            core_total += avg
            
            stage_details.append((stage, avg, max_d, min_d, count))
            print(f"{stage:<45} {avg:<12.2f} {max_d:<12.2f} {min_d:<12.2f} {count:<8}")
        
        total_schedule_time += core_total
        print("-" * 90)
        print(f"{'核心实例平均总耗时':<45} {core_total:<12.2f}")
    
    # Identify bottleneck
    print("\n")
    print("=" * 100)
    print("性能瓶颈定位 (耗时最长阶段)")
    print("=" * 100)
    
    all_stages = []
    for core_type in sched_cores:
        if core_type in core_stats:
            for stage, durations in core_stats[core_type].items():
                durations_pos = [d for d in durations if d >= 0]
                if not durations_pos:
                    continue
                avg = sum(durations_pos) / len(durations_pos)
                all_stages.append((core_type, stage, avg, len(durations_pos)))
    
    # Sort by average duration
    all_stages.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate total for percentage
    grand_total = sum(s[2] for s in all_stages)
    
    print(f"\n耗时最长的阶段 (Top 10):")
    print(f"{'排名':<5} {'核心类型':<20} {'阶段名称':<35} {'平均耗时(us)':<15} {'占比':<10}")
    print("-" * 90)
    for i, (core, stage, avg, count) in enumerate(all_stages[:10], 1):
        pct = avg / grand_total * 100 if grand_total > 0 else 0
        print(f"{i:<5} {core:<20} {stage:<35} {avg:<15.2f} {pct:<10.1f}%")
    
    # AICore compute time analysis
    print("\n")
    print("=" * 100)
    print("AICore/AIV 计算耗时分析")
    print("=" * 100)
    
    aicore_types = ['SCHED1-AIC', 'SCHED2-AIC', 'SCHED3-AIC', 'SCHED1-AIV', 'SCHED2-AIV', 'SCHED3-AIV']
    
    total_compute_time = 0
    compute_details = []
    
    for core_type in aicore_types:
        if core_type not in core_stats:
            continue
        
        core_avg = 0
        for stage, durations in sorted(core_stats[core_type].items()):
            durations_pos = [d for d in durations if d >= 0]
            if not durations_pos:
                continue
            avg = sum(durations_pos) / len(durations_pos)
            core_avg += avg
        
        total_compute_time += core_avg
        compute_details.append((core_type, core_avg))
    
    print(f"\n{'核心类型':<20} {'平均耗时(us)':<15}")
    print("-" * 40)
    for core, avg in compute_details:
        print(f"{core:<20} {avg:<15.2f}")
    print("-" * 40)
    print(f"{'计算总耗时':<20} {total_compute_time:<15.2f}")
    
    # Schedule vs compute ratio
    print("\n")
    print("=" * 100)
    print("调度与计算耗时对比")
    print("=" * 100)
    print(f"\n{'指标':<30} {'耗时(us)':<15} {'占比':<15}")
    print("-" * 60)
    total = total_schedule_time + total_compute_time
    print(f"{'AICPU 调度总耗时':<30} {total_schedule_time:<15.2f} {total_schedule_time/total*100:<15.1f}%")
    print(f"{'AICore/AIV 计算总耗时':<30} {total_compute_time:<15.2f} {total_compute_time/total*100:<15.1f}%")
    print(f"{'算子总执行时间':<30} {total:<15.2f} {'100.0%':<15}")
    
    # Variance analysis (波动分析)
    print("\n")
    print("=" * 100)
    print("波动分析 (变化系数 CV)")
    print("=" * 100)
    print("说明: CV = 标准差/平均值 × 100%, CV > 30% 表示高波动")
    
    high_variance_stages = []
    for core_type in sched_cores:
        if core_type in core_stats:
            for stage, durations in core_stats[core_type].items():
                durations_pos = [d for d in durations if d >= 0]
                if len(durations_pos) > 1:
                    avg = sum(durations_pos) / len(durations_pos)
                    variance = sum((d - avg) ** 2 for d in durations_pos) / len(durations_pos)
                    std_dev = variance ** 0.5
                    cv = std_dev / avg * 100 if avg > 0 else 0
                    high_variance_stages.append((core_type, stage, avg, std_dev, cv, len(durations_pos)))
    
    if high_variance_stages:
        print(f"\n{'核心类型':<20} {'阶段名称':<35} {'平均(us)':<12} {'标准差(us)':<12} {'CV(%)':<10} {'次数':<8}")
        print("-" * 100)
        # Sort by CV descending
        for core, stage, avg, std, cv, count in sorted(high_variance_stages, key=lambda x: x[4], reverse=True):
            volatility = "⭐⭐⭐" if cv > 50 else "⭐⭐" if cv > 30 else "⭐" if cv > 10 else ""
            print(f"{core:<20} {stage:<35} {avg:<12.2f} {std:<12.2f} {cv:<10.1f} {count:<8} {volatility}")
    
    # Detailed DEV_TASK_SCHED_EXEC analysis
    print("\n")
    print("=" * 100)
    print("DEV_TASK_SCHED_EXEC 详细分析")
    print("=" * 100)
    
    sched_exec_found = False
    for core_type in ['AICPU-SCHED']:
        if core_type in core_stats:
            for stage, durations in core_stats[core_type].items():
                if 'DEV_TASK_SCHED_EXEC' in stage or 'SCHED_EXEC' in stage:
                    durations_pos = [d for d in durations if d >= 0]
                    if durations_pos:
                        sched_exec_found = True
                        avg = sum(durations_pos) / len(durations_pos)
                        pct = avg / total_schedule_time * 100 if total_schedule_time > 0 else 0
                        print(f"\nDEV_TASK_SCHED_EXEC 平均耗时: {avg:.2f} us")
                        print(f"占调度总耗时比例: {pct:.1f}%")
                        print(f"调用次数: {len(durations_pos)}")
                        
                        if pct > 20:
                            print("\n⚠️ DEV_TASK_SCHED_EXEC 耗时占比 > 20%, 建议细化打点")
                            print("建议新增的细化打点事件:")
                            print("  1. DEV_TASK_DISPATCH_RESOLVE_DEP - 解依赖阶段")
                            print("  2. DEV_TASK_DISPATCH_NORMAL_SEND - 正常任务下发")
                            print("  3. DEV_TASK_SEND_BATCH_SEND - 批量发送")
                            print("  4. DEV_TASK_RESOLVE_RELEASE_CORE - 释放核心")
                            print("  5. DEV_TASK_RESOLVE_PUSH_READYQUE - 推入队列")
    
    if not sched_exec_found:
        print("\nDEV_TASK_SCHED_EXEC 数据未找到或耗时为 0")
    
    # Optimization suggestions
    print("\n")
    print("=" * 100)
    print("优化建议汇总")
    print("=" * 100)
    
    top_bottleneck = all_stages[0] if all_stages else None
    if top_bottleneck:
        core, stage, avg, count = top_bottleneck
        pct = avg / grand_total * 100 if grand_total > 0 else 0
        print(f"\n⭐ 主要瓶颈: {core}/{stage}")
        print(f"   平均耗时: {avg:.2f} us, 占比: {pct:.1f}%")
        
        suggestions = {
            'DEV_TASK_BUILD': [
                '优化任务构建流程，减少内核编译开销',
                '缓存编译结果，避免重复编译',
                '预编译常用算子模板'
            ],
            'INIT': [
                '优化初始化流程，减少资源分配开销',
                '预分配资源池，减少动态分配',
                '延迟初始化非必要资源'
            ],
            'DEV_TASK_RCV': [
                '优化任务接收流程，减少通信开销',
                '使用更高效的通信机制',
                '减少任务接收等待时间'
            ],
            'ALLOC_THREAD_ID': [
                '优化线程ID分配流程',
                '预分配线程ID池',
                '使用更快的分配算法'
            ],
            'CORE_HAND_SHAKE': [
                '优化核心握手流程',
                '减少握手等待时间',
                '使用更高效的核心通信机制'
            ]
        }
        
        # Match stage name pattern
        stage_key = None
        for key in suggestions:
            if key in stage:
                stage_key = key
                break
        
        if stage_key:
            print("   优化建议:")
            for i, sug in enumerate(suggestions[stage_key], 1):
                print(f"     {i}. {sug}")
    
    # High volatility optimization
    high_volatility = [v for v in high_variance_stages if v[4] > 30]
    if high_volatility:
        print(f"\n⚠️ 高波动阶段 (CV > 30%):")
        for core, stage, avg, std, cv, count in high_volatility:
            print(f"   {core}/{stage}: CV={cv:.1f}%")
            print("   可能原因: 任务数量波动、并发竞争、硬件延迟波动")
            print("   建议: 稳定任务调度策略、减少锁竞争、优化数据局部性")
    
    # Summary report
    print("\n")
    print("=" * 100)
    print("分析总结")
    print("=" * 100)
    print(f"\n1. 调度总耗时: {total_schedule_time:.2f} us")
    print(f"2. 计算总耗时: {total_compute_time:.2f} us")
    print(f"3. 调度占比: {total_schedule_time/(total_schedule_time+total_compute_time)*100:.1f}%")
    
    if total_schedule_time/(total_schedule_time+total_compute_time)*100 > 20:
        print("\n⚠️ 调度耗时占比 > 20%, 需要优化调度流程")
    
    print("\n4. 主要瓶颈: " + (all_stages[0][1] if all_stages else "未知"))
    print("5. 波动分析: " + ("存在高波动阶段" if high_volatility else "波动正常"))
    
    print("\n建议下一步操作:")
    if top_bottleneck and 'DEV_TASK_BUILD' in top_bottleneck[1]:
        print("  - 优化 DEV_TASK_BUILD 流程（任务构建）")
    else:
        print("  - 细化打点 DEV_TASK_SCHED_EXEC 内部流程")
        print("  - 分析 ResolveDepForAllAiCore 和 TryBatchSendTask 函数")
    
    print("\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 schedule_trace_analysis.py <json_file>")
        sys.exit(1)
    
    analyze_schedule_trace(sys.argv[1])