#!/usr/bin/env python3
# coding: utf-8
"""
PyPTO 泳道图性能对比工具

用于对比多个版本的性能数据，自动生成对比报告
"""
import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


def load_swimlane_json(file_path: str) -> Optional[Dict]:
    """加载泳道图JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_performance_metrics(swimlane_data: Dict) -> Dict[str, Any]:
    """从泳道图数据中提取性能指标"""
    metrics = {
        'total_time_us': 0,
        'gm_access_count': 0,
        'core_utilization': {},
        'operation_count': 0,
        'memory_usage_kb': 0
    }
    
    if not swimlane_data:
        return metrics
    
    # 提取总执行时间
    if 'total_time' in swimlane_data:
        metrics['total_time_us'] = swimlane_data['total_time']
    
    # 提取GM访问统计
    if 'gm_stats' in swimlane_data:
        metrics['gm_access_count'] = swimlane_data['gm_stats'].get('total_access', 0)
    
    # 提取核利用率
    if 'core_stats' in swimlane_data:
        for core_id, stats in swimlane_data['core_stats'].items():
            metrics['core_utilization'][core_id] = stats.get('utilization', 0)
    
    # 提取操作数量
    if 'operations' in swimlane_data:
        metrics['operation_count'] = len(swimlane_data['operations'])
    
    return metrics


def compare_versions(baseline_metrics: Dict, optimized_metrics: Dict) -> Dict[str, Any]:
    """对比两个版本的性能指标"""
    comparison = {
        'total_time': {
            'baseline': baseline_metrics['total_time_us'],
            'optimized': optimized_metrics['total_time_us'],
            'improvement_us': 0,
            'improvement_pct': 0
        },
        'gm_access': {
            'baseline': baseline_metrics['gm_access_count'],
            'optimized': optimized_metrics['gm_access_count'],
            'improvement': 0,
            'improvement_pct': 0
        },
        'operation_count': {
            'baseline': baseline_metrics['operation_count'],
            'optimized': optimized_metrics['operation_count'],
            'change': 0,
            'change_pct': 0
        }
    }
    
    # 计算总时间改进
    if baseline_metrics['total_time_us'] > 0:
        improvement_us = baseline_metrics['total_time_us'] - optimized_metrics['total_time_us']
        comparison['total_time']['improvement_us'] = improvement_us
        comparison['total_time']['improvement_pct'] = (improvement_us / baseline_metrics['total_time_us']) * 100
    
    # 计算GM访问改进
    if baseline_metrics['gm_access_count'] > 0:
        improvement = baseline_metrics['gm_access_count'] - optimized_metrics['gm_access_count']
        comparison['gm_access']['improvement'] = improvement
        comparison['gm_access']['improvement_pct'] = (improvement / baseline_metrics['gm_access_count']) * 100
    
    # 计算操作数量变化
    if baseline_metrics['operation_count'] > 0:
        change = optimized_metrics['operation_count'] - baseline_metrics['operation_count']
        comparison['operation_count']['change'] = change
        comparison['operation_count']['change_pct'] = (change / baseline_metrics['operation_count']) * 100
    
    return comparison


def generate_comparison_report(comparisons: List[Dict], version_names: List[str], output_file: str = None) -> str:
    """生成对比报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PyPTO 性能对比报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 总时间对比
    report_lines.append("## 1. 总执行时间对比")
    report_lines.append("-" * 80)
    report_lines.append(f"{'版本':<20} {'时间(us)':<15} {'改进(us)':<15} {'改进率':<15}")
    report_lines.append("-" * 80)
    
    baseline_time = comparisons[0]['total_time']['baseline']
    for i, comp in enumerate(comparisons):
        version = version_names[i]
        time_us = comp['total_time']['optimized'] if i > 0 else comp['total_time']['baseline']
        improvement = comp['total_time']['improvement_us']
        pct = comp['total_time']['improvement_pct']
        
        if i == 0:
            report_lines.append(f"{version:<20} {time_us:<15.2f} {'-':<15} {'-':<15}")
        else:
            report_lines.append(f"{version:<20} {time_us:<15.2f} {improvement:<15.2f} {pct:>13.1f}%")
    
    report_lines.append("")
    
    # GM访问对比
    report_lines.append("## 2. GM访问量对比")
    report_lines.append("-" * 80)
    report_lines.append(f"{'版本':<20} {'访问次数':<15} {'减少次数':<15} {'减少率':<15}")
    report_lines.append("-" * 80)
    
    for i, comp in enumerate(comparisons):
        version = version_names[i]
        access = comp['gm_access']['optimized'] if i > 0 else comp['gm_access']['baseline']
        improvement = comp['gm_access']['improvement']
        pct = comp['gm_access']['improvement_pct']
        
        if i == 0:
            report_lines.append(f"{version:<20} {access:<15} {'-':<15} {'-':<15}")
        else:
            report_lines.append(f"{version:<20} {access:<15} {improvement:<15} {pct:>13.1f}%")
    
    report_lines.append("")
    
    # 操作数量对比
    report_lines.append("## 3. 计算图复杂度对比")
    report_lines.append("-" * 80)
    report_lines.append(f"{'版本':<20} {'操作数量':<15} {'变化':<15} {'变化率':<15}")
    report_lines.append("-" * 80)
    
    for i, comp in enumerate(comparisons):
        version = version_names[i]
        count = comp['operation_count']['optimized'] if i > 0 else comp['operation_count']['baseline']
        change = comp['operation_count']['change']
        pct = comp['operation_count']['change_pct']
        
        if i == 0:
            report_lines.append(f"{version:<20} {count:<15} {'-':<15} {'-':<15}")
        else:
            sign = "+" if change > 0 else ""
            report_lines.append(f"{version:<20} {count:<15} {sign}{change:<14} {sign}{pct:>12.1f}%")
    
    report_lines.append("")
    
    # 性能瓶颈分析
    report_lines.append("## 4. 性能瓶颈分析")
    report_lines.append("-" * 80)
    
    if len(comparisons) > 1:
        best_improvement = comparisons[-1]['total_time']['improvement_pct']
        
        if best_improvement >= 50:
            report_lines.append("✓ 优化效果显著！达到50%+的预期目标")
        elif best_improvement >= 30:
            report_lines.append("○ 优化效果良好，建议继续探索激进优化方案")
        else:
            report_lines.append("△ 优化效果一般，建议分析泳道图找出残留瓶颈")
        
        # GM访问分析
        gm_improvement = comparisons[-1]['gm_access']['improvement_pct']
        if gm_improvement > 20:
            report_lines.append(f"✓ GM访问优化明显 ({gm_improvement:.1f}%)，Tile配置调整有效")
        elif gm_improvement > 0:
            report_lines.append(f"○ GM访问有所改善 ({gm_improvement:.1f}%)，可考虑进一步增大Tile")
        else:
            report_lines.append("△ GM访问未改善，可能需要调整cube_l1_reuse配置")
        
        # 操作数量分析
        op_change = comparisons[-1]['operation_count']['change_pct']
        if op_change < -10:
            report_lines.append(f"✓ 计算图显著简化 ({abs(op_change):.1f}%)，子图融合效果好")
        elif op_change < 0:
            report_lines.append(f"○ 计算图有所简化 ({abs(op_change):.1f}%)，融合有效")
        elif op_change > 0:
            report_lines.append(f"△ 计算图复杂度增加 ({op_change:.1f}%)，可能影响性能")
    
    report_lines.append("")
    
    # 优化建议
    report_lines.append("## 5. 进一步优化建议")
    report_lines.append("-" * 80)
    
    if len(comparisons) > 1:
        improvement = comparisons[-1]['total_time']['improvement_pct']
        
        if improvement < 30:
            report_lines.append("1. 增大s2_tile到512，测试是否进一步提升")
            report_lines.append("2. 调整cube_l1_reuse_setting为更高的复用次数")
            report_lines.append("3. 尝试不同的device_sched_mode (0, 1, 2)")
        elif improvement < 50:
            report_lines.append("1. 尝试V2激进优化配置（增大内存到8192KB）")
            report_lines.append("2. 优化循环展开策略为[16, 8, 4, 2, 1]")
            report_lines.append("3. 考虑KV组装预取优化")
        else:
            report_lines.append("1. 已达到优化目标，建议进行稳定性测试")
            report_lines.append("2. 在不同输入规模下验证性能提升")
            report_lines.append("3. 记录优化经验，更新最佳实践文档")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    # 保存报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存到: {output_file}")
    
    return report


def find_swimlane_files(output_dir: str = "output") -> List[str]:
    """查找泳道图JSON文件"""
    pattern = os.path.join(output_dir, "output_*", "merged_swimlane.json")
    files = glob.glob(pattern)
    return sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)


def interactive_compare():
    """交互式对比工具"""
    print("=" * 80)
    print("PyPTO 泳道图性能对比工具")
    print("=" * 80)
    
    # 查找可用的泳道图文件
    swimlane_files = find_swimlane_files()
    
    if not swimlane_files:
        print("\n未找到泳道图文件。请先运行性能测试生成泳道图数据。")
        print("泳道图文件应位于: output/output_*/merged_swimlane.json")
        return
    
    print(f"\n找到 {len(swimlane_files)} 个泳道图文件:")
    for i, f in enumerate(swimlane_files[:10]):  # 只显示最近10个
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        print(f"  [{i+1}] {os.path.basename(os.path.dirname(f))} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 选择baseline文件
    print("\n选择baseline版本 (输入序号):")
    try:
        baseline_idx = int(input("> ")) - 1
        if baseline_idx < 0 or baseline_idx >= len(swimlane_files):
            print("无效的序号")
            return
    except ValueError:
        print("无效的输入")
        return
    
    baseline_file = swimlane_files[baseline_idx]
    print(f"\n选择baseline: {os.path.basename(os.path.dirname(baseline_file))}")
    
    # 选择优化版本
    print("\n选择要对比的优化版本 (输入序号，可多选，用逗号分隔):")
    try:
        opt_indices = [int(x.strip()) - 1 for x in input("> ").split(",")]
        opt_files = [swimlane_files[i] for i in opt_indices if 0 <= i < len(swimlane_files)]
    except (ValueError, IndexError):
        print("无效的输入")
        return
    
    if not opt_files:
        print("未选择有效的优化版本")
        return
    
    # 加载和对比
    print("\n加载性能数据...")
    baseline_data = load_swimlane_json(baseline_file)
    baseline_metrics = extract_performance_metrics(baseline_data)
    
    comparisons = []
    version_names = ["Baseline"]
    
    for opt_file in opt_files:
        opt_data = load_swimlane_json(opt_file)
        opt_metrics = extract_performance_metrics(opt_data)
        
        comparison = compare_versions(baseline_metrics, opt_metrics)
        comparisons.append(comparison)
        version_names.append(os.path.basename(os.path.dirname(opt_file)))
    
    # 生成报告
    output_file = f"performance_comparison_{int(datetime.now().timestamp())}.txt"
    report = generate_comparison_report(comparisons, version_names, output_file)
    
    print("\n" + report)


def compare_from_files(baseline_file: str, optimized_files: List[str], output_file: str = None):
    """从指定文件进行对比"""
    # 加载baseline
    baseline_data = load_swimlane_json(baseline_file)
    baseline_metrics = extract_performance_metrics(baseline_data)
    
    comparisons = []
    version_names = ["Baseline"]
    
    # 加载优化版本
    for opt_file in optimized_files:
        opt_data = load_swimlane_json(opt_file)
        opt_metrics = extract_performance_metrics(opt_data)
        
        comparison = compare_versions(baseline_metrics, opt_metrics)
        comparisons.append(comparison)
        version_names.append(os.path.basename(os.path.dirname(opt_file)))
    
    # 生成报告
    report = generate_comparison_report(comparisons, version_names, output_file)
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == "--help":
            print("用法:")
            print("  python compare_swimlane.py                    # 交互式模式")
            print("  python compare_swimlane.py <baseline> <opt1> [opt2] ...  # 命令行模式")
            print("")
            print("示例:")
            print("  python compare_swimlane.py output/output_baseline/merged_swimlane.json \\")
            print("         output/output_v1/merged_swimlane.json \\")
            print("         output/output_v2/merged_swimlane.json")
        else:
            baseline_file = sys.argv[1]
            optimized_files = sys.argv[2:]
            output_file = f"comparison_report_{int(datetime.now().timestamp())}.txt"
            compare_from_files(baseline_file, optimized_files, output_file)
    else:
        # 交互式模式
        interactive_compare()
