#!/usr/bin/env python3
"""
PyPTO 算子性能对比脚本
用于对比优化前后的性能数据
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import argparse


class PerformanceComparator:
    """性能对比器"""

    def __init__(self, before_dir: str, after_dir: str):
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)

    def load_performance_data(self, directory: Path) -> Dict[str, Any]:
        """加载性能数据"""
        data = {}

        # 查找并加载 bubble_analysis.log
        bubble_file = self._find_file(directory, "bubble_analysis.log")
        if bubble_file:
            data['bubble'] = self._parse_bubble_file(bubble_file)

        # 查找并加载 merged_swimlane.json
        swimlane_file = self._find_file(directory, "merged_swimlane.json")
        if swimlane_file:
            data['swimlane'] = self._parse_swimlane_file(swimlane_file)

        # 查找并加载 machine_runtime_operator_trace.json
        trace_file = self._find_file(directory, "machine_runtime_operator_trace.json")
        if trace_file:
            data['trace'] = self._parse_trace_file(trace_file)

        return data

    def _find_file(self, directory: Path, filename: str) -> Path | None:
        """查找文件"""
        for root, dirs, files in os.walk(directory):
            if filename in files:
                return Path(root) / filename
        return None

    def _parse_bubble_file(self, file_path: Path) -> Dict[str, Any]:
        """解析气泡分析文件"""
        result = {
            "total_wait_time": 0.0,
            "total_work_time": 0.0,
            "bubble_rate": 0.0,
            "schedule_wait_rate": 0.0,
            "pred_wait_rate": 0.0
        }

        per_core = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_core = None
        for line in lines:
            line = line.strip()
            if line.startswith("[AIV_") or line.startswith("[AIC_"):
                core_info = line.split(']')[0][1:]
                current_core = core_info
                per_core[current_core] = {
                    "work_time": 0.0,
                    "wait_time": 0.0,
                    "wait_schedule": 0.0,
                    "wait_predecessor": 0.0
                }
            elif current_core and "Core Total Work Time:" in line:
                time_str = line.split("Core Total Work Time:")[1].strip().split()[0]
                per_core[current_core]["work_time"] = float(time_str)
            elif current_core and "Total Wait Time:" in line:
                time_str = line.split("Total Wait Time:")[1].strip().split()[0]
                per_core[current_core]["wait_time"] = float(time_str)
            elif current_core and "Wait Schedule Time:" in line:
                time_str = line.split("Wait Schedule Time:")[1].strip().split()[0]
                per_core[current_core]["wait_schedule"] = float(time_str)
            elif current_core and "Wait Predecessor Time:" in line:
                time_str = line.split("Wait Predecessor Time:")[1].strip().split()[0]
                per_core[current_core]["wait_predecessor"] = float(time_str)

        for core, data in per_core.items():
            result["total_work_time"] += data["work_time"]
            result["total_wait_time"] += data["wait_time"]

        total_time = result["total_work_time"] + result["total_wait_time"]
        if total_time > 0:
            result["bubble_rate"] = (result["total_wait_time"] / total_time) * 100

        total_schedule_wait = sum(d["wait_schedule"] for d in per_core.values())
        total_pred_wait = sum(d["wait_predecessor"] for d in per_core.values())

        if result["total_wait_time"] > 0:
            result["schedule_wait_rate"] = (total_schedule_wait / result["total_wait_time"]) * 100
            result["pred_wait_rate"] = (total_pred_wait / result["total_wait_time"]) * 100

        return result

    def _parse_swimlane_file(self, file_path: Path) -> Dict[str, Any]:
        """解析泳道图文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {
            "total_tasks": 0,
            "avg_execution_time": 0.0,
            "max_execution_time": 0.0,
            "min_execution_time": float(1e9),
            "total_execution_time": 0.0,
            "memory_usage": 0
        }

        if isinstance(data, list):
            for item in data:
                if "dur" in item:
                    dur = self._parse_time(item["dur"])
                    result["total_tasks"] += 1
                    result["total_execution_time"] += dur
                    result["max_execution_time"] = max(result["max_execution_time"], dur)
                    result["min_execution_time"] = min(result["min_execution_time"], dur)

                    if "args" in item and "ioperand-hint" in item["args"]:
                        try:
                            ioperand = json.loads(item["args"]["ioperand-hint"])
                            for tensor_info in ioperand.values():
                                if "mem_usage" in tensor_info:
                                    result["memory_usage"] += tensor_info["mem_usage"]
                        except:
                            pass

        if result["total_tasks"] > 0:
            result["avg_execution_time"] = result["total_execution_time"] / result["total_tasks"]

        return result

    def _parse_trace_file(self, file_path: Path) -> Dict[str, Any]:
        """解析性能追踪文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {
            "total_time": 0.0,
            "control_overhead": 0.0,
            "computation_time": 0.0
        }

        if isinstance(data, list):
            for item in data:
                if "dur" in item:
                    dur = self._parse_time(item["dur"])
                    result["total_time"] += dur

                    if "cat" in item:
                        if "AICPU-CTRL" in item["cat"]:
                            result["control_overhead"] += dur
                        elif "AIC" in item["cat"] or "AIV" in item["cat"]:
                            result["computation_time"] += dur

        if result["total_time"] > 0:
            result["control_overhead_ratio"] = (result["control_overhead"] / result["total_time"]) * 100
            result["computation_ratio"] = (result["computation_time"] / result["total_time"]) * 100

        return result

    def _parse_time(self, time: Any) -> float:
        """解析时间"""
        if isinstance(time, (int, float)):
            return float(time)

        time_str = str(time).lower()
        if "us" in time_str:
            return float(time_str.replace("us", ""))
        elif "ms" in time_str:
            return float(time_str.replace("ms", "")) * 1000
        elif "s" in time_str:
            return float(time_str.replace("s", "")) * 1000000
        else:
            return float(time_str)

    def compare_metrics(self, before: Dict, after: Dict) -> Dict[str, Any]:
        """对比性能指标"""
        comparison = {}

        # 对比气泡率
        if 'bubble' in before and 'bubble' in after:
            before_bubble = before['bubble']['bubble_rate']
            after_bubble = after['bubble']['bubble_rate']
            improvement = ((before_bubble - after_bubble) / before_bubble) * 100 if before_bubble > 0 else 0

            comparison['bubble_rate'] = {
                'before': before_bubble,
                'after': after_bubble,
                'improvement': improvement,
                'direction': '↓' if improvement > 0 else '↑'
            }

        # 对比控制开销占比
        if 'trace' in before and 'trace' in after:
            before_ctrl = before['trace']['control_overhead_ratio']
            after_ctrl = after['trace']['control_overhead_ratio']
            improvement = ((before_ctrl - after_ctrl) / before_ctrl) * 100 if before_ctrl > 0 else 0

            comparison['control_overhead_ratio'] = {
                'before': before_ctrl,
                'after': after_ctrl,
                'improvement': improvement,
                'direction': '↓' if improvement > 0 else '↑'
            }

        # 对比计算占比
        if 'trace' in before and 'trace' in after:
            before_comp = before['trace']['computation_ratio']
            after_comp = after['trace']['computation_ratio']
            improvement = ((after_comp - before_comp) / before_comp) * 100 if before_comp > 0 else 0

            comparison['computation_ratio'] = {
                'before': before_comp,
                'after': after_comp,
                'improvement': improvement,
                'direction': '↑' if improvement > 0 else '↓'
            }

        # 对比平均执行时间
        if 'swimlane' in before and 'swimlane' in after:
            before_time = before['swimlane']['avg_execution_time']
            after_time = after['swimlane']['avg_execution_time']
            improvement = ((before_time - after_time) / before_time) * 100 if before_time > 0 else 0

            comparison['avg_execution_time'] = {
                'before': before_time,
                'after': after_time,
                'improvement': improvement,
                'direction': '↓' if improvement > 0 else '↑'
            }

        # 对比总执行时间
        if 'trace' in before and 'trace' in after:
            before_total = before['trace']['total_time']
            after_total = after['trace']['total_time']
            improvement = ((before_total - after_total) / before_total) * 100 if before_total > 0 else 0

            comparison['total_time'] = {
                'before': before_total,
                'after': after_total,
                'improvement': improvement,
                'direction': '↓' if improvement > 0 else '↑'
            }

        return comparison

    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        before_data = self.load_performance_data(self.before_dir)
        after_data = self.load_performance_data(self.after_dir)
        comparison = self.compare_metrics(before_data, after_data)

        report = []
        report.append("=" * 80)
        report.append("PyPTO 算子性能对比报告")
        report.append("=" * 80)
        report.append("")

        report.append(f"优化前目录: {self.before_dir}")
        report.append(f"优化后目录: {self.after_dir}")
        report.append("")

        # 性能指标对比
        report.append("【性能指标对比】")
        report.append("-" * 40)
        report.append(f"{'指标':<20} {'优化前':>15} {'优化后':>15} {'改善':>15} {'趋势':>5}")
        report.append("-" * 80)

        if 'bubble_rate' in comparison:
            metric = comparison['bubble_rate']
            report.append(f"{'气泡率':<20} {metric['before']:>14.2f}% {metric['after']:>14.2f}% {metric['improvement']:>13.2f}% {metric['direction']:>5}")

        if 'control_overhead_ratio' in comparison:
            metric = comparison['control_overhead_ratio']
            report.append(f"{'控制开销占比':<20} {metric['before']:>14.2f}% {metric['after']:>14.2f}% {metric['improvement']:>13.2f}% {metric['direction']:>5}")

        if 'computation_ratio' in comparison:
            metric = comparison['computation_ratio']
            report.append(f"{'计算占比':<20} {metric['before']:>14.2f}% {metric['after']:>14.2f}% {metric['improvement']:>13.2f}% {metric['direction']:>5}")

        if 'avg_execution_time' in comparison:
            metric = comparison['avg_execution_time']
            report.append(f"{'平均执行时间':<20} {metric['before']:>14.2f}us {metric['after']:>14.2f}us {metric['improvement']:>13.2f}% {metric['direction']:>5}")

        if 'total_time' in comparison:
            metric = comparison['total_time']
            report.append(f"{'总执行时间':<20} {metric['before']:>14.2f}us {metric['after']:>14.2f}us {metric['improvement']:>13.2f}% {metric['direction']:>5}")

        report.append("")

        # 总体评估
        report.append("【总体评估】")
        report.append("-" * 40)

        improvements = [v['improvement'] for v in comparison.values()]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        if avg_improvement > 20:
            assessment = "⭐⭐⭐⭐⭐ 优化效果显著"
        elif avg_improvement > 10:
            assessment = "⭐⭐⭐⭐ 优化效果良好"
        elif avg_improvement > 5:
            assessment = "⭐⭐⭐ 优化效果一般"
        elif avg_improvement > 0:
            assessment = "⭐⭐ 优化效果轻微"
        else:
            assessment = "⭐ 优化效果不明显，建议重新评估"

        report.append(assessment)
        report.append("")

        # 详细分析
        report.append("【详细分析】")
        report.append("-" * 40)

        if 'control_overhead_ratio' in comparison:
            metric = comparison['control_overhead_ratio']
            if metric['improvement'] > 0:
                report.append(f"✅ 控制开销降低 {abs(metric['improvement']):.2f}%，表明内存访问优化有效")
            else:
                report.append(f"⚠️ 控制开销增加 {abs(metric['improvement']):.2f}%，需要检查优化是否引入了新的开销")

        if 'bubble_rate' in comparison:
            metric = comparison['bubble_rate']
            if metric['improvement'] > 0:
                report.append(f"✅ 气泡率降低 {abs(metric['improvement']):.2f}%，表明调度效率提升")
            else:
                report.append(f"⚠️ 气泡率增加 {abs(metric['improvement']):.2f}%，需要检查任务调度策略")

        if 'computation_ratio' in comparison:
            metric = comparison['computation_ratio']
            if metric['improvement'] > 0:
                report.append(f"✅ 计算占比提升 {abs(metric['improvement']):.2f}%，表明计算效率提升")
            else:
                report.append(f"⚠️ 计算占比降低 {abs(metric['improvement']):.2f}%，需要检查计算效率")

        report.append("")

        # 建议
        report.append("【建议】")
        report.append("-" * 40)

        if avg_improvement > 10:
            report.append("✅ 优化效果良好，建议：")
            report.append("   1. 进行功能验证，确保输出正确")
            report.append("   2. 进行稳定性测试，验证性能一致性")
            report.append("   3. 更新文档，记录优化效果")
        elif avg_improvement > 0:
            report.append("⚠️ 优化效果一般，建议：")
            report.append("   1. 分析优化效果不明显的原因")
            report.append("   2. 考虑组合多种优化策略")
            report.append("   3. 重新评估性能瓶颈")
        else:
            report.append("❌ 优化效果不明显，建议：")
            report.append("   1. 检查优化是否正确应用")
            report.append("   2. 回退到优化前的代码")
            report.append("   3. 重新进行性能诊断")
            report.append("   4. 尝试其他优化策略")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='PyPTO 算子性能对比工具')
    parser.add_argument('before_dir', help='优化前的输出目录')
    parser.add_argument('after_dir', help='优化后的输出目录')
    args = parser.parse_args()

    print(f"对比目录:")
    print(f"  优化前: {args.before_dir}")
    print(f"  优化后: {args.after_dir}")
    print()

    try:
        comparator = PerformanceComparator(args.before_dir, args.after_dir)
        report = comparator.generate_comparison_report()
        print(report)

        # 保存报告
        report_file = Path(args.after_dir) / "performance_comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {report_file}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保两个目录都存在且包含性能数据文件")
        sys.exit(1)
    except Exception as e:
        print(f"对比失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
