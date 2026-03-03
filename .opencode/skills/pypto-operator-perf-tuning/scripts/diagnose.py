#!/usr/bin/env python3
"""
PyPTO 算子性能诊断脚本
用于识别性能瓶颈、分析根因并提供优化建议
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse


class PerformanceDiagnostics:
    """性能诊断器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.swimlane_file = self._find_file("merged_swimlane.json")
        self.trace_file = self._find_file("machine_runtime_operator_trace.json")
        self.bubble_file = self._find_file("bubble_analysis.log")

    def _find_file(self, filename: str) -> Path:
        """查找性能数据文件"""
        for root, dirs, files in os.walk(self.output_dir):
            if filename in files:
                return Path(root) / filename
        raise FileNotFoundError(f"未找到文件: {filename}")

    def analyze_bubble(self) -> Dict[str, Any]:
        """分析气泡报告"""
        if not self.bubble_file.exists():
            return {}

        per_core = {}
        with open(self.bubble_file, 'r', encoding='utf-8') as f:
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

        result = {
            "total_wait_time": 0.0,
            "total_work_time": 0.0,
            "bubble_rate": 0.0,
            "schedule_wait_rate": 0.0,
            "pred_wait_rate": 0.0,
            "cores": per_core
        }

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

    def analyze_swimlane(self) -> Dict[str, Any]:
        """分析泳道图数据"""
        if not self.swimlane_file.exists():
            return {}

        with open(self.swimlane_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {
            "total_tasks": 0,
            "avg_execution_time": 0.0,
            "max_execution_time": 0.0,
            "min_execution_time": float(1e9),
            "total_execution_time": 0.0,
            "memory_usage": 0,
            "tasks": []
        }

        if isinstance(data, list):
            for item in data:
                if "dur" in item:
                    dur = self._parse_time(item["dur"])
                    result["tasks"].append(dur)
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

        if result["tasks"]:
            result["avg_execution_time"] = sum(result["tasks"]) / len(result["tasks"])

        return result

    def analyze_trace(self) -> Dict[str, Any]:
        """分析性能追踪数据"""
        if not self.trace_file.exists():
            return {}

        with open(self.trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {
            "total_time": 0.0,
            "control_overhead": 0.0,
            "computation_time": 0.0,
            "stages": {}
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

                    if "name" in item:
                        stage_name = item["name"]
                        if stage_name not in result["stages"]:
                            result["stages"][stage_name] = 0.0
                        result["stages"][stage_name] += dur

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

    def diagnose_bottlenecks(self) -> List[Dict[str, Any]]:
        """诊断性能瓶颈"""
        bottlenecks = []

        bubble_data = self.analyze_bubble()
        trace_data = self.analyze_trace()

        # 控制开销瓶颈
        if trace_data and trace_data.get('control_overhead_ratio', 0) > 50:
            bottlenecks.append({
                "type": "控制开销过高",
                "severity": "高" if trace_data['control_overhead_ratio'] > 70 else "中",
                "value": f"{trace_data['control_overhead_ratio']:.2f}%",
                "threshold": "50%",
                "root_causes": self._analyze_control_overhead_root_causes(bubble_data, trace_data),
                "strategies": ["数据局部性优化", "减少同步操作", "优化控制流"]
            })

        # 气泡率瓶颈
        if bubble_data and bubble_data.get('bubble_rate', 0) > 10:
            bottlenecks.append({
                "type": "气泡率过高",
                "severity": "高" if bubble_data['bubble_rate'] > 20 else "中",
                "value": f"{bubble_data['bubble_rate']:.2f}%",
                "threshold": "10%",
                "root_causes": self._analyze_bubble_root_causes(bubble_data),
                "strategies": ["任务调度优化", "并行度优化", "依赖关系优化"]
            })

        # 调度等待瓶颈
        if bubble_data and bubble_data.get('schedule_wait_rate', 0) > 30:
            bottlenecks.append({
                "type": "调度等待过多",
                "severity": "高" if bubble_data['schedule_wait_rate'] > 50 else "中",
                "value": f"{bubble_data['schedule_wait_rate']:.2f}%",
                "threshold": "30%",
                "root_causes": ["任务分配不均衡", "调度策略不当", "资源竞争"],
                "strategies": ["负载均衡优化", "调度算法改进", "资源隔离"]
            })

        # 前驱等待瓶颈
        if bubble_data and bubble_data.get('pred_wait_rate', 0) > 30:
            bottlenecks.append({
                "type": "前驱等待过多",
                "severity": "高" if bubble_data['pred_wait_rate'] > 50 else "中",
                "value": f"{bubble_data['pred_wait_rate']:.2f}%",
                "threshold": "30%",
                "root_causes": ["依赖链过长", "任务串行化", "关键路径过长"],
                "strategies": ["任务重组", "依赖优化", "流水线化"]
            })

        # 计算效率瓶颈
        if trace_data and trace_data.get('computation_ratio', 0) < 30:
            bottlenecks.append({
                "type": "计算效率低",
                "severity": "高" if trace_data['computation_ratio'] < 20 else "中",
                "value": f"{trace_data['computation_ratio']:.2f}%",
                "threshold": "30%",
                "root_causes": ["数据传输瓶颈", "指令停顿", "资源利用率低"],
                "strategies": ["内存访问优化", "向量化优化", "流水线优化"]
            })

        return bottlenecks

    def _analyze_control_overhead_root_causes(self, bubble_data: Dict, trace_data: Dict) -> List[str]:
        """分析控制开销根因"""
        causes = []

        if bubble_data and bubble_data.get('schedule_wait_rate', 0) > 20:
            causes.append("频繁的任务调度导致控制开销")

        if bubble_data and bubble_data.get('bubble_rate', 0) > 15:
            causes.append("高气泡率表明核心空闲时间长")

        if trace_data and 'stages' in trace_data:
            stages = trace_data['stages']
            if 'DEV_TASK_SCHED_EXEC' in stages and stages['DEV_TASK_SCHED_EXEC'] > 0:
                causes.append("任务调度执行时间长")

        if not causes:
            causes.append("可能存在频繁的内存访问或同步操作")

        return causes

    def _analyze_bubble_root_causes(self, bubble_data: Dict) -> List[str]:
        """分析气泡根因"""
        causes = []

        if bubble_data.get('schedule_wait_rate', 0) > bubble_data.get('pred_wait_rate', 0):
            causes.append("调度等待是主要因素")
        else:
            causes.append("前驱等待是主要因素")

        return causes

    def generate_optimization_recommendations(self, bottlenecks: List[Dict]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []

        for bottleneck in bottlenecks:
            for strategy in bottleneck['strategies']:
                if strategy == "数据局部性优化":
                    recommendations.append({
                        "strategy": "数据局部性优化",
                        "priority": "高",
                        "description": "将高频访问的数据放在栈上或缓存中",
                        "actions": [
                            "识别高频访问的数据",
                            "将堆分配改为栈分配",
                            "使用引用减少解引用",
                            "紧凑化数据结构"
                        ],
                        "expected_improvement": "20-50% 控制开销降低"
                    })
                elif strategy == "内存访问优化":
                    recommendations.append({
                        "strategy": "内存访问优化",
                        "priority": "高",
                        "description": "优化内存访问模式，提高缓存命中率",
                        "actions": [
                            "使用连续内存布局",
                            "避免跨页访问",
                            "预取数据",
                            "对齐内存边界"
                        ],
                        "expected_improvement": "10-30% 性能提升"
                    })
                elif strategy == "任务调度优化":
                    recommendations.append({
                        "strategy": "任务调度优化",
                        "priority": "中",
                        "description": "改进任务调度策略，减少等待时间",
                        "actions": [
                            "分析任务依赖关系",
                            "优化任务分配策略",
                            "实现负载均衡",
                            "减少调度开销"
                        ],
                        "expected_improvement": "15-40% 气泡率降低"
                    })
                elif strategy == "并行度优化":
                    recommendations.append({
                        "strategy": "并行度优化",
                        "priority": "中",
                        "description": "提高并行度，充分利用多核资源",
                        "actions": [
                            "识别可并行任务",
                            "使用线程池",
                            "实现数据并行",
                            "减少锁竞争"
                        ],
                        "expected_improvement": "2-10x 吞吐量提升"
                    })

        # 去重
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            key = rec['strategy']
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)

        return unique_recommendations

    def generate_diagnostic_report(self) -> str:
        """生成诊断报告"""
        bubble_data = self.analyze_bubble()
        swimlane_data = self.analyze_swimlane()
        trace_data = self.analyze_trace()
        bottlenecks = self.diagnose_bottlenecks()
        recommendations = self.generate_optimization_recommendations(bottlenecks)

        report = []
        report.append("=" * 80)
        report.append("PyPTO 算子性能诊断报告")
        report.append("=" * 80)
        report.append("")

        # 性能概览
        report.append("【性能概览】")
        report.append("-" * 40)

        if trace_data:
            report.append(f"总执行时间: {trace_data.get('total_time', 0):.2f} us")
            report.append(f"控制开销: {trace_data.get('control_overhead', 0):.2f} us ({trace_data.get('control_overhead_ratio', 0):.2f}%)")
            report.append(f"计算时间: {trace_data.get('computation_time', 0):.2f} us ({trace_data.get('computation_ratio', 0):.2f}%)")
            report.append("")

        if bubble_data:
            report.append(f"气泡率: {bubble_data.get('bubble_rate', 0):.2f}%")
            report.append(f"调度等待占比: {bubble_data.get('schedule_wait_rate', 0):.2f}%")
            report.append(f"前驱等待占比: {bubble_data.get('pred_wait_rate', 0):.2f}%")
            report.append("")

        if swimlane_data:
            report.append(f"总任务数: {swimlane_data.get('total_tasks', 0)}")
            report.append(f"平均执行时间: {swimlane_data.get('avg_execution_time', 0):.2f} us")
            report.append(f"内存使用: {swimlane_data.get('memory_usage', 0)} bytes")
            report.append("")

        # 性能评级
        report.append("【性能评级】")
        report.append("-" * 40)

        overall_score = 5
        if trace_data and trace_data.get('control_overhead_ratio', 0) > 50:
            overall_score -= 1
        if bubble_data and bubble_data.get('bubble_rate', 0) > 10:
            overall_score -= 1
        if bubble_data and bubble_data.get('schedule_wait_rate', 0) > 30:
            overall_score -= 1
        if bubble_data and bubble_data.get('pred_wait_rate', 0) > 30:
            overall_score -= 1

        rating = "⭐" * max(1, overall_score)
        report.append(f"综合评分: {rating}")
        report.append("")

        # 性能瓶颈
        if bottlenecks:
            report.append("【性能瓶颈】")
            report.append("-" * 40)

            for i, bottleneck in enumerate(bottlenecks, 1):
                report.append(f"{i}. {bottleneck['type']} ({bottleneck['severity']})")
                report.append(f"   当前值: {bottleneck['value']} (阈值: {bottleneck['threshold']})")
                report.append(f"   根因分析:")
                for cause in bottleneck['root_causes']:
                    report.append(f"     - {cause}")
                report.append(f"   推荐策略: {', '.join(bottleneck['strategies'])}")
                report.append("")
        else:
            report.append("【性能瓶颈】")
            report.append("-" * 40)
            report.append("✅ 未检测到明显的性能瓶颈")
            report.append("")

        # 优化建议
        if recommendations:
            report.append("【优化建议】")
            report.append("-" * 40)

            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec['strategy']} ({rec['priority']}优先级)")
                report.append(f"   {rec['description']}")
                report.append(f"   预期改善: {rec['expected_improvement']}")
                report.append(f"   具体措施:")
                for action in rec['actions']:
                    report.append(f"     - {action}")
                report.append("")
        else:
            report.append("【优化建议】")
            report.append("-" * 40)
            report.append("✅ 性能表现良好，无需优化")
            report.append("")

        # 下一步行动
        report.append("【下一步行动】")
        report.append("-" * 40)

        if bottlenecks:
            report.append("1. 根据性能瓶颈选择合适的优化策略")
            report.append("2. 参考 SKILL.md 中的代码示例进行修改")
            report.append("3. 重新编译并运行，生成新的性能数据")
            report.append("4. 使用 compare.py 对比优化前后的性能")
            report.append("5. 验证功能正确性")
        else:
            report.append("✅ 当前性能表现良好，无需进一步优化")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='PyPTO 算子性能诊断工具')
    parser.add_argument('output_dir', nargs='?', default='output',
                       help='输出目录路径（默认: output）')
    args = parser.parse_args()

    print(f"分析目录: {args.output_dir}")
    print()

    try:
        diagnostics = PerformanceDiagnostics(args.output_dir)
        report = diagnostics.generate_diagnostic_report()
        print(report)

        # 保存报告
        report_file = Path(args.output_dir) / "performance_diagnostic_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {report_file}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保已运行算子并生成了性能数据文件")
        sys.exit(1)
    except Exception as e:
        print(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
