#!/usr/bin/env python3
# coding: utf-8
"""
反向算子多Shape性能基准测试程序
为 create_asymmetric_qat_backward_kernel 的每个shape单独采集性能数据
"""

import os
import sys
import json
import shutil
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch_npu
import pypto


@dataclass
class CoreMetrics:
    """核心性能指标"""
    core_name: str
    task_num: int
    total_work_time: float
    total_wait_time: float
    wait_schedule_time: float
    wait_predecessor_time: float
    
    @property
    def aicore_time(self) -> float:
        return self.total_work_time - self.total_wait_time
    
    @property
    def core_utilization(self) -> float:
        total_time = self.aicore_time + self.total_wait_time
        return (self.aicore_time / total_time * 100) if total_time > 0 else 0.0
    
    @property
    def bubble_rate(self) -> float:
        total_time = self.aicore_time + self.wait_schedule_time
        return (self.wait_schedule_time / total_time * 100) if total_time > 0 else 0.0


def parse_bubble_analysis(log_path: str) -> List[CoreMetrics]:
    """解析bubble_analysis.log文件"""
    cores = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    core_pattern = (
        r'\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+Core Total Work Time: ([\d.]+)\s+'
        r'Total Wait Time: ([\d.]+)\s+Wait Schedule Time: ([\d.]+)\s+'
        r'Wait Predecessor Time: ([\d.]+)'
    )
    
    matches = re.findall(core_pattern, content)
    
    for match in matches:
        core = CoreMetrics(
            core_name=match[0],
            task_num=int(match[1]),
            total_work_time=float(match[2]),
            total_wait_time=float(match[3]),
            wait_schedule_time=float(match[4]),
            wait_predecessor_time=float(match[5])
        )
        cores.append(core)
    
    return cores


def calculate_performance_metrics(cores: List[CoreMetrics]) -> Dict:
    """计算性能指标"""
    if not cores:
        return {}
    
    aic_cores = [c for c in cores if c.core_name.startswith('AIC')]
    aiv_cores = [c for c in cores if c.core_name.startswith('AIV')]
    
    avg_core_utilization = sum(c.core_utilization for c in cores) / len(cores)
    avg_bubble_rate = sum(c.bubble_rate for c in cores) / len(cores)
    max_work_time = max(c.total_work_time for c in cores)
    
    aic_times = [c.aicore_time for c in aic_cores] if aic_cores else [0]
    mean_time = sum(aic_times) / len(aic_times)
    variance = sum((t - mean_time) ** 2 for t in aic_times) / len(aic_times) if aic_times else 0
    std_dev = variance ** 0.5
    load_balance = (1 - std_dev / mean_time) * 100 if mean_time > 0 else 100
    
    return {
        'avg_core_utilization': avg_core_utilization,
        'avg_bubble_rate': avg_bubble_rate,
        'max_work_time': max_work_time,
        'load_balance': load_balance,
        'aic_cores_count': len(aic_cores),
        'aiv_cores_count': len(aiv_cores)
    }


def get_rating(value: float, metric_type: str) -> str:
    """获取性能评级"""
    ratings = {
        'core_utilization': [(90, '⭐⭐⭐⭐⭐'), (80, '⭐⭐⭐⭐'), (60, '⭐⭐⭐'), (50, '⭐⭐'), (0, '⭐')],
        'bubble_rate': [(2, '⭐⭐⭐⭐⭐'), (5, '⭐⭐⭐⭐'), (10, '⭐⭐⭐'), (20, '⭐⭐'), (100, '⭐')],
        'load_balance': [(90, '⭐⭐⭐⭐⭐'), (80, '⭐⭐⭐⭐'), (60, '⭐⭐⭐'), (50, '⭐⭐'), (0, '⭐')]
    }
    
    for threshold, rating in ratings.get(metric_type, []):
        if metric_type == 'bubble_rate':
            if value < threshold:
                return rating
        else:
            if value > threshold:
                return rating
    return '⭐'


class BackwardPerfBenchmark:
    """反向算子性能基准测试"""
    
    def __init__(self, output_base_dir: str = "output_backward_perf"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
        self.test_configs = {
            "3B": [
                {"weight_shape": (1024, 2048), "description": "3B-1024x2048"},
                {"weight_shape": (768, 2048), "description": "3B-768x2048"},
                {"weight_shape": (2048, 768), "description": "3B-2048x768"},
                {"weight_shape": (4096, 2048), "description": "3B-4096x2048"},
            ],
            "7B": [
                {"weight_shape": (3072, 768), "description": "7B-3072x768"},
                {"weight_shape": (768, 3072), "description": "7B-768x3072"},
                {"weight_shape": (1024, 3072), "description": "7B-1024x3072"},
                {"weight_shape": (5632, 3072), "description": "7B-5632x3072"},
                {"weight_shape": (3072, 2816), "description": "7B-3072x2816"},
            ],
            "30B": [
                {"weight_shape": (2560, 1536), "description": "30B-2560x1536"},
                {"weight_shape": (2480, 2560), "description": "30B-2480x2560"},
                {"weight_shape": (6144, 2560), "description": "30B-6144x2560"},
                {"weight_shape": (2560, 3072), "description": "30B-2560x3072"},
            ]
        }
        
        self.group_size = 128
        self.bit = 4
        
    def create_backward_kernel(self, run_mode: str = "npu"):
        """创建带性能采集的反向kernel"""
        from qat.asy_backward import create_asymmetric_qat_backward_kernel
        # 新接口：create_asymmetric_qat_backward_kernel(group_size, bit, ..., enable_perf)
        return create_asymmetric_qat_backward_kernel(
            group_size=self.group_size,
            bit=self.bit,
            run_mode=run_mode,
            enable_perf=True
        )
    
    def run_single_shape_test(self, weight_shape: tuple, device_id: int) -> Dict:
        """运行单个shape的性能测试"""
        n, m = weight_shape
        num_groups = (n * m) // self.group_size
        
        device = f'npu:{device_id}'
        
        # 准备输入数据
        grad_output = torch.randn(weight_shape, dtype=torch.bfloat16, device=device)
        weight = torch.randn(weight_shape, dtype=torch.bfloat16, device=device)
        scale = (torch.rand((num_groups, 1), dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
        offset = (torch.randn((num_groups, 1), dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
        
        # 创建并运行kernel
        # 新接口返回 asymmetric_qat_kernel_back 函数
        # 它会自动处理reshape: (N, M) -> (G, group_size)
        kernel_back = self.create_backward_kernel(run_mode="npu")
        
        # 调用kernel，返回
        grad_weight, grad_scale, grad_offset = kernel_back(
            grad_output, weight, scale, offset
        )
        
        return {"status": "success", "shape": weight_shape}
    
    def find_latest_output_dir(self) -> Optional[Path]:
        """查找最新的output目录"""
        output_root = Path("output")
        if not output_root.exists():
            return None
        
        output_dirs = sorted(
            [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("output_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        return output_dirs[0] if output_dirs else None
    
    def analyze_shape_performance(self, shape_name: str, output_dir: Path) -> Dict:
        """分析单个shape的性能"""
        bubble_log = output_dir / "bubble_analysis.log"
        
        if not bubble_log.exists():
            return {"error": f"bubble_analysis.log not found in {output_dir}"}
        
        cores = parse_bubble_analysis(str(bubble_log))
        if not cores:
            return {"error": "No core data found"}
        
        metrics = calculate_performance_metrics(cores)
        
        util_rating = get_rating(metrics['avg_core_utilization'], 'core_utilization')
        bubble_rating = get_rating(metrics['avg_bubble_rate'], 'bubble_rate')
        balance_rating = get_rating(metrics['load_balance'], 'load_balance')
        
        return {
            "shape_name": shape_name,
            "output_dir": str(output_dir),
            "metrics": metrics,
            "ratings": {
                "core_utilization": util_rating,
                "bubble_rate": bubble_rating,
                "load_balance": balance_rating
            }
        }
    
    def run_benchmark(self, model_type: str = "3B", device_id: int = 0):
        """运行基准测试"""
        print("=" * 80)
        print(f"PyPTO 反向算子性能基准测试 - {model_type}")
        print("算子: create_asymmetric_qat_backward_kernel")
        print("=" * 80)
        print()
        
        if model_type not in self.test_configs:
            print(f"错误: 不支持的模型类型 {model_type}")
            return
        
        configs = self.test_configs[model_type]
        results = []
        
        # 清理旧的output目录
        output_root = Path("output")
        if output_root.exists():
            for d in list(output_root.glob("output_*")):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"警告: 无法删除 {d}: {e}")
        
        # 为每个shape运行测试
        for idx, config in enumerate(configs, 1):
            weight_shape = config["weight_shape"]
            description = config["description"]
            
            print(f"[{idx}/{len(configs)}] 测试 Shape: {weight_shape} ({description})")
            
            try:
                # 运行测试
                self.run_single_shape_test(weight_shape, device_id)
                
                # 查找生成的性能数据
                output_dir = self.find_latest_output_dir()
                
                if output_dir:
                    # 保存到shape专属目录
                    shape_dir = self.output_base_dir / description
                    shape_dir.mkdir(exist_ok=True)
                    
                    # 复制性能数据
                    for file_name in ["bubble_analysis.log", "merged_swimlane.json", 
                                     "machine_runtime_operator_trace.json"]:
                        src_file = output_dir / file_name
                        if src_file.exists():
                            shutil.copy2(src_file, shape_dir / file_name)
                    
                    # 分析性能
                    perf_result = self.analyze_shape_performance(description, shape_dir)
                    results.append(perf_result)
                    
                    # 打印结果
                    if "metrics" in perf_result:
                        m = perf_result["metrics"]
                        print(f"  ✓ 核心利用率: {m['avg_core_utilization']:.2f}% {perf_result['ratings']['core_utilization']}")
                        print(f"  ✓ 气泡率: {m['avg_bubble_rate']:.2f}% {perf_result['ratings']['bubble_rate']}")
                        print(f"  ✓ 执行时间: {m['max_work_time']:.2f} us")
                        print(f"  ✓ 性能数据: {shape_dir}")
                    else:
                        print(f"  ✗ 性能分析失败: {perf_result.get('error', 'Unknown')}")
                else:
                    print("  ✗ 未找到性能数据文件")
                    results.append({"error": "No output directory found", "shape": weight_shape})
                
            except Exception as e:
                print(f"  ✗ 测试失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({"error": str(e), "shape": weight_shape})
            
            print()
        
        # 生成汇总报告
        self.generate_summary_report(model_type, results)
        
        return results
    
    def generate_summary_report(self, model_type: str, results: List[Dict]):
        """生成汇总报告"""
        report_path = self.output_base_dir / f"{model_type}_backward_perf_summary.md"
        
        report = f"""# PyPTO 反向算子性能基准测试报告 - {model_type}

## 测试配置
- **算子名称**: create_asymmetric_qat_backward_kernel
- **模型类型**: {model_type}
- **量化参数**: group_size={self.group_size}, bit={self.bit}
- **Tensor形状**: (G, group_size) 其中 G = N*M/group_size
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 性能测试结果

| Shape | 核心利用率 | 气泡率 | 执行时间(us) | 负载均衡度 | 综合评级 |
|-------|-----------|--------|------------|-----------|---------|
"""
        
        for result in results:
            if "metrics" in result:
                m = result["metrics"]
                r = result["ratings"]
                
                # 计算综合评级
                ratings = [r['core_utilization'], r['bubble_rate'], r['load_balance']]
                stars = [s.count('⭐') for s in ratings]
                avg_stars = sum(stars) / len(stars)
                overall = '⭐' * int(round(avg_stars))
                
                report += f"| {result['shape_name']} | {m['avg_core_utilization']:.2f}% {r['core_utilization']} | "
                report += f"{m['avg_bubble_rate']:.2f}% {r['bubble_rate']} | "
                report += f"{m['max_work_time']:.2f} | {m['load_balance']:.2f}% {r['load_balance']} | {overall} |\n"
            else:
                report += f"| {result.get('shape_name', result.get('shape', 'Unknown'))} | - | - | - | - | ✗ 失败 |\n"
        
        report += f"""
## 接口说明

### 新接口特点
- **输入Tensor形状**: grad_output/weight 为 (N, M)，内部自动reshape为 (G, group_size)
- **输出**: 直接返回 (grad_weight, grad_scale, grad_offset) tuple
- **group_size**: 128（默认），G = N*M/128

### 反向算子特点
- **输入**: grad_output (N, M), weight (N, M), scale (G, 1), offset (G, 1)
- **输出**: grad_weight (N, M), grad_scale (G, 1), grad_offset (G, 1)
- **计算复杂度**: 高于前向算子（需要重计算中间结果）

## 性能数据文件

每个shape的性能数据保存在独立目录下：
- `bubble_analysis.log` - 气泡分析报告
- `merged_swimlane.json` - 泳道图数据
- `machine_runtime_operator_trace.json` - 性能追踪文件

可在 https://ui.perfetto.dev/ 上传泳道图文件进行可视化分析。

## 详细性能数据目录

"""
        for result in results:
            if "output_dir" in result:
                report += f"- **{result['shape_name']}**: `{result['output_dir']}`\n"
        
        report += "\n---\n"
        report += "Generated by PyPTO Backward Kernel Performance Benchmark Tool\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print("=" * 80)
        print(f"性能汇总报告已生成: {report_path}")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PyPTO 反向算子多Shape性能基准测试")
    parser.add_argument('--model', type=str, default="3B", 
                       choices=["3B", "7B", "30B", "all"],
                       help="模型类型")
    parser.add_argument('--device_id', type=int, default=0,
                       help="NPU设备ID")
    args = parser.parse_args()
    
    # 检查环境变量
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("警告: TILE_FWK_DEVICE_ID 未设置，尝试设置为 0")
        os.environ['TILE_FWK_DEVICE_ID'] = '0'
    
    if 'PTO_TILE_LIB_CODE_PATH' not in os.environ:
        print("错误: PTO_TILE_LIB_CODE_PATH 未设置")
        print("请设置: export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa")
        sys.exit(1)
    
    # 设置NPU设备
    torch.npu.set_device(args.device_id)
    print(f"使用 NPU 设备: {args.device_id}\n")
    
    # 创建基准测试实例
    benchmark = BackwardPerfBenchmark()
    
    # 运行测试
    if args.model == "all":
        for model in ["3B", "7B", "30B"]:
            benchmark.run_benchmark(model, args.device_id)
            print()
    else:
        benchmark.run_benchmark(args.model, args.device_id)


if __name__ == "__main__":
    main()