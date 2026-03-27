#!/usr/bin/env python3
# coding: utf-8
"""
非对称QAT算子多Shape性能基准测试程序
支持正向算子 create_asymmetric_qat_kernel 和反向算子 create_asymmetric_qat_backward_kernel
"""

import os
import sys
import re
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


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

    pattern = (
        r'\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+Core Total Work Time: ([\d.]+)\s+'
        r'Total Wait Time: ([\d.]+)\s+Wait Schedule Time: ([\d.]+)\s+'
        r'Wait Predecessor Time: ([\d.]+)'
    )

    for match in re.findall(pattern, content):
        cores.append(CoreMetrics(
            core_name=match[0],
            task_num=int(match[1]),
            total_work_time=float(match[2]),
            total_wait_time=float(match[3]),
            wait_schedule_time=float(match[4]),
            wait_predecessor_time=float(match[5])
        ))
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


# 测试脚本模板
_TEST_SCRIPT_TEMPLATES = {
    "forward": '''
import os
import torch
import torch_npu

device_id = {device_id}
os.environ['TILE_FWK_DEVICE_ID'] = str(device_id)
torch.npu.set_device(device_id)
torch.manual_seed(42)

weight_shape = {weight_shape}
n, m = weight_shape
group_size = 128
num_groups = (n * m) // group_size

weight = torch.randn(weight_shape, dtype=torch.bfloat16, device=f'npu:{{device_id}}')
scale = (torch.rand((num_groups, 1), dtype=torch.float32, device=f'npu:{{device_id}}') * 0.1 + 0.01).to(torch.bfloat16)
offset = (torch.randn((num_groups, 1), dtype=torch.float32, device=f'npu:{{device_id}}') * 0.1).to(torch.bfloat16)

from asymmetric_qat.asymmetric_qat import create_asymmetric_qat_kernel
output = create_asymmetric_qat_kernel(weight, scale, offset)
torch.npu.synchronize()
print("SUCCESS")
''',
    "backward": '''
import os
import torch
import torch_npu

device_id = {device_id}
os.environ['TILE_FWK_DEVICE_ID'] = str(device_id)
torch.npu.set_device(device_id)
torch.manual_seed(42)

weight_shape = {weight_shape}
n, m = weight_shape
group_size = 128
num_groups = (n * m) // group_size

grad_output = torch.randn(weight_shape, dtype=torch.bfloat16, device=f'npu:{{device_id}}')
weight = torch.randn(weight_shape, dtype=torch.bfloat16, device=f'npu:{{device_id}}')
scale = (torch.rand((num_groups, 1), dtype=torch.float32, device=f'npu:{{device_id}}') * 0.1 + 0.01).to(torch.bfloat16)
offset = (torch.randn((num_groups, 1), dtype=torch.float32, device=f'npu:{{device_id}}') * 0.1).to(torch.bfloat16)

from asymmetric_qat_backward.asymmetric_qat_backward import create_asymmetric_qat_backward_kernel
grad_weight, grad_scale, grad_offset = create_asymmetric_qat_backward_kernel(grad_output, weight, scale, offset)
torch.npu.synchronize()
print("SUCCESS")
'''
}

# 测试配置
TEST_CONFIGS = {
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

OPERATOR_NAMES = {
    "forward": "create_asymmetric_qat_kernel",
    "backward": "create_asymmetric_qat_backward_kernel"
}


class PerfBenchmark:
    """性能基准测试（支持正向/反向算子）"""

    def __init__(self, direction: str, output_base_dir: str):
        self.direction = direction
        self.operator_name = OPERATOR_NAMES[direction]
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.project_root = Path(__file__).parent.parent
        self.temp_script_dir = self.output_base_dir / "temp_scripts"
        self.temp_script_dir.mkdir(exist_ok=True)
        self.test_configs = TEST_CONFIGS  # 默认使用预设配置

    def run_single_shape_test(self, weight_shape: tuple, device_id: int) -> bool:
        """在子进程中运行单个shape的性能测试"""
        script_content = _TEST_SCRIPT_TEMPLATES[self.direction].format(
            device_id=device_id,
            weight_shape=weight_shape
        )

        n, m = weight_shape
        script_path = self.temp_script_dir / f"test_{self.direction}_{n}x{m}_{int(time.time())}.py"
        script_path.write_text(script_content)

        # 清理之前的output目录和缓存
        output_root = self.project_root / "output"
        if output_root.exists():
            for d in list(output_root.glob("output_*")):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass

        for cache_dir in [self.project_root / ".pypto_cache", self.project_root / "pypto_cache"]:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                except Exception:
                    pass

        time.sleep(0.1)
        print(f"  → 编译执行中...", flush=True)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path.absolute())],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                print(f"  ✗ 失败 (退出码: {result.returncode})")
                if result.stderr:
                    print(f"    {result.stderr[-300:]}")
                return False

            if "SUCCESS" not in (result.stdout + result.stderr):
                print(f"  ✗ 未成功完成")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时")
            return False
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            return False
        finally:
            time.sleep(0.1)

    def find_latest_output_dir(self) -> Optional[Path]:
        """查找最新的output目录"""
        output_root = self.project_root / "output"
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
            return {"error": f"bubble_analysis.log not found", "shape_name": shape_name}

        cores = parse_bubble_analysis(str(bubble_log))
        if not cores:
            return {"error": "No core data found", "shape_name": shape_name}

        metrics = calculate_performance_metrics(cores)
        return {
            "shape_name": shape_name,
            "metrics": metrics,
            "ratings": {
                "core_utilization": get_rating(metrics['avg_core_utilization'], 'core_utilization'),
                "bubble_rate": get_rating(metrics['avg_bubble_rate'], 'bubble_rate'),
                "load_balance": get_rating(metrics['load_balance'], 'load_balance')
            }
        }

    def run_benchmark(self, model_type: str, device_id: int) -> List[Dict]:
        """运行基准测试"""
        print(f"\n{'='*60}")
        print(f"{self.direction.upper()} 算子 - {model_type} 模型")
        print(f"算子: {self.operator_name}")
        print(f"{'='*60}\n")

        if model_type not in self.test_configs:
            print(f"错误: 不支持的模型类型 {model_type}")
            return []

        results = []
        configs = self.test_configs[model_type]

        for idx, config in enumerate(configs, 1):
            weight_shape = config["weight_shape"]
            description = config["description"]
            print(f"[{idx}/{len(configs)}] {weight_shape} ({description})")

            try:
                success = self.run_single_shape_test(weight_shape, device_id)
                if success:
                    print(f"  ✓ 执行成功")
                    output_dir = self.find_latest_output_dir()
                    if output_dir:
                        time.sleep(0.5)
                        result = self.analyze_shape_performance(description, output_dir)
                        results.append(result)
                        if "metrics" in result:
                            m = result["metrics"]
                            r = result["ratings"]
                            print(f"    核心利用率: {m['avg_core_utilization']:.1f}% {r['core_utilization']}")
                            print(f"    气泡率: {m['avg_bubble_rate']:.1f}% {r['bubble_rate']}")
                            print(f"    执行时间: {m['max_work_time']:.1f} us")
                        else:
                            print(f"    ✗ 分析失败: {result.get('error', 'Unknown')}")
                    else:
                        print(f"  ✗ 未找到输出目录")
                        results.append({"error": "No output", "shape_name": description})
                else:
                    results.append({"error": "Test failed", "shape_name": description})
            except Exception as e:
                print(f"  ✗ 异常: {e}")
                results.append({"error": str(e), "shape_name": description})

        return results


def generate_final_report(all_results: Dict, output_dir: Path):
    """生成最终汇总报告"""
    report_path = output_dir / "perf_summary.md"

    lines = [
        "# 性能基准测试报告",
        f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    for direction in ["forward", "backward"]:
        if direction not in all_results or not all_results[direction]:
            continue

        operator_name = OPERATOR_NAMES[direction]
        lines.append(f"\n## {direction.upper()} 算子 ({operator_name})")
        lines.append("")
        lines.append("| Shape | 核心利用率 | 气泡率 | 执行时间(us) | 负载均衡 | 评级 |")
        lines.append("|-------|-----------|--------|-------------|---------|------|")

        for model_type in ["custom", "3B", "7B", "30B"]:
            results = all_results[direction].get(model_type, [])
            for r in results:
                if "metrics" in r:
                    m = r["metrics"]
                    rt = r["ratings"]
                    stars = [rt['core_utilization'].count('⭐'), rt['bubble_rate'].count('⭐'), rt['load_balance'].count('⭐')]
                    overall = '⭐' * round(sum(stars) / 3)
                    lines.append(
                        f"| {r['shape_name']} | {m['avg_core_utilization']:.1f}% {rt['core_utilization']} | "
                        f"{m['avg_bubble_rate']:.1f}% {rt['bubble_rate']} | {m['max_work_time']:.1f} | "
                        f"{m['load_balance']:.1f}% {rt['load_balance']} | {overall} |"
                    )
                else:
                    lines.append(f"| {r.get('shape_name', 'Unknown')} | - | - | - | - | ✗ |")

    report_path.write_text("\n".join(lines))
    print(f"\n{'='*60}")
    print(f"报告已生成: {report_path}")
    print(f"{'='*60}")


def parse_shapes(shapes_str: str) -> List[Dict]:
    """解析自定义shape字符串，格式: '1024x2048,768x3072' 或 '1024,2048;768,3072'"""
    if not shapes_str:
        return []

    configs = []
    shapes_str = shapes_str.strip().strip('"').strip("'")

    for idx, part in enumerate(shapes_str.replace(';', ',').split(',')):
        part = part.strip()
        if 'x' in part:
            dims = part.split('x')
        else:
            dims = part.split('X')

        if len(dims) == 2:
            n, m = int(dims[0].strip()), int(dims[1].strip())
            configs.append({
                "weight_shape": (n, m),
                "description": f"{n}x{m}"
            })

    return configs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyPTO 非对称QAT算子性能基准测试")
    parser.add_argument('--direction', type=str, default="backward",
                        choices=["forward", "backward", "both"],
                        help="测试方向")
    parser.add_argument('--model', type=str, default=None,
                        choices=["3B", "7B", "30B", "all"],
                        help="模型类型预设 (与--shapes互斥)")
    parser.add_argument('--shapes', type=str, default=None,
                        help="自定义shape，格式: '1024x2048,768x3072'")
    parser.add_argument('--device_id', type=int, default=5,
                        help="NPU设备ID")
    args = parser.parse_args()

    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print(f"设置 TILE_FWK_DEVICE_ID = {args.device_id}")
        os.environ['TILE_FWK_DEVICE_ID'] = str(args.device_id)

    # 确定测试配置
    custom_configs = parse_shapes(args.shapes) if args.shapes else None

    if custom_configs:
        model_list = ["custom"]
        all_results = {"forward": {}, "backward": {}}
        test_configs = {"custom": custom_configs}
    elif args.model:
        model_list = ["3B", "7B", "30B"] if args.model == "all" else [args.model]
        all_results = {"forward": {}, "backward": {}}
        test_configs = TEST_CONFIGS
    else:
        print("错误: 请指定 --model 或 --shapes")
        return

    directions = ["forward", "backward"] if args.direction == "both" else [args.direction]

    for direction in directions:
        benchmark = PerfBenchmark(direction, f"output_asym_{direction}_perf")
        benchmark.test_configs = test_configs
        for model in model_list:
            results = benchmark.run_benchmark(model, args.device_id)
            all_results[direction][model] = results

    generate_final_report(all_results, Path("output_asym_perf"))


if __name__ == "__main__":
    main()