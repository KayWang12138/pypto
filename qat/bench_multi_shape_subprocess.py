#!/usr/bin/env python3
# coding: utf-8
"""
PyTorch小算子 vs PyPTO融合算子 性能对比测试（子进程版本）
解决多次profiling导致的内存问题
"""

import os
import sys
import csv
import json
import shutil
import argparse
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch_npu
import torch.nn.functional as F


def clamp_relu(x, min_val, max_val):
    min_val -= 1e-6
    max_val += 1e-6
    x = min_val + F.relu(x - min_val)
    x = max_val - F.relu(max_val - x)
    return x


def enhanced_lsq_plus(weight, scale, offset, group_size, bit, eps=1e-4, clip_val=0.99):
    """PyTorch前向实现"""
    eps = torch.tensor(eps, device=scale.device).float()
    scale = torch.where(scale > eps, scale, eps)

    orig_shape = weight.shape
    num_groups = weight.numel() // group_size
    weight = weight.view(num_groups, group_size)

    n_levels = 2 ** (bit - 1)
    shift = 0.5

    weight = weight - offset
    alpha = scale * n_levels

    weight = clamp_relu(weight / alpha, -clip_val, clip_val) * n_levels - shift
    weight = (weight.round() - weight).detach() + weight
    weight = (weight + shift) / n_levels
    weight = weight * alpha + offset

    weight = weight.view(orig_shape)
    return weight


PERFORMANCE_TEST_CONFIGS = {
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

FORWARD_KERNELS = [
    "aclnnGtTensor_CastAiCore_Cast",
    "aclnnGtTensor_GreaterAiCore_Greater",
    "aclnnSWhere_CastAiCore_Cast",
    "aclnnSWhere_SelectV2AiCore_SelectV2",
    "aclnnSWhere_CastAiCore_Cast",
    "aclnnSub_SubAiCore_Sub",
    "aclnnMuls_CastAiCore_Cast",
    "aclnnMuls_MulAiCore_Mul",
    "aclnnMuls_CastAiCore_Cast",
    "aclnnDiv_RealDivAiCore_RealDiv",
    "aclnnSubs_SubAiCore_Sub",
    "aclnnRelu_Relu_Relu",
    "aclnnAdds_AddAiCore_Add",
    "aclnnRsubs_SubAiCore_Sub",
    "aclnnRelu_Relu_Relu",
    "aclnnRsubs_SubAiCore_Sub",
    "aclnnMuls_CastAiCore_Cast",
    "aclnnMuls_MulAiCore_Mul",
    "aclnnMuls_CastAiCore_Cast",
    "aclnnSubs_SubAiCore_Sub",
    "aclnnRound_RoundDecimalsAiCore_Round",
    "aclnnSub_SubAiCore_Sub",
    "aclnnAdd_AddAiCore_Add",
    "aclnnAdds_AddAiCore_Add",
    "aclnnDivs_RealDivAiCore_RealDiv",
    "aclnnMul_MulAiCore_Mul",
    "aclnnAdd_AddAiCore_Add",
]

GROUP_SIZE = 128
BIT = 4


def benchmark_pytorch_backward(weight_shape, group_size, bit, device_id, output_dir):
    """
    PyTorch小算子基准测试（包含前向和反向）
    """
    device = f'npu:{device_id}'
    n, m = weight_shape
    groups_per_row = m // group_size
    num_groups = n * groups_per_row

    torch.manual_seed(42)
    weight = torch.randn(weight_shape, dtype=torch.float32, device=device).to(torch.bfloat16)
    scale = (torch.rand((num_groups, 1), dtype=torch.float32, device=device) * 0.1 + 0.01).to(torch.bfloat16)
    offset = (torch.randn((num_groups, 1), dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16)
    grad_output = torch.randn(weight_shape, dtype=torch.bfloat16, device=device)

    weight_pt = weight.detach().requires_grad_(True)
    scale_pt = scale.detach().requires_grad_(True)
    offset_pt = offset.detach().requires_grad_(True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=0),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir)
    ) as prof:
        
        total_steps = 10 
        for i in range(total_steps):
            output = enhanced_lsq_plus(weight_pt, scale_pt, offset_pt, group_size, bit)
            torch.npu.synchronize()
            output.backward(grad_output)
            torch.npu.synchronize()
            prof.step()


def parse_kernel_details_csv(csv_path: str, target_step_id: int = 6) -> Tuple[float, float, Dict]:
    """
    解析kernel_details.csv文件，统计前向和反向时间
    
    Args:
        csv_path: kernel_details.csv文件路径
        target_step_id: 目标Step ID（默认6）
    
    Returns:
        (forward_time_us, backward_time_us, details_dict)
    """
    forward_time = 0.0
    backward_time = 0.0
    forward_kernels = []
    backward_kernels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    step_rows = [row for row in rows if int(row['Step Id']) == target_step_id]
    
    if not step_rows:
        print(f"警告: 未找到 Step Id = {target_step_id} 的数据")
        return 0.0, 0.0, {}
    
    forward_phase = True
    
    for row in step_rows:
        kernel_name = row['Name'].strip()
        duration = float(row['Duration(us)'])
        
        if forward_phase:
            is_forward_kernel = any(fk in kernel_name for fk in [
                "GtTensor", "SWhere", "Greater", "SelectV2"
            ]) or any(fk == kernel_name for fk in FORWARD_KERNELS)
            
            if is_forward_kernel or kernel_name.startswith("aclnnAdd_AddAiCore_Add") and forward_time > 0:
                forward_time += duration
                forward_kernels.append((kernel_name, duration))
            else:
                forward_phase = False
                backward_time += duration
                backward_kernels.append((kernel_name, duration))
        else:
            backward_time += duration
            backward_kernels.append((kernel_name, duration))
    
    details = {
        'step_id': target_step_id,
        'total_kernels': len(step_rows),
        'forward_kernels': forward_kernels,
        'backward_kernels': backward_kernels,
        'forward_count': len(forward_kernels),
        'backward_count': len(backward_kernels)
    }
    
    return forward_time, backward_time, details


def find_kernel_details_csv(base_dir: str) -> str:
    """查找kernel_details.csv文件"""
    base_path = Path(base_dir)
    csv_files = list(base_path.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"未找到kernel_details.csv文件在 {base_dir}")
    
    return str(sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])


def get_pypto_backward_perf(pypto_perf_dir: str, shape_name: str) -> float:
    """
    从PyPTO性能数据中提取反向算子执行时间
    
    Args:
        pypto_perf_dir: PyPTO性能数据根目录
        shape_name: shape名称（如 "3B-1024x2048"）
    
    Returns:
        执行时间（us）
    """
    shape_dir = Path(pypto_perf_dir) / shape_name
    summary_file = Path(pypto_perf_dir) / f"{shape_name.split('-')[0]}_backward_perf_summary.md"
    
    bubble_log = shape_dir / "bubble_analysis.log"
    if bubble_log.exists():
        import re
        with open(bubble_log, 'r') as f:
            content = f.read()
        
        pattern = r'\[(AIV_\d+)\].*?Core Total Work Time: ([\d.]+)'
        matches = re.findall(pattern, content)
        
        if matches:
            max_work_time = max(float(m[1]) for m in matches)
            return max_work_time
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            content = f.read()
        
        pattern = rf'\| {shape_name} \|.*? \| (\d+\.?\d*) \|'
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
    
    return 0.0


def run_single_shape_benchmark(weight_shape, description, device_id, output_base, pypto_perf_dir):
    """
    运行单个shape的基准测试（在子进程中调用）
    返回JSON格式的结果
    """
    profiling_dir = Path(output_base) / "op_profiling_result"
    if profiling_dir.exists():
        shutil.rmtree(profiling_dir, ignore_errors=True)
    
    try:
        benchmark_pytorch_backward(
            weight_shape, GROUP_SIZE, BIT, device_id,
            str(profiling_dir)
        )
        
        csv_path = find_kernel_details_csv(str(profiling_dir))
        
        forward_time, backward_time, details = parse_kernel_details_csv(csv_path, target_step_id=6)
        
        pypto_time = get_pypto_backward_perf(pypto_perf_dir, description)
        
        speedup = (backward_time / pypto_time) if pypto_time > 0 else 0
        
        result = {
            'shape': list(weight_shape),
            'description': description,
            'pytorch_forward_us': forward_time,
            'pytorch_backward_us': backward_time,
            'pypto_backward_us': pypto_time,
            'speedup': speedup,
            'forward_count': details['forward_count'],
            'backward_count': details['backward_count'],
            'success': True
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        import traceback
        error_result = {
            'shape': list(weight_shape),
            'description': description,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }
        print(json.dumps(error_result))


def run_multi_shape_benchmark_subprocess(model_type: str, device_id: int, output_base: str, pypto_perf_dir: str):
    """
    使用子进程方式运行多shape基准测试
    每个shape测试在独立的子进程中运行，避免内存累积
    """
    print("=" * 80)
    print(f"PyTorch小算子 vs PyPTO融合算子 性能对比测试 - {model_type} (子进程模式)")
    print("=" * 80)
    print()
    
    configs = PERFORMANCE_TEST_CONFIGS.get(model_type, [])
    if not configs:
        print(f"错误: 未找到模型类型 {model_type} 的配置")
        return
    
    results = []
    
    for idx, config in enumerate(configs, 1):
        weight_shape = config["weight_shape"]
        description = config["description"]
        
        print(f"[{idx}/{len(configs)}] 测试 Shape: {weight_shape} ({description})")
        
        cmd = [
            sys.executable, __file__,
            '--single_shape',
            '--shape', f"{weight_shape[0]},{weight_shape[1]}",
            '--description', description,
            '--device_id', str(device_id),
            '--output', output_base,
            '--pypto_perf', pypto_perf_dir
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('{'):
                        result_data = json.loads(line)
                        
                        if result_data.get('success'):
                            print(f"  ✓ PyTorch前向时间: {result_data['pytorch_forward_us']:.2f} us ({result_data['forward_count']} kernels)")
                            print(f"  ✓ PyTorch反向时间: {result_data['pytorch_backward_us']:.2f} us ({result_data['backward_count']} kernels)")
                            print(f"  ✓ PyPTO反向时间:   {result_data['pypto_backward_us']:.2f} us")
                            print(f"  ✓ 加速比: {result_data['speedup']:.2f}x")
                            print()
                            
                            results.append({
                                'shape': tuple(result_data['shape']),
                                'description': result_data['description'],
                                'pytorch_forward_us': result_data['pytorch_forward_us'],
                                'pytorch_backward_us': result_data['pytorch_backward_us'],
                                'pypto_backward_us': result_data['pypto_backward_us'],
                                'speedup': result_data['speedup'],
                                'details': {
                                    'forward_count': result_data['forward_count'],
                                    'backward_count': result_data['backward_count']
                                }
                            })
                        else:
                            print(f"  ✗ 测试失败: {result_data.get('error', '未知错误')}")
                            results.append({
                                'shape': weight_shape,
                                'description': description,
                                'error': result_data.get('error', '未知错误')
                            })
                            print()
                            break
            else:
                print(f"  ✗ 子进程失败: {result.stderr}")
                results.append({
                    'shape': weight_shape,
                    'description': description,
                    'error': f"子进程失败: {result.stderr}"
                })
                print()
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 测试超时（120秒）")
            results.append({
                'shape': weight_shape,
                'description': description,
                'error': '测试超时'
            })
            print()
        except Exception as e:
            print(f"  ✗ 测试异常: {e}")
            results.append({
                'shape': weight_shape,
                'description': description,
                'error': str(e)
            })
            print()
    
    generate_comparison_report(model_type, results, output_base)
    
    return results


def generate_comparison_report(model_type: str, results: List[Dict], output_base: str):
    """生成性能对比报告"""
    report_path = Path(output_base) / f"{model_type}_performance_comparison.md"
    
    report = f"""# PyTorch小算子 vs PyPTO融合算子 性能对比报告 - {model_type}

## 测试配置
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **量化参数**: group_size={GROUP_SIZE}, bit={BIT}
- **测试方法**: PyTorch AutoGrad (前向+反向) vs PyPTO融合算子
- **测试模式**: 子进程隔离（避免内存累积）

## 性能对比结果

| Shape | PyTorch前向(us) | PyTorch反向(us) | PyPTO反向(us) | 加速比 |
|-------|----------------|----------------|--------------|--------|
"""
    
    total_pytorch_backward = 0
    total_pypto_backward = 0
    
    for r in results:
        if 'error' not in r:
            report += f"| {r['description']} | {r['pytorch_forward_us']:.2f} | {r['pytorch_backward_us']:.2f} | {r['pypto_backward_us']:.2f} | **{r['speedup']:.2f}x** |\n"
            total_pytorch_backward += r['pytorch_backward_us']
            total_pypto_backward += r['pypto_backward_us']
        else:
            report += f"| {r['description']} | - | - | - | ✗ 失败 |\n"
    
    avg_speedup = total_pytorch_backward / total_pypto_backward if total_pypto_backward > 0 else 0
    
    report += f"""
## 性能总结

| 指标 | 数值 |
|------|------|
| PyTorch反向总时间 | {total_pytorch_backward:.2f} us |
| PyPTO反向总时间 | {total_pypto_backward:.2f} us |
| **平均加速比** | **{avg_speedup:.2f}x** |

## 性能分析

### PyPTO融合算子优势

1. **算子融合**: 将多个小算子融合为一个算子，减少kernel launch开销
2. **内存优化**: 减少中间结果的内存读写，提高带宽利用率
3. **并行优化**: 利用PyPTO的并行优化能力，提高核心利用率

### PyTorch小算子劣势

1. **多次kernel launch**: 每个小算子都需要单独launch，开销大
2. **中间结果读写**: 需要将中间结果写回内存再读取，带宽浪费
3. **调度开销**: 多个kernel之间的调度和同步开销

## 测试说明

本测试使用子进程隔离模式，每个shape测试在独立的子进程中运行，避免多次profiling导致的内存累积问题。

---
Generated by PyPTO Performance Benchmark Tool (Subprocess Mode)
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("=" * 80)
    print(f"性能对比报告已生成: {report_path}")
    print(f"平均加速比: {avg_speedup:.2f}x")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PyTorch小算子 vs PyPTO融合算子性能对比（子进程模式）")
    parser.add_argument('--model', type=str, default="3B", 
                       choices=["3B", "7B", "30B", "all"],
                       help="模型类型")
    parser.add_argument('--device_id', type=int, default=0,
                       help="NPU设备ID")
    parser.add_argument('--output', type=str, default="perf_comparison",
                       help="输出目录")
    parser.add_argument('--pypto_perf', type=str, default="output_backward_perf",
                       help="PyPTO性能数据目录")
    
    parser.add_argument('--single_shape', action='store_true',
                       help="单shape测试模式（内部使用）")
    parser.add_argument('--shape', type=str,
                       help="Shape格式: N,M")
    parser.add_argument('--description', type=str,
                       help="Shape描述")
    args = parser.parse_args()
    
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        os.environ['TILE_FWK_DEVICE_ID'] = '0'
    
    torch.npu.set_device(args.device_id)
    
    if args.single_shape:
        if not args.shape or not args.description:
            print("错误: 单shape模式需要 --shape 和 --description 参数")
            sys.exit(1)
        
        n, m = map(int, args.shape.split(','))
        weight_shape = (n, m)
        
        run_single_shape_benchmark(
            weight_shape, args.description, args.device_id,
            args.output, args.pypto_perf
        )
    else:
        print(f"使用 NPU 设备: {args.device_id}\n")
        
        os.makedirs(args.output, exist_ok=True)
        
        if args.model == "all":
            for model in ["3B", "7B", "30B"]:
                run_multi_shape_benchmark_subprocess(model, args.device_id, args.output, args.pypto_perf)
                print()
        else:
            run_multi_shape_benchmark_subprocess(args.model, args.device_id, args.output, args.pypto_perf)


if __name__ == "__main__":
    main()