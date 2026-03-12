#!/usr/bin/env python3
# coding: utf-8
"""
PyTorch小算子 vs PyPTO融合算子 性能对比测试
支持多shape测试，自动解析性能数据，生成对比报告
"""

import os
import sys
import csv
import json
import shutil
import argparse
import re
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


# 性能测试配置
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

# 前向算子kernel名称列表（用于区分前向和反向）
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
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行profiling
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
    
    # 找到目标Step ID的所有行
    step_rows = [row for row in rows if int(row['Step Id']) == target_step_id]
    
    if not step_rows:
        print(f"警告: 未找到 Step Id = {target_step_id} 的数据")
        return 0.0, 0.0, {}
    
    # 分析kernel执行顺序
    # 前向算子结束后开始反向算子
    # 通过检测第一个不在前向列表中的kernel来区分
    forward_phase = True
    
    for row in step_rows:
        kernel_name = row['Name'].strip()
        duration = float(row['Duration(us)'])
        
        # 判断是否为前向阶段
        if forward_phase:
            # 检查是否还在前向kernel列表中
            is_forward_kernel = any(fk in kernel_name for fk in [
                "GtTensor", "SWhere", "Greater", "SelectV2"
            ]) or any(fk == kernel_name for fk in FORWARD_KERNELS)
            
            if is_forward_kernel or kernel_name.startswith("aclnnAdd_AddAiCore_Add") and forward_time > 0:
                # 还在前向阶段
                forward_time += duration
                forward_kernels.append((kernel_name, duration))
            else:
                # 进入反向阶段
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
    
    # 返回最新的文件
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
    # 查找对应的性能分析报告
    shape_dir = Path(pypto_perf_dir) / shape_name
    summary_file = Path(pypto_perf_dir) / f"{shape_name.split('-')[0]}_backward_perf_summary.md"
    
    # 方法1: 从bubble_analysis.log中获取
    bubble_log = shape_dir / "bubble_analysis.log"
    if bubble_log.exists():
        import re
        with open(bubble_log, 'r') as f:
            content = f.read()
        
        # 查找所有核心的总工作时间，取最大值
        pattern = r'\[(AIV_\d+)\].*?Core Total Work Time: ([\d.]+)'
        matches = re.findall(pattern, content)
        
        if matches:
            max_work_time = max(float(m[1]) for m in matches)
            return max_work_time
    
    # 方法2: 从汇总报告中获取
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # 查找对应shape的执行时间
        pattern = rf'\| {shape_name} \|.*? \| (\d+\.?\d*) \|'
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
    
    return 0.0


def run_multi_shape_benchmark(model_type: str, device_id: int, output_base: str, pypto_perf_dir: str):
    """
    运行多shape基准测试并生成对比报告
    """
    print("=" * 80)
    print(f"PyTorch小算子 vs PyPTO融合算子 性能对比测试 - {model_type}")
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
        
        # 清理旧的profiling结果
        profiling_dir = Path(output_base) / "op_profiling_result"
        if profiling_dir.exists():
            shutil.rmtree(profiling_dir, ignore_errors=True)
        
        try:
            # 运行PyTorch基准测试
            benchmark_pytorch_backward(
                weight_shape, GROUP_SIZE, BIT, device_id,
                str(profiling_dir)
            )
            
            # 查找并解析kernel_details.csv
            csv_path = find_kernel_details_csv(str(profiling_dir))
            print(f"  找到性能数据: {csv_path}")
            
            # 解析性能数据（使用Step 6的数据）
            forward_time, backward_time, details = parse_kernel_details_csv(csv_path, target_step_id=6)
            
            # 获取PyPTO反向算子性能
            pypto_time = get_pypto_backward_perf(pypto_perf_dir, description)
            
            # 计算加速比
            speedup = (backward_time / pypto_time) if pypto_time > 0 else 0
            
            result = {
                'shape': weight_shape,
                'description': description,
                'pytorch_forward_us': forward_time,
                'pytorch_backward_us': backward_time,
                'pypto_backward_us': pypto_time,
                'speedup': speedup,
                'details': details
            }
            results.append(result)
            
            print(f"  ✓ PyTorch前向时间: {forward_time:.2f} us ({details['forward_count']} kernels)")
            print(f"  ✓ PyTorch反向时间: {backward_time:.2f} us ({details['backward_count']} kernels)")
            print(f"  ✓ PyPTO反向时间:   {pypto_time:.2f} us")
            print(f"  ✓ 加速比: {speedup:.2f}x")
            print()
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'shape': weight_shape,
                'description': description,
                'error': str(e)
            })
            print()
    
    # 生成对比报告
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

## 详细数据

"""
    
    for r in results:
        if 'error' not in r and 'details' in r:
            d = r['details']
            report += f"""### {r['description']}

**前向算子 ({d['forward_count']} kernels)**:
"""
            for kernel, duration in d['forward_kernels'][:5]:  # 只显示前5个
                report += f"- {kernel}: {duration:.2f} us\n"
            if len(d['forward_kernels']) > 5:
                report += f"- ... (共 {len(d['forward_kernels'])} 个kernel)\n"
            
            report += f"\n**反向算子 ({d['backward_count']} kernels)**:\n"
            for kernel, duration in d['backward_kernels'][:5]:
                report += f"- {kernel}: {duration:.2f} us\n"
            if len(d['backward_kernels']) > 5:
                report += f"- ... (共 {len(d['backward_kernels'])} 个kernel)\n"
            report += "\n"
    
    report += """---
Generated by PyPTO Performance Benchmark Tool
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("=" * 80)
    print(f"性能对比报告已生成: {report_path}")
    print(f"平均加速比: {avg_speedup:.2f}x")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PyTorch小算子 vs PyPTO融合算子性能对比")
    parser.add_argument('--model', type=str, default="3B", 
                       choices=["3B", "7B", "30B", "all"],
                       help="模型类型")
    parser.add_argument('--device_id', type=int, default=0,
                       help="NPU设备ID")
    parser.add_argument('--output', type=str, default="perf_comparison",
                       help="输出目录")
    parser.add_argument('--pypto_perf', type=str, default="output_backward_perf",
                       help="PyPTO性能数据目录")
    args = parser.parse_args()
    
    # 检查环境变量
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("警告: TILE_FWK_DEVICE_ID 未设置，尝试设置为 0")
        os.environ['TILE_FWK_DEVICE_ID'] = '0'
    
    # 设置NPU设备
    torch.npu.set_device(args.device_id)
    print(f"使用 NPU 设备: {args.device_id}\n")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 运行测试
    if args.model == "all":
        for model in ["3B", "7B", "30B"]:
            run_multi_shape_benchmark(model, args.device_id, args.output, args.pypto_perf)
            print()
    else:
        run_multi_shape_benchmark(args.model, args.device_id, args.output, args.pypto_perf)


if __name__ == "__main__":
    main()