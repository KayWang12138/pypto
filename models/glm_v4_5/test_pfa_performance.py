#!/usr/bin/env python3
# coding: utf-8
"""
PFA 性能测试脚本
用于对比 baseline、V1（稳健优化）和 V2（激进优化）版本的性能
"""
import os
import sys
import time
import json
from pathlib import Path
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose

# 导入三个版本
from glm_attention_ifa_pfa import get_pfa_config, attention_pfa as attention_pfa_baseline
from glm_attention_ifa_pfa_opt_v1 import get_pfa_config_opt_v1, attention_pfa as attention_pfa_v1
from glm_attention_ifa_pfa_opt_v2 import get_pfa_config_opt_v2, attention_pfa as attention_pfa_v2

# 从原始文件导入辅助函数
from glm_attention_ifa_pfa import gen_block_table, kv_cache_concat_bsnd, softmax


def get_test_config(device="cpu"):
    """获取测试配置"""
    b = 8
    s1 = 128
    s2 = s1
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s1 + block_size - 1) // block_size)

    actual_seq_values = [s1] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    return {
        'b': b,
        's1': s1,
        's2': s2,
        'q_d': q_d,
        'nq': nq,
        'nkv': nkv,
        'kv_layout': kv_layout,
        'softmax_scale': softmax_scale,
        'block_table_batch': block_table_batch,
        'block_size': block_size,
        'kv_num_blocks': kv_num_blocks,
        'actual_seq': actual_seq_tensor
    }


def prepare_test_data(config, device):
    """准备测试数据"""
    b = config['b']
    s1 = config['s1']
    nq = config['nq']
    nkv = config['nkv']
    d = config['q_d']
    block_size = config['block_size']
    kv_num_blocks = config['kv_num_blocks']
    block_table_batch = config['block_table_batch']
    actual_seq = config['actual_seq']
    
    max_num_blocks_per_query = (s1 + block_size - 1) // block_size
    
    q_shape = [b * s1, nq, d]
    kv_shape = [kv_num_blocks, block_size, nkv, d]
    block_table_shape = [block_table_batch, max_num_blocks_per_query]
    
    q = torch.empty(q_shape, dtype=torch.bfloat16).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch.bfloat16).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch.bfloat16).uniform_(-1, 1).to(device=device)
    
    block_table = gen_block_table(actual_seq, block_size, block_table_shape)
    
    return {
        'q': q,
        'k': k,
        'v': v,
        'block_table': block_table.to(dtype=torch.int32, device=device),
        'actual_seq': actual_seq.to(dtype=torch.int32, device=device),
        'q_shape': q_shape,
        'kv_shape': kv_shape,
        'block_table_shape': block_table_shape
    }


def run_single_test(attention_func, test_data, num_runs=10, warmup_runs=3):
    """运行单次性能测试"""
    q = test_data['q']
    k = test_data['k']
    v = test_data['v']
    block_table = test_data['block_table']
    actual_seq = test_data['actual_seq']
    q_shape = test_data['q_shape']
    
    # Warmup
    for _ in range(warmup_runs):
        out = torch.zeros(q_shape, dtype=torch.bfloat16).to(device=q.device)
        attention_func(q, k, v, block_table, actual_seq, out)
    
    # Synchronize
    torch.npu.synchronize()
    
    # Measure
    times = []
    for _ in range(num_runs):
        out = torch.zeros(q_shape, dtype=torch.bfloat16).to(device=q.device)
        
        torch.npu.synchronize()
        start = time.time()
        attention_func(q, k, v, block_table, actual_seq, out)
        torch.npu.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times)
    }


def verify_correctness(attention_func, test_data, config, ref_output=None):
    """验证正确性"""
    q = test_data['q']
    k = test_data['k']
    v = test_data['v']
    block_table = test_data['block_table']
    actual_seq = test_data['actual_seq']
    q_shape = test_data['q_shape']
    
    # Run optimized version
    out_opt = torch.zeros(q_shape, dtype=torch.bfloat16).to(device=q.device)
    attention_func(q, k, v, block_table, actual_seq, out_opt)
    
    if ref_output is None:
        # Generate reference using PyTorch
        from glm_attention_ifa_pfa import kv_cache_concat_bsnd
        from dataclasses import dataclass
        
        @dataclass
        class TempConfig:
            b: int
            s1: int
            s2: int
            n1: int
            n2: int
            q_d: int
            kv_d: int
            block_size: int
            actual_seq: torch.Tensor
        
        temp_cfg = TempConfig(
            b=config['b'],
            s1=config['s1'],
            s2=config['s2'],
            n1=config['nq'],
            n2=config['nkv'],
            q_d=config['q_d'],
            kv_d=config['q_d'],
            block_size=config['block_size'],
            actual_seq=config['actual_seq']
        )
        
        k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, temp_cfg)
        
        ref_output = torch.zeros(q_shape, dtype=torch.bfloat16).to(device=q.device)
        
        b = config['b']
        s1 = config['s1']
        nkv = config['nkv']
        d = config['q_d']
        
        for i in range(b):
            seq_len = actual_seq[i].item()
            for j in range(s1):
                cur_kv_len = j + 1
                for n2_idx in range(nkv):
                    q_bs = q[i * s1 + j]
                    k_bs = k_cache_bsnd[i, :cur_kv_len, n2_idx:n2_idx + 1].reshape(cur_kv_len, d)
                    v_bs = v_cache_bsnd[i, :cur_kv_len, n2_idx:n2_idx + 1].reshape(cur_kv_len, d)
                    
                    qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
                    qk_ele_res = qk_bmm_res * config['softmax_scale']
                    softmax_res, _, _ = softmax(qk_ele_res, True)
                    bmm2_res = torch.matmul(softmax_res, v_bs)
                    
                    ref_output[i * s1 + j] = bmm2_res
    
    # Compare
    ref_flat = np.array(ref_output.float().cpu().flatten().tolist())
    out_flat = np.array(out_opt.float().cpu().flatten().tolist())
    
    diff = np.abs(ref_flat - out_flat)
    max_diff = diff.max()
    mean_diff = diff.mean()
    mismatched = np.sum(diff > 0.001)
    
    try:
        assert_allclose(ref_flat, out_flat, rtol=0.05, atol=0.005)
        passed = True
    except AssertionError:
        passed = False
    
    return {
        'passed': passed,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'mismatched_count': mismatched,
        'total_count': len(ref_flat),
        'ref_output': ref_output
    }


def benchmark_all_versions(device_id=0):
    """对比所有版本的性能"""
    print("=" * 80)
    print("PFA 性能对比测试")
    print("=" * 80)
    
    # Setup
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    
    config = get_test_config(device)
    test_data = prepare_test_data(config, device)
    
    versions = [
        ('Baseline', attention_pfa_baseline, get_pfa_config),
        ('V1 (稳健优化)', attention_pfa_v1, get_pfa_config_opt_v1),
        ('V2 (激进优化)', attention_pfa_v2, get_pfa_config_opt_v2)
    ]
    
    results = {}
    ref_output = None
    
    for name, func, config_func in versions:
        print(f"\n{'=' * 80}")
        print(f"测试版本: {name}")
        print(f"{'=' * 80}")
        
        # Verify correctness
        print("验证正确性...")
        verify_result = verify_correctness(func, test_data, config, ref_output)
        
        if verify_result['passed']:
            print(f"✓ 正确性验证通过")
        else:
            print(f"✗ 正确性验证失败！")
            print(f"  最大差异: {verify_result['max_diff']:.6f}")
            print(f"  平均差异: {verify_result['mean_diff']:.6f}")
            print(f"  不匹配数量: {verify_result['mismatched_count']}/{verify_result['total_count']}")
        
        # Save reference output for subsequent comparisons
        if ref_output is None:
            ref_output = verify_result['ref_output']
        
        # Benchmark performance
        print("\n性能测试 (预热3次，测试10次)...")
        perf_result = run_single_test(func, test_data)
        
        print(f"  平均时间: {perf_result['mean']:.3f} ms")
        print(f"  标准差: {perf_result['std']:.3f} ms")
        print(f"  最小时间: {perf_result['min']:.3f} ms")
        print(f"  最大时间: {perf_result['max']:.3f} ms")
        print(f"  中位数: {perf_result['median']:.3f} ms")
        
        results[name] = {
            'correctness': verify_result['passed'],
            'performance': perf_result
        }
    
    # Summary
    print(f"\n{'=' * 80}")
    print("性能对比总结")
    print(f"{'=' * 80}")
    
    baseline_time = results['Baseline']['performance']['mean']
    
    print(f"\n{'版本':<20} {'时间(ms)':<12} {'提升':<10} {'正确性':<10}")
    print("-" * 60)
    
    for name in ['Baseline', 'V1 (稳健优化)', 'V2 (激进优化)']:
        time_ms = results[name]['performance']['mean']
        speedup = (baseline_time - time_ms) / baseline_time * 100
        correctness = "✓" if results[name]['correctness'] else "✗"
        
        if name == 'Baseline':
            print(f"{name:<20} {time_ms:<12.3f} {'-':<10} {correctness:<10}")
        else:
            print(f"{name:<20} {time_ms:<12.3f} {speedup:>8.1f}% {correctness:<10}")
    
    # Save results to JSON
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    results_json = {
        'config': {
            'b': config['b'],
            's1': config['s1'],
            'nq': config['nq'],
            'nkv': config['nkv'],
            'd': config['q_d'],
            'block_size': config['block_size']
        },
        'results': {}
    }
    
    for name, data in results.items():
        results_json['results'][name] = {
            'correctness': data['correctness'],
            'performance': {
                'mean_ms': data['performance']['mean'],
                'std_ms': data['performance']['std'],
                'min_ms': data['performance']['min'],
                'max_ms': data['performance']['max'],
                'median_ms': data['performance']['median']
            }
        }
    
    output_file = output_dir / f"pfa_benchmark_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    
    try:
        results = benchmark_all_versions(device_id)
        print("\n测试完成！")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
