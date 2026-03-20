#!/usr/bin/env python3
# coding: utf-8
"""
性能对比脚本模板

使用方法：
1. 替换 {OPERATOR_NAME} 为实际算子名
2. 替换 {SHAPE} 为实际 shape
3. 实现 PyPTO debug kernel 和 Ascend C API 调用

特别注意： 以下是伪代码，主要展示思想。实际使用时仅参考其流程即可，具体代码以实际为准！
"""

import os
import re
import glob
import torch
import torch_npu
import pypto

# ============ 配置 ============
BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN = 16
HEAD_DIM = 64
DEVICE_ID = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


def get_pypto_kernel_time_us(output_dir):
    """从 bubble_analysis.log 获取 PyPTO kernel 时间 (μs)"""
    output_dirs = glob.glob(os.path.join(output_dir, 'output_*'))
    if not output_dirs:
        return None
    latest_dir = max(output_dirs, key=lambda x: os.path.getmtime(x))
    log_path = os.path.join(latest_dir, 'bubble_analysis.log')
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    matches = re.findall(r'Core Total Work Time:\s+([\d.]+)', content)
    times = [float(m) for m in matches if float(m) > 0]
    
    return max(times) if times else None


# ============ PyPTO Debug Kernel ============
@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})
def {OPERATOR_NAME}_kernel_debug(input1, input2, output, param):
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    # ... 与普通版本相同的实现 ...
    output.move(result)


def run_performance_comparison():
    """性能对比"""
    torch.npu.set_device(DEVICE_ID)
    device = f'npu:{DEVICE_ID}'
    
    print(f"Device: {device}")
    print(f"Config: B={BATCH_SIZE}, N={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}")
    
    # 1. PyPTO kernel 时间
    print("\n[1] PyPTO Kernel Time")
    
    input1 = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, 
                         dtype=torch.bfloat16, device=device)
    input2 = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, 
                         dtype=torch.bfloat16, device=device)
    output = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, 
                         dtype=torch.bfloat16, device=device)
    
    {OPERATOR_NAME}_kernel_debug(input1, input2, output, param)
    torch.npu.synchronize()
    
    pypto_time = get_pypto_kernel_time_us('output')
    
    # 2. Ascend C kernel 时间 (msprof)
    print("\n[2] Ascend C Kernel Time (msprof)")
    
    # 创建测试脚本
    with open('test_ascend_c.py', 'w') as f:
        f.write(f'''
import torch
import torch_npu
torch.npu.set_device({DEVICE_ID})
device = 'npu:{DEVICE_ID}'

# 准备数据
input1 = torch.randn({BATCH_SIZE}, {NUM_HEADS}, {SEQ_LEN}, {HEAD_DIM}, dtype=torch.bfloat16, device=device)
input2 = torch.randn({BATCH_SIZE}, {NUM_HEADS}, {SEQ_LEN}, {HEAD_DIM}, dtype=torch.bfloat16, device=device)

# warmup
for _ in range(3):
    torch_npu.npu_xxx(input1, input2, ...)
torch.npu.synchronize()

# test
torch_npu.npu_xxx(input1, input2, ...)
torch.npu.synchronize()
''')
    
    import subprocess
    subprocess.run(
        'msprof --output=./msprof_output --task-time=on --ai-core=on python3 test_ascend_c.py',
        shell=True, capture_output=True
    )
    
    # 读取 op_summary.csv
    ascend_time = None
    csv_files = glob.glob('msprof_output/PROF_*/mindstudio_profiler_output/op_summary_*.csv')
    if csv_files:
        import csv
        with open(csv_files[0], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if '{OPERATOR_NAME}' in row.get('Op Name', ''):
                    ascend_time = float(row.get('Task Duration(us)', 0))
                    break
    
    # 3. 输出结果
    print("\n" + "=" * 60)
    print("Kernel Time Summary")
    print("=" * 60)
    print(f"\n{'Kernel':<30} {'Time (μs)':<15}")
    print("-" * 45)
    
    if pypto_time:
        print(f"{'PyPTO':<30} {pypto_time:<15.2f}")
    
    if ascend_time:
        print(f"{'Ascend C':<30} {ascend_time:<15.2f}")
    
    if pypto_time and ascend_time:
        ratio = pypto_time / ascend_time
        print(f"\n比值 (PyPTO / Ascend C): {ratio:.2f}x")


if __name__ == "__main__":
    run_performance_comparison()