#!/usr/bin/env python3
# coding: utf-8
"""
输出验证脚本模板

使用方法：
1. 替换 {OPERATOR_NAME} 为实际算子名
2. 替换 {SHAPE} 为实际 shape
3. 实现 PyPTO kernel 和 Ascend C API 调用

特别注意： 以下是伪代码，主要展示思想。实际使用时仅参考其流程即可，具体代码以实际为准！
"""

import os
import torch
import torch_npu

# ============ 配置 ============
BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN = 16
HEAD_DIM = 64
DEVICE_ID = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))

# ============ PyPTO Kernel ============
import pypto

@pypto.frontend.jit
def {OPERATOR_NAME}_kernel(input1, input2, output, param):
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    # ... 实现 ...
    output.move(result)


def run_output_verification():
    """输出验证"""
    torch.npu.set_device(DEVICE_ID)
    device = f'npu:{DEVICE_ID}'
    
    # 1. 生成相同输入（固定随机种子）
    torch.manual_seed(42)
    input1 = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, 
                         dtype=torch.bfloat16, device=device)
    input2 = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, 
                         dtype=torch.bfloat16, device=device)
    
    # 2. PyPTO 执行
    output_pypto = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, 
                               dtype=torch.bfloat16, device=device)
    {OPERATOR_NAME}_kernel(input1, input2, output_pypto, param)
    torch.npu.synchronize()
    
    # 3. Ascend C 执行（直接调用底层 API）
    result = torch_npu.npu_xxx(input1, input2, ...)
    output_ascend = result[0]
    torch.npu.synchronize()
    
    # 4. 对比
    diff = (output_pypto.float() - output_ascend.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    # 5. 验证
    rtol, atol = 0.01, 0.01
    matches = torch.allclose(output_pypto.float(), output_ascend.float(), 
                            rtol=rtol, atol=atol)
    
    if matches:
        print("✓ OUTPUT VERIFICATION PASSED")
    else:
        print("✗ OUTPUT VERIFICATION FAILED")
    
    return matches


if __name__ == "__main__":
    run_output_verification()