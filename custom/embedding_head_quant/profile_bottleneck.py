import os
import pypto
import torch
import torch_npu
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from embedding_head_quant import embedding_head_quant_golden

def create_profiled_kernel(shape, eps=1e-4, min_v=-128.0, max_v=127.0):
    @pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU},
        debug_options={"runtime_debug_mode": 1}
    )
    def embedding_head_quant_kernel(
        weight: pypto.Tensor(shape, pypto.DT_FP32),
        scale: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(32, 32)

        protected_scale = pypto.maximum(scale, eps)
        normalized = pypto.div(weight, protected_scale)
        rounded = pypto.round(normalized, decimals=0)
        clamped = pypto.clip(rounded, min_v, max_v)
        output = pypto.mul(clamped, protected_scale)

        return output

    return embedding_head_quant_kernel

def main():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: TILE_FWK_DEVICE_ID not set")
        return
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
    torch.npu.set_device(device_id)

    shape = (256, 256)
    device = f'npu:{device_id}'

    print("Creating kernel with performance profiling enabled...")
    kernel = create_profiled_kernel(shape)

    print("Generating test data...")
    torch.manual_seed(42)
    weight_torch = torch.randn(shape, dtype=torch.float32, device=device) * 50
    scale_torch = torch.rand(shape, dtype=torch.float32, device=device) * 3 + 0.5

    print("Running kernel with profiling...")
    output_torch = kernel(weight_torch, scale_torch)

    print("Verifying correctness...")
    expected = embedding_head_quant_golden(weight_torch, scale_torch)
    max_diff = (output_torch - expected).abs().max().item()
    print(f"Max error: {max_diff:.6f}")

    print("\nPerformance profiling data generated in output/ directory")
    print("Check for:")
    print("  - merged_swimlane.json")
    print("  - machine_runtime_operator_trace.json")
    print("  - bubble_analysis.log")

if __name__ == "__main__":
    main()
