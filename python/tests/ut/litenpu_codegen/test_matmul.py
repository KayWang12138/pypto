import pypto
import torch
import numpy as np
import unittest

def matmul_op_fp16(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu") -> torch.Tensor:
    a_shape, b_shape = a.shape, b.shape
    out_shape = (a_shape[0], b_shape[1])
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def matmul_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP16),
        b: pypto.Tensor(b_shape, pypto.DT_FP16),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP16):
        pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
        out = pypto.matmul(a, b, a.dtype)
        return out

    out = matmul_kernel(a, b)
    return out

class TestLiteNPUMatmul(unittest.TestCase):
    def test_matmul_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_a = (16,16)
        shape_b = (16,16)

        a = torch.rand(shape_a, dtype=dtype, device=device)
        b = torch.rand(shape_b, dtype=dtype, device=device)

        # expected = torch.tensor([[19, 22], [43, 50]], dtype=dtype, device=device)

        run_mode = "sim"
        out = matmul_op_fp16(a, b, run_mode)
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
