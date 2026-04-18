import pypto
import torch
import numpy as np
import unittest

def add_op_fp16(run_mode: str = "npu") -> torch.Tensor:
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_kernel(
        a: pypto.Tensor([...], pypto.DT_FP16),
        b: pypto.Tensor([...], pypto.DT_FP16),
        out: pypto.Tensor([...], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(16,16)
        out[:] = a + b

    return add_kernel

class TestLiteNPUAdd(unittest.TestCase):
    def test_add_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_a = (16,16)
        shape_b = (16,16)
        shape_out = (16,16)

        a = torch.rand(shape_a, dtype=dtype, device=device)
        b = torch.rand(shape_b, dtype=dtype, device=device)
        out = torch.rand(shape_out, dtype=dtype, device=device)

        run_mode = "sim"
        add_op_fp16(run_mode)(a, b, out)

        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
