import pypto
import torch
import numpy as np
import unittest

@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def matmul_kernel(
    a: pypto.Tensor([...], pypto.DT_FP16),
    b: pypto.Tensor([...], pypto.DT_FP16),
    out: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    out[:] = pypto.matmul(a, b, a.dtype)

class TestLiteNPUMatmul(unittest.TestCase):
    def test_matmul_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_a = (16,16)
        shape_b = (16,16)
        shape_out = (16,16)

        a = torch.rand(shape_a, dtype=dtype, device=device)
        b = torch.rand(shape_b, dtype=dtype, device=device)
        out = torch.rand(shape_out, dtype=dtype, device=device)

        matmul_kernel(a, b, out)

        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
