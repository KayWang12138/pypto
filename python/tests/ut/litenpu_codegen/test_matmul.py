import pypto
import torch
import numpy as np
import unittest

def compare_cos(davinci1_input, davinci2_input):
    davinci1_input = davinci1_input.reshape(-1).astype(np.float64)
    davinci2_input = davinci2_input.reshape(-1).astype(np.float64)
    print(davinci1_input.shape)
    print(davinci2_input.shape)

    print("max diff: ", np.max(np.abs(davinci1_input-davinci2_input)))
    index = np.argmax(np.abs(davinci1_input-davinci2_input))
    print("max diff index = ", index, " dav1 value: ", davinci1_input[index], "dav2 value: ", davinci2_input[index])
    print("average diff: ", np.mean(np.abs(davinci1_input - davinci2_input)))
    ab = np.sum(np.multiply(davinci1_input, davinci2_input))
    aa = np.sqrt(np.sum(np.multiply(davinci1_input, davinci2_input)))
    bb = np.sqrt(np.sum(np.multiply(davinci1_input, davinci2_input)))
    if aa*bb == 0 and ab == 0:
        cos = 1.0
    else:
        cos = ab / (aa*bb) # this value should be greater than 0.9999 to be correct
    print(cos)
    print()
    return cos

@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
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

        golden_out = a@b
        cos_value = compare_cos(np.array(out.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

if __name__ == '__main__':
    unittest.main()
