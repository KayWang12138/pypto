import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

verify_options = {
      "enable_pass_verify": True,
      "pass_verify_save_tensor": True,
}
def concat_op(b: torch.Tensor) -> torch.Tensor:
    b_shape = b.shape
    out_shape = (64, 64)
    mode = pypto.RunMode.NPU
    
    @pypto.frontend.jit(
    # host_options={"only_codegen": True}, 
    runtime_options={"run_mode": mode}, debug_options={"compile_debug_mode": 1}
    # ,verify_options=verify_options
    )
    def concat_kernel(
        b: pypto.Tensor(b_shape, pypto.DT_BF16)
    ) -> pypto.Tensor(out_shape, pypto.DT_BF16):
        pypto.set_options('profile_enable', True)
        pypto.set_debug_options(runtime_debug_mode=1)

        # 构造c
        pypto.set_vec_tile_shapes(64, 64)

        # # 1. 两层assemble
        d = pypto.full([16, 16], 0.0, pypto.DT_BF16)
        c1 = pypto.concat([b, d, d, d], 1)
        c2 = pypto.concat([d, b, d, d], 1)
        c3 = pypto.concat([d, d, b, d], 1)
        c4 = pypto.concat([d, d, d, b], 1)
        c = pypto.concat([c1, c2, c3, c4], 0)
        # 2. 精度对的版本
        # d = pypto.full([16, 16], 0.0, pypto.DT_BF16)
        # c = pypto.concat([b, d, d, d, d, b, d, d, d, d, b, d, d, d, d, b], 1)
        # c = pypto.reshape(c, [16, 4, 64])
        # pypto.set_vec_tile_shapes(64, 64, 64)
        # c = pypto.transpose(c, 0, 1)
        # c = pypto.reshape(c, [64, 64])
        
        return c


    out = concat_kernel(b)
    return out


def test_concat(device_id: int = 5):
    """Test basic concat"""
    print("=" * 60)
    print("Test: Basic Concat")
    print("=" * 60)
    
    device = f'npu:{device_id}'
    
    torch.manual_seed(42)
    b = torch.rand((16, 16), dtype=torch.bfloat16, device=device)
    expected = torch.block_diag(b, b, b, b)
    # expected = torch.block_diag(0, 0, 0, b)

    out = concat_op(b)
    # assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-2, atol=1e-2)
    a = out.cpu().float().numpy()
    for row in a:
        print('[' + ','.join(f'{x:.2f}' for x in row) + ']')
    print("-------------------------------")
    print("-------------------------------")
    b = expected.cpu().float().numpy()
    for row in b:
        print('[' + ','.join(f'{x:.2f}' for x in row) + ']')
    print(f"Output: {out}")
    print(f"Expected: {expected}")

    assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-2, atol=1e-2)
    print("✓ Basic concat completed successfully")

if __name__ == "__main__":
    test_concat(5)