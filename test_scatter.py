import pypto
import torch
import os
from numpy.testing import assert_allclose

def update_op(a, b, index, out):
    a1 = pypto.from_torch(a, dynamic_axis=[])
    b1 = pypto.from_torch(b, dynamic_axis=[])
    index = pypto.from_torch(index, dynamic_axis=[])
    out1 = pypto.from_torch(out, dynamic_axis=[])
    out2 = pypto.from_torch(a, dynamic_axis=[])
    update_kernel(a1, b1, index, out1, out2)
    
@pypto.jit(
    debug_options=dict(compile_debug_mode=1, runtime_debug_mode=1)
)
def update_kernel(a, b, index, out1, out2):

    pypto.set_vec_tile_shapes(16, 128)
    index1 = pypto.view(index, [1, 8], [0, 0])
    A = pypto.scatter_update(a, -2, index1, b)
    out1[:] = A + 1.0
    
    # for i in pypto.loop(1, submit_before_loop=True):
    #     continue

    index2 = pypto.view(index, [1, 8], [0, 8])
    out2[:] = pypto.scatter_update(a, -2, index2, b)

def test_update():
    """Test scatter update"""
    print("=" * 60)
    print("Test: scatter update")
    print("=" * 60)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    
    torch.manual_seed(42)
    
    a = torch.zeros((16, 128), dtype=torch.float32, device=device)
    b = torch.rand((8, 128),  dtype=torch.float32, device=device)
    index = torch.arange(16, dtype=torch.int32, device=device).reshape(1, 16)
    out = torch.zeros((16, 128), dtype=torch.float32, device=device)

    expected = torch.zeros((16, 128), dtype=torch.float32, device=device)
    expected[:8,:] = b
    expected[8:,:] = b

    expected_out = torch.zeros((16, 128), dtype=torch.float32, device=device)
    expected_out[:8,:] = b
    expected_out[8:,:] = b
    expected_out = expected_out + 1.0
 

    update_op(a, b, index, out)
    print(f"a: {a}")
    print(f"out: {out}")
    print(f"Expected_out: {expected_out}")

    assert_allclose(a.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-2, atol=1e-2)
    assert_allclose(out.cpu().float().numpy(), expected_out.cpu().float().numpy(), rtol=1e-2, atol=1e-2)
    print("✓ Basic scatter update completed successfully")

test_update()