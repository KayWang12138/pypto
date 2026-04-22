#!/usr/bin/env python3
"""Test if pypto.reshape works correctly for 1D → 2D."""

import sys, os, torch, torch_npu
import pypto

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'

BT = 64

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_reshape_kernel(
    betac:    pypto.Tensor([BT], pypto.DT_FP32),
    dvb:      pypto.Tensor([BT, 128], pypto.DT_FP32),
    out_mul:  pypto.Tensor([BT, 128], pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(128, 128)
    betac_2d = betac.reshape([BT, 1])
    result = pypto.mul(dvb, betac_2d)
    out_mul[:] = result

# Create test data
betac_cpu = torch.rand(BT) * 0.5 + 0.25  # [0.25, 0.75]
dvb_cpu = torch.randn(BT, 128) * 0.5

betac_npu = betac_cpu.clone().to(npu)
dvb_npu = dvb_cpu.clone().to(npu)
out_npu = torch.zeros(BT, 128, device=npu, dtype=torch.float32)

test_reshape_kernel(betac_npu, dvb_npu, out_npu)
torch_npu.npu.synchronize()

# Compare
expected = dvb_cpu * betac_cpu[:, None]  # [BT, 128] * [BT, 1]
actual = out_npu.cpu()

diff = (expected - actual).abs()
print(f"reshape test: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")
print(f"  expected range: [{expected.min():.6f}, {expected.max():.6f}]")
print(f"  actual range: [{actual.min():.6f}, {actual.max():.6f}]")

# Also test if the reshape result is correct by checking specific values
print(f"  betac[0]={betac_cpu[0]:.6f}, dvb[0,0]={dvb_cpu[0,0]:.6f}, expected[0,0]={expected[0,0]:.6f}, actual[0,0]={actual[0,0]:.6f}")
print(f"  betac[5]={betac_cpu[5]:.6f}, dvb[5,10]={dvb_cpu[5,10]:.6f}, expected[5,10]={expected[5,10]:.6f}, actual[5,10]={actual[5,10]:.6f}")

# Check if actual ≈ dvb (betac multiplication not applied)
diff_no_mul = (dvb_cpu - actual).abs()
print(f"  diff from dvb (no mul): max={diff_no_mul.max():.6e}")

if diff.max() > 1e-4:
    print("FAIL: reshape or broadcast not working correctly!")
else:
    print("PASS: reshape works correctly")
