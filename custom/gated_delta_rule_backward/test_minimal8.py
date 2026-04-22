"""Test #8: pypto.view with valid_shape parameter."""
import os
os.environ['TILE_FWK_DEVICE_ID'] = '1'

import pypto
import torch
import torch_npu

K, H, BT = 128, 4, 64

@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kernel_with_valid(
    q_3d: pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
    dq_3d: pypto.Tensor([pypto.DYNAMIC, H, K], pypto.DT_FP32),
):
    T_val = q_3d.shape[0]
    NT_val = T_val // BT

    for i_idx in pypto.loop(NT_val, name="LOOP", idx_name="i_idx", unroll_list=[16, 1]):
        t0 = i_idx * BT
        pypto.set_vec_tile_shapes(16, 16, 128, 128)
        # Use valid_shape like forward does
        qc = pypto.view(q_3d, [BT, 1, K], [t0, 0, 0], valid_shape=[BT, 1, K])
        pypto.set_vec_tile_shapes(128, 128)
        qc_2d = pypto.reshape(qc, [BT, K], valid_shape=[BT, K])
        dq_3d[t0:t0 + BT, 0, :] = qc_2d


T = 128
print("=== Test with valid_shape ===")
try:
    q3 = torch.randn(T, H, K, dtype=torch.float32).npu()
    dq3 = torch.zeros(T, H, K, dtype=torch.float32).npu()
    kernel_with_valid(q3, dq3)
    torch.npu.synchronize()
    print("SUCCESS!")
    print(f"nan: {torch.isnan(dq3.cpu()).any()}")
except Exception as e:
    print(f"FAILED - {str(e)[:300]}")
