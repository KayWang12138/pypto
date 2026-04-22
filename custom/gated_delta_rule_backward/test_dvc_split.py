#!/usr/bin/env python3
"""Split dv computation to isolate the error source."""

import sys, os, math, torch, numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_models_dir = os.path.join(_project_root, "models", "qwen3_next")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from gated_delta_rule_golden import forward_ref
import torch_npu
import pypto

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'
BT = 64
K_DIM = 128
V_DIM = 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

def compare(name, golden, impl):
    d = (golden - impl).abs()
    print(f"  {name:25s}: max_diff={d.max():.6e}, mean={d.mean():.6e}")

def run():
    B, T, H, K, V, bt = 1, 128, 4, 128, 128, 64
    scale = 1.0 / math.sqrt(K)
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K) * 0.5
    k = torch.randn(B, T, H, K) * 0.5
    v = torch.randn(B, T, H, V) * 0.5
    g_raw = torch.randn(B, T, H) * 0.1
    beta = torch.rand(B, T, H)
    is_ = torch.randn(B, H, K, V) * 0.1
    do_t = torch.randn(B, T, H, V) * 0.5
    dht = torch.randn(B, H, K, V) * 0.1

    ones_bt = torch.ones(bt, bt)
    m_le = torch.tril(ones_bt)
    c_cum = torch.tril(ones_bt)
    _out, _, cache = forward_ref(q, k, v, g_raw, beta, is_, bt,
        True, 1e-6, torch.eye(bt), m_le, torch.tril(ones_bt, diagonal=-1), c_cum)

    b, h, c = 0, 0, 1
    t0, t1 = c * bt, (c + 1) * bt
    d_s = dht[b, h].to(torch.float32).clone()
    qc = cache["q_norm"][b, t0:t1, h, :]
    kc = cache["k_norm"][b, t0:t1, h, :]
    gc_raw = g_raw.to(torch.float32)[b, t0:t1, h]
    doc = do_t.to(torch.float32)[b, t0:t1, h, :]
    a = cache["A"][b, h, c]

    # Golden intermediates
    g_cum_golden = c_cum @ gc_raw  # [BT]
    eg_golden = torch.exp(g_cum_golden)  # [BT]
    gl_golden = g_cum_golden[-1]  # scalar
    decay_golden = torch.exp(g_cum_golden[:, None] - g_cum_golden[None, :])  # [BT, BT]
    qk_golden = qc @ kc.t()  # [BT, BT]
    a_local_golden = (qk_golden * decay_golden) * m_le  # [BT, BT]
    dv0_golden = (a_local_golden.t() @ doc) * scale  # [BT, V]
    s_tok_golden = torch.exp(gl_golden - g_cum_golden)  # [BT]
    dv_state_golden = (kc @ d_s) * s_tok_golden[:, None]  # [BT, V]

    # Test kernel for each step
    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def test_step1(
        gc_in:      pypto.Tensor([BT], pypto.DT_FP32),
        c_cum_in:   pypto.Tensor([BT, BT], pypto.DT_FP32),
        g_cum_out:  pypto.Tensor([BT, 1], pypto.DT_FP32),
        eg_out:     pypto.Tensor([BT, 1], pypto.DT_FP32),
        decay_out:  pypto.Tensor([BT, BT], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        g_cum = pypto.matmul(c_cum_in, gc_in.reshape([BT, 1]), pypto.DT_FP32)
        pypto.set_vec_tile_shapes(128, 128)
        eg = pypto.exp(g_cum)
        decay = pypto.exp(pypto.sub(g_cum, g_cum.reshape([1, BT])))
        g_cum_out[:] = g_cum
        eg_out[:] = eg
        decay_out[:] = decay

    # Run step 1
    g_cum_out = torch.zeros(BT, 1, device=npu, dtype=torch.float32)
    eg_out = torch.zeros(BT, 1, device=npu, dtype=torch.float32)
    decay_out = torch.zeros(BT, BT, device=npu, dtype=torch.float32)
    test_step1(to_npu(gc_raw), to_npu(c_cum), g_cum_out, eg_out, decay_out)
    torch_npu.npu.synchronize()

    print("=== Step 1: g_cum, eg, decay ===")
    compare("g_cum", g_cum_golden[:, None], g_cum_out.cpu())
    compare("eg", eg_golden[:, None], eg_out.cpu())
    compare("decay", decay_golden, decay_out.cpu())

    # Check qk and a_local
    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def test_step2(
        qc_in:    pypto.Tensor([BT, K_DIM], pypto.DT_FP32),
        kc_in:    pypto.Tensor([BT, K_DIM], pypto.DT_FP32),
        decay_in: pypto.Tensor([BT, BT], pypto.DT_FP32),
        m_le_in:  pypto.Tensor([BT, BT], pypto.DT_FP32),
        qk_out:   pypto.Tensor([BT, BT], pypto.DT_FP32),
        aloc_out: pypto.Tensor([BT, BT], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(128, 128)
        a_local = pypto.mul(pypto.mul(qk, decay_in), m_le_in)
        qk_out[:] = qk
        aloc_out[:] = a_local

    qk_out = torch.zeros(BT, BT, device=npu, dtype=torch.float32)
    aloc_out = torch.zeros(BT, BT, device=npu, dtype=torch.float32)
    test_step2(to_npu(qc), to_npu(kc), decay_out, to_npu(m_le), qk_out, aloc_out)
    torch_npu.npu.synchronize()

    print("\n=== Step 2: qk, a_local ===")
    compare("qk", qk_golden, qk_out.cpu())
    compare("a_local", a_local_golden, aloc_out.cpu())

    # Check dv0
    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def test_step3(
        aloc_in:  pypto.Tensor([BT, BT], pypto.DT_FP32),
        doc_in:   pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
        scale_in: pypto.Tensor([1], pypto.DT_FP32),
        dv0_out:  pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dv0 = pypto.mul(
            pypto.matmul(aloc_in, doc_in, pypto.DT_FP32, a_trans=True),
            scale_in)
        dv0_out[:] = dv0

    dv0_out = torch.zeros(BT, V_DIM, device=npu, dtype=torch.float32)
    test_step3(aloc_out, to_npu(doc), to_npu(torch.tensor([scale])), dv0_out)
    torch_npu.npu.synchronize()

    print("\n=== Step 3: dv0 ===")
    compare("dv0", dv0_golden, dv0_out.cpu())

    # Check dv_state
    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def test_step4(
        kc_in:  pypto.Tensor([BT, K_DIM], pypto.DT_FP32),
        ds_in:  pypto.Tensor([K_DIM, V_DIM], pypto.DT_FP32),
        gl_in:  pypto.Tensor([1, 1], pypto.DT_FP32),
        gcum_in: pypto.Tensor([BT, 1], pypto.DT_FP32),
        dvs_out: pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_vec_tile_shapes(128, 128)
        s_tok_2d = pypto.exp(pypto.sub(gl_in, gcum_in))
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dv_state = pypto.mul(pypto.matmul(kc_in, ds_in, pypto.DT_FP32), s_tok_2d)
        dvs_out[:] = dv_state

    gl_npu = g_cum_out[BT-1:BT, :].clone()
    dvs_out = torch.zeros(BT, V_DIM, device=npu, dtype=torch.float32)
    test_step4(to_npu(kc), to_npu(d_s), gl_npu, g_cum_out, dvs_out)
    torch_npu.npu.synchronize()

    print("\n=== Step 4: dv_state ===")
    compare("dv_state", dv_state_golden, dvs_out.cpu())

if __name__ == "__main__":
    run()
