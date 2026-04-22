#!/usr/bin/env python3
"""Minimal dv_c kernel: only computes dv_c, everything else from golden."""

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

def run():
    B, T, H, K, V, bt = 1, 128, 4, 128, 128, 64
    scale = 1.0 / math.sqrt(K)
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K) * 0.5
    k = torch.randn(B, T, H, K) * 0.5
    v = torch.randn(B, T, H, V) * 0.5
    g_raw = torch.randn(B, T, H) * 0.1
    beta = torch.rand(B, T, H)
    initial_state = torch.randn(B, H, K, V) * 0.1
    do = torch.randn(B, T, H, V) * 0.5
    dht = torch.randn(B, H, K, V) * 0.1

    ones_bt = torch.ones(bt, bt)
    m_le = torch.tril(ones_bt)
    c_cum = torch.tril(ones_bt)
    _out, _, cache = forward_ref(q, k, v, g_raw, beta, initial_state, bt,
        True, 1e-6, torch.eye(bt), m_le, torch.tril(ones_bt, diagonal=-1), c_cum)

    b, h, c = 0, 0, 1
    t0, t1 = c * bt, (c + 1) * bt
    d_s = dht[b, h].to(torch.float32).clone()
    qc = cache["q_norm"][b, t0:t1, h, :]
    kc = cache["k_norm"][b, t0:t1, h, :]
    betac = beta.to(torch.float32)[b, t0:t1, h]
    gc_raw = g_raw.to(torch.float32)[b, t0:t1, h]
    doc = do.to(torch.float32)[b, t0:t1, h, :]
    a = cache["A"][b, h, c]

    # Golden
    g_cum = c_cum @ gc_raw
    eg = torch.exp(g_cum)
    gl = g_cum[-1]
    decay = torch.exp(g_cum[:, None] - g_cum[None, :])
    qk = qc @ kc.t()
    a_local = (qk * decay) * m_le
    dv0 = (a_local.t() @ doc) * scale
    s_tok = torch.exp(gl - g_cum)
    dv_state = (kc @ d_s) * s_tok[:, None]
    dv_total = dv_state + dv0
    dvb_golden = a.t() @ dv_total
    dv_c_golden = dvb_golden * betac[:, None]

    def to_npu(t):
        return t.clone().contiguous().to(npu).to(torch.float32)

    # Test 1: Full computation path inside kernel
    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def test_full_dvpath(
        qc_in:      pypto.Tensor([BT, K_DIM], pypto.DT_FP32),
        kc_in:      pypto.Tensor([BT, K_DIM], pypto.DT_FP32),
        doc_in:     pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
        ds_in:      pypto.Tensor([K_DIM, V_DIM], pypto.DT_FP32),
        gc_in:      pypto.Tensor([BT], pypto.DT_FP32),
        a_in:       pypto.Tensor([BT, BT], pypto.DT_FP32),
        betac_in:   pypto.Tensor([BT], pypto.DT_FP32),
        c_cum_in:   pypto.Tensor([BT, BT], pypto.DT_FP32),
        m_le_in:    pypto.Tensor([BT, BT], pypto.DT_FP32),
        scale_in:   pypto.Tensor([1], pypto.DT_FP32),
        dv_out:     pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
        dv_total_out: pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
        dvb_out:    pypto.Tensor([BT, V_DIM], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)

        # Step 1: g_cum, eg, decay
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        g_cum = pypto.matmul(c_cum_in, gc_in.reshape([BT, 1]), pypto.DT_FP32)

        pypto.set_vec_tile_shapes(128, 128)
        eg = pypto.exp(g_cum)
        gl = g_cum[BT - 1:BT, :]

        decay = pypto.exp(pypto.sub(g_cum, g_cum.reshape([1, BT])))

        # Step 2: dv0
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)

        pypto.set_vec_tile_shapes(128, 128)
        a_local = pypto.mul(pypto.mul(qk, decay), m_le_in)

        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dv0 = pypto.mul(
            pypto.matmul(a_local, doc_in, pypto.DT_FP32, a_trans=True),
            scale_in)

        # Step 3: dv_state
        pypto.set_vec_tile_shapes(128, 128)
        s_tok_2d = pypto.exp(pypto.sub(gl, g_cum))

        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dv_state = pypto.mul(pypto.matmul(kc_in, ds_in, pypto.DT_FP32), s_tok_2d)

        dv_total = pypto.add(dv_state, dv0)

        # Step 4: dvb and dv_c
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dvb = pypto.matmul(a_in, dv_total, pypto.DT_FP32, a_trans=True)

        pypto.set_vec_tile_shapes(128, 128)
        betac_2d = betac_in.reshape([BT, 1])
        dv_c = pypto.mul(dvb, betac_2d)

        # Outputs
        dv_total_out[:] = dv_total
        dvb_out[:] = dvb
        dv_out[:] = dv_c

    dv_out = torch.zeros(BT, V_DIM, device=npu, dtype=torch.float32)
    dvt_out = torch.zeros(BT, V_DIM, device=npu, dtype=torch.float32)
    dvb_out = torch.zeros(BT, V_DIM, device=npu, dtype=torch.float32)

    test_full_dvpath(
        to_npu(qc), to_npu(kc), to_npu(doc), to_npu(d_s), to_npu(gc_raw),
        to_npu(a), to_npu(betac),
        to_npu(c_cum), to_npu(m_le), to_npu(torch.tensor([scale])),
        dv_out, dvt_out, dvb_out)
    torch_npu.npu.synchronize()

    dv_impl = dv_out.cpu()
    dvt_impl = dvt_out.cpu()
    dvb_impl = dvb_out.cpu()

    print("=== Minimal dv_c path ===")
    print(f"  dv_total: max_diff={(dv_total - dvt_impl).abs().max():.6e}")
    print(f"  dvb:      max_diff={(dvb_golden - dvb_impl).abs().max():.6e}")
    print(f"  dv_c:     max_diff={(dv_c_golden - dv_impl).abs().max():.6e}")
    print(f"  dv_c range: impl=[{dv_impl.min():.6f},{dv_impl.max():.6f}], golden=[{dv_c_golden.min():.6f},{dv_c_golden.max():.6f}]")
    print(f"  dvb range: impl=[{dvb_impl.min():.6f},{dvb_impl.max():.6f}], golden=[{dvb_golden.min():.6f},{dvb_golden.max():.6f}]")

    # Detailed row check
    for i in [0, 5, 32, 63]:
        print(f"  Row {i}: impl_dv_c={dv_impl[i,:3].tolist()}, golden_dv_c={dv_c_golden[i,:3].tolist()}")
        print(f"          impl_dvb={dvb_impl[i,:3].tolist()}, golden_dvb={dvb_golden[i,:3].tolist()}")

if __name__ == "__main__":
    run()
