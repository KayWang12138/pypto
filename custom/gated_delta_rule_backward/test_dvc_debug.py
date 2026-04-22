#!/usr/bin/env python3
"""Targeted test: feed golden dv_total directly to kernel to isolate dv_c error."""

import sys, os, math, torch, numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_models_dir = os.path.join(_project_root, "models", "qwen3_next")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from gated_delta_rule_golden import forward_ref, l2norm_bwd_chunk

import torch_npu
import pypto

device_id = 4
torch.npu.set_device(device_id)
npu = f'npu:{device_id}'

def run():
    B, T, H, K, V, bt = 1, 128, 4, 128, 128, 64
    scale = 1.0 / math.sqrt(K)
    seed = 42
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K) * 0.5
    k = torch.randn(B, T, H, K) * 0.5
    v = torch.randn(B, T, H, V) * 0.5
    g_raw = torch.randn(B, T, H) * 0.1
    beta = torch.rand(B, T, H)
    initial_state = torch.randn(B, H, K, V) * 0.1
    do = torch.randn(B, T, H, V) * 0.5
    dht = torch.randn(B, H, K, V) * 0.1

    ones_bt = torch.ones(bt, bt)
    i_mat = torch.eye(bt)
    m_le = torch.tril(ones_bt)
    m_lt = torch.tril(ones_bt, diagonal=-1)
    c_cum = torch.tril(ones_bt)

    _out, _, cache = forward_ref(q, k, v, g_raw, beta, initial_state, bt,
        True, 1e-6, i_mat, m_le, m_lt, c_cum)

    b, h, c = 0, 0, 1
    t0, t1 = c * bt, (c + 1) * bt
    d_s = dht[b, h].to(torch.float32).clone()

    qc = cache["q_norm"][b, t0:t1, h, :]
    kc = cache["k_norm"][b, t0:t1, h, :]
    vc = v.to(torch.float32)[b, t0:t1, h, :]
    betac = beta.to(torch.float32)[b, t0:t1, h]
    gc_raw = g_raw.to(torch.float32)[b, t0:t1, h]
    doc = do.to(torch.float32)[b, t0:t1, h, :]
    a = cache["A"][b, h, c]
    w = cache["w"][b, h, c]
    s_before = cache["S_before"][b, h, c]
    v_new = cache["v_new"][b, h, c]

    # Compute golden dv_total
    g_cum = c_cum @ gc_raw
    eg = torch.exp(g_cum)
    gl = g_cum[-1]
    decay = torch.exp(g_cum[:, None] - g_cum[None, :])
    qk = qc @ kc.t()
    a_local = (qk * decay) * m_le
    dv0 = (a_local.t() @ doc) * scale
    s_tok = torch.exp(gl - g_cum)
    dv_state = (kc @ d_s) * s_tok[:, None]
    dv_total_golden = dv_state + dv0

    # Golden dvb and dv_c
    du_golden = dv_total_golden
    dvb_golden = a.t() @ du_golden
    dv_c_golden = dvb_golden * betac[:, None]

    print(f"Golden dv_total range: [{dv_total_golden.min():.6f}, {dv_total_golden.max():.6f}]")
    print(f"Golden dvb range: [{dvb_golden.min():.6f}, {dvb_golden.max():.6f}]")
    print(f"Golden dv_c range: [{dv_c_golden.min():.6f}, {dv_c_golden.max():.6f}]")

    # Test 1: Minimal kernel that only computes dvb @ betac
    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def minimal_dvb_kernel(
        a_in:       pypto.Tensor([bt, bt], pypto.DT_FP32),
        du_in:      pypto.Tensor([bt, V], pypto.DT_FP32),
        betac_in:   pypto.Tensor([bt], pypto.DT_FP32),
        dvb_out:    pypto.Tensor([bt, V], pypto.DT_FP32),
        dvc_out:    pypto.Tensor([bt, V], pypto.DT_FP32),
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dvb = pypto.matmul(a_in, du_in, pypto.DT_FP32, a_trans=True)
        dvb_out[:] = dvb

        pypto.set_vec_tile_shapes(128, 128)
        betac_2d = betac_in.reshape([bt, 1])
        dvc = pypto.mul(dvb, betac_2d)
        dvc_out[:] = dvc

    def to_npu(t):
        return t.clone().contiguous().to(npu).to(torch.float32)

    dvb_npu = torch.zeros(bt, V, device=npu, dtype=torch.float32)
    dvc_npu = torch.zeros(bt, V, device=npu, dtype=torch.float32)

    minimal_dvb_kernel(to_npu(a), to_npu(du_golden), to_npu(betac), dvb_npu, dvc_npu)
    torch_npu.npu.synchronize()

    dvb_impl = dvb_npu.cpu()
    dvc_impl = dvc_npu.cpu()

    print(f"\nTest 1 (minimal kernel with golden dv_total):")
    print(f"  dvb: max_diff={(dvb_golden - dvb_impl).abs().max():.6e}")
    print(f"  dv_c: max_diff={(dv_c_golden - dvc_impl).abs().max():.6e}")
    print(f"  dvb_impl range: [{dvb_impl.min():.6f}, {dvb_impl.max():.6f}]")
    print(f"  dvc_impl range: [{dvc_impl.min():.6f}, {dvc_impl.max():.6f}]")

    # Test 2: Run full kernel and check dv output
    from gated_delta_rule_backward_impl import gated_delta_rule_backward_factory
    kernel = gated_delta_rule_backward_factory(K, V, bt)

    ones_npu = torch.ones(bt, bt, device=npu, dtype=torch.float32)
    c_rcum_npu = torch.triu(ones_npu)

    qr = cache["q_rstd"][b, t0:t1, h]
    kr = cache["k_rstd"][b, t0:t1, h]

    ds_out = torch.zeros(K, V, device=npu, dtype=torch.float32)
    dq_out = torch.zeros(bt, K, device=npu, dtype=torch.float32)
    dk_out = torch.zeros(bt, K, device=npu, dtype=torch.float32)
    dv_out = torch.zeros(bt, V, device=npu, dtype=torch.float32)
    db_out = torch.zeros(bt, device=npu, dtype=torch.float32)
    dg_out = torch.zeros(bt, device=npu, dtype=torch.float32)

    kernel(to_npu(qc), to_npu(kc), to_npu(vc), to_npu(betac), to_npu(gc_raw),
           to_npu(doc), to_npu(qr), to_npu(kr),
           to_npu(a), to_npu(w), to_npu(s_before), to_npu(v_new),
           to_npu(d_s),
           to_npu(m_le), to_npu(m_lt), to_npu(c_cum), c_rcum_npu,
           torch.tensor([scale], device=npu, dtype=torch.float32),
           ds_out, dq_out, dk_out, dv_out, db_out, dg_out)
    torch_npu.npu.synchronize()

    dv_full_impl = dv_out.cpu()
    print(f"\nTest 2 (full kernel dv_c):")
    print(f"  dv_c: max_diff={(dv_c_golden - dv_full_impl).abs().max():.6e}")
    print(f"  dv_full range: [{dv_full_impl.min():.6f}, {dv_full_impl.max():.6f}]")

    # Check if full kernel dv_c ≈ golden dvb (betac not applied)
    print(f"  diff from golden dvb: {(dvb_golden - dv_full_impl).abs().max():.6e}")

    # Check if dv_c ≈ dv_total (shouldn't be, but let's check)
    print(f"  diff from golden dv_total: {(dv_total_golden - dv_full_impl).abs().max():.6e}")

if __name__ == "__main__":
    run()
