#!/usr/bin/env python3
"""Focused diagnostic: check dv_c against golden dvb and dv_total."""

import sys, os, math, torch, numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_models_dir = os.path.join(_project_root, "models", "qwen3_next")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from gated_delta_rule_golden import forward_ref

import torch_npu

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
    qr = cache["q_rstd"][b, t0:t1, h]
    kr = cache["k_rstd"][b, t0:t1, h]

    # Golden intermediates
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

    print(f"betac stats: min={betac.min():.4f}, max={betac.max():.4f}, mean={betac.mean():.4f}")

    # Run full kernel
    from gated_delta_rule_backward_impl import gated_delta_rule_backward_factory
    kernel = gated_delta_rule_backward_factory(K, V, bt)

    def to_npu(t):
        return t.clone().contiguous().to(npu).to(torch.float32)

    ones_npu = torch.ones(bt, bt, device=npu, dtype=torch.float32)
    scale_npu = torch.tensor([scale], device=npu, dtype=torch.float32)

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
           to_npu(m_le), to_npu(m_lt), to_npu(c_cum), torch.triu(ones_npu), scale_npu,
           ds_out, dq_out, dk_out, dv_out, db_out, dg_out)
    torch_npu.npu.synchronize()

    dv_impl = dv_out.cpu()

    # Compare with various golden intermediates
    diff_dvc = (dv_c_golden - dv_impl).abs()
    diff_dvb = (dvb_golden - dv_impl).abs()
    diff_dvt = (dv_total - dv_impl).abs()
    diff_dv0 = (dv0 - dv_impl).abs()
    diff_dvs = (dv_state - dv_impl).abs()

    print(f"\ndv_impl vs various golden intermediates:")
    print(f"  vs dv_c (correct):   max_diff={diff_dvc.max():.6e}, mean={diff_dvc.mean():.6e}")
    print(f"  vs dvb (a^T@dv_tot): max_diff={diff_dvb.max():.6e}, mean={diff_dvb.mean():.6e}")
    print(f"  vs dv_total:         max_diff={diff_dvt.max():.6e}, mean={diff_dvt.mean():.6e}")
    print(f"  vs dv0:              max_diff={diff_dv0.max():.6e}, mean={diff_dv0.mean():.6e}")
    print(f"  vs dv_state:         max_diff={diff_dvs.max():.6e}, mean={diff_dvs.mean():.6e}")
    
    print(f"\n  dv_impl  range: [{dv_impl.min():.6f}, {dv_impl.max():.6f}], norm={dv_impl.norm():.6f}")
    print(f"  dv_c_gold range: [{dv_c_golden.min():.6f}, {dv_c_golden.max():.6f}], norm={dv_c_golden.norm():.6f}")
    print(f"  dvb_gold range: [{dvb_golden.min():.6f}, {dvb_golden.max():.6f}], norm={dvb_golden.norm():.6f}")
    print(f"  dv_total range: [{dv_total.min():.6f}, {dv_total.max():.6f}], norm={dv_total.norm():.6f}")

    # Check specific rows
    print(f"\n  Row 0: impl={dv_impl[0,:5].tolist()}, dv_c={dv_c_golden[0,:5].tolist()}, dvb={dvb_golden[0,:5].tolist()}")
    print(f"  Row 5: impl={dv_impl[5,:5].tolist()}, dv_c={dv_c_golden[5,:5].tolist()}, dvb={dvb_golden[5,:5].tolist()}")
    print(f"  Row 32: impl={dv_impl[32,:5].tolist()}, dv_c={dv_c_golden[32,:5].tolist()}, dvb={dvb_golden[32,:5].tolist()}")

if __name__ == "__main__":
    run()
