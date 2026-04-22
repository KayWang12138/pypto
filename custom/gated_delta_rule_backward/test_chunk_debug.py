#!/usr/bin/env python3
"""Single-chunk diagnostic: compare ALL intermediate values between golden and impl."""

import sys, os, math, torch, numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_models_dir = os.path.join(_project_root, "models", "qwen3_next")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from gated_delta_rule_golden import forward_ref, l2norm_bwd_chunk

def compare(name, golden_t, impl_t):
    diff = (golden_t - impl_t).abs()
    rel = diff.max() / (golden_t.abs().max() + 1e-8)
    print(f"  {name:30s}: max_diff={diff.max():.6e}, mean={diff.mean():.6e}, "
          f"rel={rel:.4f}, golden=[{golden_t.min():.4f},{golden_t.max():.4f}], "
          f"impl=[{impl_t.min():.4f},{impl_t.max():.4f}]")

def run_diagnostic():
    B, T, H, K, V, bt = 1, 128, 4, 128, 128, 64
    nt = T // bt
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
    c_rcum = torch.triu(ones_bt)

    _out, _fs, cache = forward_ref(q, k, v, g_raw, beta, initial_state, bt,
        use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
        i=i_mat, m_le=m_le, m_lt=m_lt, c_cum=c_cum)

    b, h, c = 0, 0, 1  # last chunk
    t0, t1 = c * bt, (c + 1) * bt

    q_used = cache["q_norm"]
    k_used = cache["k_norm"]
    v32 = v.to(torch.float32)
    beta32 = beta.to(torch.float32)
    g_raw32 = g_raw.to(torch.float32)
    do32 = do.to(torch.float32)
    q_rstd = cache["q_rstd"]
    k_rstd = cache["k_rstd"]

    # Slice chunk inputs
    qc = q_used[b, t0:t1, h, :]
    kc = k_used[b, t0:t1, h, :]
    vc = v32[b, t0:t1, h, :]
    betac = beta32[b, t0:t1, h]
    gc_raw = g_raw32[b, t0:t1, h]
    doc = do32[b, t0:t1, h, :]
    a = cache["A"][b, h, c]
    w = cache["w"][b, h, c]
    s_before = cache["S_before"][b, h, c]
    v_new = cache["v_new"][b, h, c]
    qr = q_rstd[b, t0:t1, h]
    kr = k_rstd[b, t0:t1, h]
    d_s = dht[b, h].to(torch.float32).clone()

    # ======== Golden: compute ALL intermediates ========
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
    v_scaled = v_new * s_tok[:, None]
    dk_state = v_scaled @ d_s.t()

    q_eff = qc * eg[:, None]
    d_s_decay = d_s * torch.exp(gl)
    d_s_q = (q_eff.t() @ doc) * scale
    d_s_w = w.t() @ dv_total
    d_s_out = d_s_decay + d_s_q - d_s_w

    dq1 = (doc @ s_before.t()) * eg[:, None] * scale
    dq_c = dq1.clone()
    dg_cum = (dq1 * qc).sum(-1)
    dk_c = dk_state.clone()
    scalar = (kc * dk_state).sum(-1)
    dg_cum = dg_cum - scalar
    delta_scalar_sum = scalar.sum()
    delta_sb_ds = torch.exp(gl) * (s_before * d_s).sum()

    d_a_base = (doc @ v_new.t()) * m_le * scale
    dq_c = dq_c + (d_a_base * decay) @ kc
    dk_c = dk_c + (d_a_base * decay).t() @ qc
    a_base = (qk * decay) * m_le
    tmp = d_a_base * a_base
    dg_cum_local = tmp.sum(-1) - tmp.sum(-2)

    # WY
    dw = -(dv_total @ s_before.t())
    du = dv_total
    vb = vc * betac[:, None]
    kbg = kc * (betac[:, None] * eg[:, None])

    dvb = a.t() @ du
    dkbg = a.t() @ dw
    dv_c = dvb * betac[:, None]
    db_c_init = (dvb * vc).sum(-1)

    dk_c_wy1 = dk_c + dkbg * (betac[:, None] * eg[:, None])
    db_c_wy1 = db_c_init + (dkbg * (kc * eg[:, None])).sum(-1)
    dg_cum_wy1 = dg_cum + (dkbg * kbg).sum(-1)

    d_a = dw @ kbg.t() + du @ vb.t()
    d_l = -(a.t() @ (d_a @ a.t()))
    d_l = d_l * m_lt
    kkt = kc @ kc.t()

    db_c_wy2 = db_c_wy1 + (d_l * (kkt * decay)).sum(-1)
    l_mat = (betac[:, None] * kkt) * decay * m_lt
    tmp2 = d_l * l_mat
    dg_cum_wy2 = dg_cum_wy1 + tmp2.sum(-1) - tmp2.sum(-2)
    m_mat = d_l * (betac[:, None] * decay)
    dk_c_wy2 = dk_c_wy1 + (m_mat + m_mat.t()) @ kc

    dg_cum_total = dg_cum_wy2 + dg_cum_local
    dg_raw_c = c_rcum @ dg_cum_total
    # Add delta corrections
    dg_raw_c = dg_raw_c + delta_scalar_sum + delta_sb_ds

    dq_raw_c = l2norm_bwd_chunk(qc, qr, dq_c)
    dk_raw_c = l2norm_bwd_chunk(kc, kr, dk_c_wy2)

    # ======== Now run kernel ========
    import torch_npu
    device_id = 4
    torch.npu.set_device(device_id)
    npu = f'npu:{device_id}'

    ones_npu = torch.ones(bt, bt, device=npu, dtype=torch.float32)
    m_le_npu = torch.tril(ones_npu)
    m_lt_npu = torch.tril(ones_npu, diagonal=-1)
    c_cum_npu = torch.tril(ones_npu)
    c_rcum_npu = torch.triu(ones_npu)
    scale_npu = torch.tensor([scale], device=npu, dtype=torch.float32)

    from gated_delta_rule_backward_impl import gated_delta_rule_backward_factory
    kernel = gated_delta_rule_backward_factory(K, V, bt)

    def to_npu(t):
        return t.clone().contiguous().to(npu).to(torch.float32)

    d_s_npu_out = torch.zeros(K, V, device=npu, dtype=torch.float32)
    dq_npu = torch.zeros(bt, K, device=npu, dtype=torch.float32)
    dk_npu = torch.zeros(bt, K, device=npu, dtype=torch.float32)
    dv_npu = torch.zeros(bt, V, device=npu, dtype=torch.float32)
    db_npu = torch.zeros(bt, device=npu, dtype=torch.float32)
    dg_npu = torch.zeros(bt, device=npu, dtype=torch.float32)

    kernel(to_npu(qc), to_npu(kc), to_npu(vc), to_npu(betac), to_npu(gc_raw),
           to_npu(doc), to_npu(qr), to_npu(kr),
           to_npu(a), to_npu(w), to_npu(s_before), to_npu(v_new),
           to_npu(d_s),
           m_le_npu, m_lt_npu, c_cum_npu, c_rcum_npu, scale_npu,
           d_s_npu_out, dq_npu, dk_npu, dv_npu, db_npu, dg_npu)
    torch_npu.npu.synchronize()

    dq_impl = dq_npu.cpu()
    dk_impl = dk_npu.cpu()
    dv_impl = dv_npu.cpu()
    db_impl = db_npu.cpu()
    dg_impl = dg_npu.cpu()
    ds_impl = d_s_npu_out.cpu()

    print("=== Per-output comparison (chunk c=1, b=0, h=0) ===")
    compare("dv_c", dv_c, dv_impl)
    compare("db_c", db_c_wy2, db_impl)
    compare("dk_c (before L2)", dk_c_wy2, dk_impl)
    compare("dk_raw_c (after L2)", dk_raw_c, dk_impl)
    compare("dq_c (before L2)", dq_c, dq_impl)
    compare("dq_raw_c (after L2)", dq_raw_c, dq_impl)
    compare("dg_raw_c", dg_raw_c, dg_impl)
    compare("d_s_out", d_s_out, ds_impl)

    # Also check golden's dv_c = dvb * betac step by step
    print("\n=== Golden dv_c components ===")
    print(f"  dvb range: [{dvb.min():.6f}, {dvb.max():.6f}], norm={dvb.norm():.6f}")
    print(f"  dv_c range: [{dv_c.min():.6f}, {dv_c.max():.6f}], norm={dv_c.norm():.6f}")
    print(f"  dv_impl range: [{dv_impl.min():.6f}, {dv_impl.max():.6f}], norm={dv_impl.norm():.6f}")
    print(f"  dv_total range: [{dv_total.min():.6f}, {dv_total.max():.6f}]")

    # Check dvb = a^T @ du
    dvb_check = a.t() @ du
    print(f"  dvb_check range: [{dvb_check.min():.6f}, {dvb_check.max():.6f}]")

if __name__ == "__main__":
    run_diagnostic()
