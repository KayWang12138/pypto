#!/usr/bin/env python3
"""Check if NPU tensor transfer is correct."""

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

inputs = {
    "qc": cache["q_norm"][b, t0:t1, h, :],
    "kc": cache["k_norm"][b, t0:t1, h, :],
    "vc": v.to(torch.float32)[b, t0:t1, h, :],
    "betac": beta.to(torch.float32)[b, t0:t1, h],
    "gc_raw": g_raw.to(torch.float32)[b, t0:t1, h],
    "doc": do_t.to(torch.float32)[b, t0:t1, h, :],
    "a": cache["A"][b, h, c],
    "w": cache["w"][b, h, c],
    "s_before": cache["S_before"][b, h, c],
    "v_new": cache["v_new"][b, h, c],
    "d_s": dht[b, h].to(torch.float32).clone(),
}

print("=== CPU→NPU transfer check ===")
for name, cpu_t in inputs.items():
    npu_t = cpu_t.clone().contiguous().to(npu).to(torch.float32)
    back = npu_t.cpu()
    diff = (cpu_t - back).abs().max().item()
    print(f"  {name:15s}: shape={str(cpu_t.shape):20s}, max_diff={diff:.2e}, "
          f"range=[{cpu_t.min():.6f},{cpu_t.max():.6f}]")

# Now check: does the golden backward produce the SAME dv0 with the SAME inputs?
d_s = dht[b, h].to(torch.float32).clone()
qc = inputs["qc"]
kc = inputs["kc"]
gc_raw = inputs["gc_raw"]
doc = inputs["doc"]
a_mat = inputs["a"]

# Recompute golden intermediates step by step
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
du = dv_total
dvb = a_mat.t() @ du
dv_c = dvb * inputs["betac"][:, None]

# Also check: is dvb = a^T @ dv_total?
dvb_check = a_mat.t() @ dv_total
print(f"\n=== Golden intermediates ===")
print(f"  dv0 range: [{dv0.min():.6f}, {dv0.max():.6f}]")
print(f"  dv_state range: [{dv_state.min():.6f}, {dv_state.max():.6f}]")
print(f"  dv_total range: [{dv_total.min():.6f}, {dv_total.max():.6f}]")
print(f"  dvb range: [{dvb.min():.6f}, {dvb.max():.6f}]")
print(f"  dvb_check range: [{dvb_check.min():.6f}, {dvb_check.max():.6f}]")
print(f"  dvb == dvb_check: {torch.allclose(dvb, dvb_check)}")

# Now run the full kernel and compare
from gated_delta_rule_backward_impl import gated_delta_rule_backward_factory
kernel = gated_delta_rule_backward_factory(K, V, bt)

ones_npu = torch.ones(bt, bt, device=npu, dtype=torch.float32)

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)

qr = cache["q_rstd"][b, t0:t1, h]
kr = cache["k_rstd"][b, t0:t1, h]

ds_out = torch.zeros(K, V, device=npu, dtype=torch.float32)
dq_out = torch.zeros(bt, K, device=npu, dtype=torch.float32)
dk_out = torch.zeros(bt, K, device=npu, dtype=torch.float32)
dv_out = torch.zeros(bt, V, device=npu, dtype=torch.float32)
db_out = torch.zeros(bt, device=npu, dtype=torch.float32)
dg_out = torch.zeros(bt, device=npu, dtype=torch.float32)

kernel(to_npu(inputs["qc"]), to_npu(inputs["kc"]), to_npu(inputs["vc"]),
       to_npu(inputs["betac"]), to_npu(inputs["gc_raw"]),
       to_npu(inputs["doc"]), to_npu(qr), to_npu(kr),
       to_npu(inputs["a"]), to_npu(inputs["w"]), to_npu(inputs["s_before"]),
       to_npu(inputs["v_new"]),
       to_npu(inputs["d_s"]),
       to_npu(m_le), to_npu(torch.tril(ones_bt, diagonal=-1)),
       to_npu(c_cum), torch.triu(ones_npu),
       torch.tensor([scale], device=npu, dtype=torch.float32),
       ds_out, dq_out, dk_out, dv_out, db_out, dg_out)
torch_npu.npu.synchronize()

dv_impl = dv_out.cpu()

# Now do the element-wise ratio analysis
print(f"\n=== Element-wise analysis ===")
for i in [0, 1, 32, 63]:
    # Check if dv_impl[i] ≈ dv_c[i]
    ratio = dv_impl[i] / (dv_c[i].abs() + 1e-10)
    print(f"  Row {i}: dv_impl[:5]={dv_impl[i,:5].tolist()}")
    print(f"          dv_c[:5]  ={dv_c[i,:5].tolist()}")
    print(f"          dvb[:5]   ={dvb[i,:5].tolist()}")
    print(f"          impl/dv_c ratio[:5]={ratio[:5].tolist()}")
