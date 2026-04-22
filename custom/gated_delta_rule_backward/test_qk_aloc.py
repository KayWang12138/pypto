#!/usr/bin/env python3
"""Test dv0 intermediates — minimal output version."""

import sys, os, math, torch
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
BT, KD, VD = 64, 128, 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)
def compare(name, g, i):
    d = (g - i).abs()
    print(f"  {name:20s}: max_diff={d.max():.6e}")

torch.manual_seed(42)
B,T,H,K,V,bt = 1,128,4,128,128,64
scale = 1.0/math.sqrt(K)
q=torch.randn(B,T,H,K)*0.5; k=torch.randn(B,T,H,K)*0.5; v=torch.randn(B,T,H,V)*0.5
g_raw=torch.randn(B,T,H)*0.1; beta=torch.rand(B,T,H)
is_=torch.randn(B,H,K,V)*0.1; do_t=torch.randn(B,T,H,V)*0.5; dht=torch.randn(B,H,K,V)*0.1
ones_bt=torch.ones(bt,bt); m_le=torch.tril(ones_bt); c_cum=torch.tril(ones_bt)
_,_,cache=forward_ref(q,k,v,g_raw,beta,is_,bt,True,1e-6,torch.eye(bt),m_le,torch.tril(ones_bt,diagonal=-1),c_cum)
b,h,c=0,0,1; t0,t1=c*bt,(c+1)*bt
qc=cache["q_norm"][b,t0:t1,h,:]; kc=cache["k_norm"][b,t0:t1,h,:]
gc_raw=g_raw.float()[b,t0:t1,h]; doc=do_t.float()[b,t0:t1,h,:]

g_cum_g=c_cum@gc_raw; decay_g=torch.exp(g_cum_g[:,None]-g_cum_g[None,:])
qk_g=qc@kc.t(); a_local_g=(qk_g*decay_g)*m_le

# Test: output only qk and a_local (both [BT, BT])
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_qk_aloc(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    gc_in: pypto.Tensor([BT],pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    m_le_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    qk_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
    aloc_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    g_cum = pypto.matmul(c_cum_in, gc_in.reshape([BT,1]), pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128,128)
    decay = pypto.exp(pypto.sub(g_cum, g_cum.reshape([1,BT])))
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)
    pypto.set_vec_tile_shapes(128,128)
    a_local = pypto.mul(pypto.mul(qk, decay), m_le_in)
    qk_out[:] = qk
    aloc_out[:] = a_local

qk_o=torch.zeros(BT,BT,device=npu,dtype=torch.float32)
aloc_o=torch.zeros(BT,BT,device=npu,dtype=torch.float32)
test_qk_aloc(to_npu(qc),to_npu(kc),to_npu(gc_raw),to_npu(c_cum),to_npu(m_le),qk_o,aloc_o)
torch_npu.npu.synchronize()

print("=== qk and a_local ===")
compare("qk", qk_g, qk_o.cpu())
compare("a_local", a_local_g, aloc_o.cpu())

# Show specific values
for i in [0, 32, 63]:
    print(f"  qk[{i},:5]: golden={qk_g[i,:5].tolist()}, impl={qk_o.cpu()[i,:5].tolist()}")
    print(f"  aloc[{i},:5]: golden={a_local_g[i,:5].tolist()}, impl={aloc_o.cpu()[i,:5].tolist()}")
