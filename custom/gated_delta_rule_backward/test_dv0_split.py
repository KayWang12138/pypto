#!/usr/bin/env python3
"""Test qk and a_local separately."""

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
BT, KD = 64, 128

def to_npu(t):
    return t.clone().contiguous().to(npu).to(torch.float32)
def compare(name, g, i):
    d = (g - i).abs()
    print(f"  {name:20s}: max_diff={d.max():.6e}, mean={d.mean():.6e}")

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

g_cum=c_cum@gc_raw; decay_g=torch.exp(g_cum[:,None]-g_cum[None,:])
qk_g=qc@kc.t(); a_local_g=(qk_g*decay_g)*m_le

# Test 1: Just qk
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_qk(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    qk_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)
    qk_out[:] = qk

qk_out=torch.zeros(BT,BT,device=npu,dtype=torch.float32)
test_qk(to_npu(qc),to_npu(kc),qk_out)
torch_npu.npu.synchronize()
print("=== qk (qc @ kc^T) ===")
compare("qk", qk_g, qk_out.cpu())

# Test 2: decay + qk
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_decay_qk(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    gc_in: pypto.Tensor([BT],pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    decay_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
    qk_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    g_cum = pypto.matmul(c_cum_in, gc_in.reshape([BT,1]), pypto.DT_FP32)
    pypto.set_vec_tile_shapes(128,128)
    decay = pypto.exp(pypto.sub(g_cum, g_cum.reshape([1,BT])))
    decay_out[:] = decay
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, b_trans=True)
    qk_out[:] = qk

decay_out=torch.zeros(BT,BT,device=npu,dtype=torch.float32)
qk_out2=torch.zeros(BT,BT,device=npu,dtype=torch.float32)
test_decay_qk(to_npu(qc),to_npu(kc),to_npu(gc_raw),to_npu(c_cum),decay_out,qk_out2)
torch_npu.npu.synchronize()
print("\n=== decay and qk (in one kernel) ===")
compare("decay", decay_g, decay_out.cpu())
compare("qk", qk_g, qk_out2.cpu())

# Test 3: a_local = (qk * decay) * m_le, then dv0 = (a_local^T @ doc) * scale
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_alocal_dv0(
    qc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    kc_in: pypto.Tensor([BT,KD],pypto.DT_FP32),
    doc_in: pypto.Tensor([BT,128],pypto.DT_FP32),
    gc_in: pypto.Tensor([BT],pypto.DT_FP32),
    c_cum_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    m_le_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    scale_in: pypto.Tensor([1],pypto.DT_FP32),
    aloc_out: pypto.Tensor([BT,BT],pypto.DT_FP32),
    dv0_out: pypto.Tensor([BT,128],pypto.DT_FP32),
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
    aloc_out[:] = a_local
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    dv0 = pypto.mul(pypto.matmul(a_local, doc_in, pypto.DT_FP32, a_trans=True), scale_in)
    dv0_out[:] = dv0

aloc_out=torch.zeros(BT,BT,device=npu,dtype=torch.float32)
dv0_out=torch.zeros(BT,128,device=npu,dtype=torch.float32)
test_alocal_dv0(to_npu(qc),to_npu(kc),to_npu(doc),to_npu(gc_raw),to_npu(c_cum),
                to_npu(m_le),to_npu(torch.tensor([scale])),aloc_out,dv0_out)
torch_npu.npu.synchronize()
print("\n=== a_local and dv0 ===")
compare("a_local", a_local_g, aloc_out.cpu())
dv0_g_check = (a_local_g.t() @ doc) * scale
compare("dv0", dv0_g_check, dv0_out.cpu())

# Extra: check if a_local^T @ doc uses a_trans correctly
# Try WITHOUT a_trans (should be wrong, but let's see the magnitude)
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def test_no_trans(
    aloc_in: pypto.Tensor([BT,BT],pypto.DT_FP32),
    doc_in: pypto.Tensor([BT,128],pypto.DT_FP32),
    scale_in: pypto.Tensor([1],pypto.DT_FP32),
    dv0_no_trans: pypto.Tensor([BT,128],pypto.DT_FP32),
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_cube_tile_shapes([128,128],[128,128],[128,128])
    dv0 = pypto.mul(pypto.matmul(aloc_in, doc_in, pypto.DT_FP32), scale_in)
    dv0_no_trans[:] = dv0

dv0_no_trans=torch.zeros(BT,128,device=npu,dtype=torch.float32)
test_no_trans(aloc_out, to_npu(doc), to_npu(torch.tensor([scale])), dv0_no_trans)
torch_npu.npu.synchronize()
print("\n=== dv0 WITHOUT a_trans (a_local @ doc) vs golden ===")
dv0_no_trans_g = (a_local_g @ doc) * scale
compare("dv0_no_trans", dv0_no_trans_g, dv0_no_trans.cpu())
compare("dv0_impl_vs_notrans", dv0_out.cpu(), dv0_no_trans.cpu())
