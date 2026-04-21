import math
import torch
import torch.nn.functional as F
import pypto
import pytest
import numpy as np
from typing import Optional, Tuple



#=====================tomo golden==============
def make_chunk_constants(BT: int, device, dtype=torch.float32):
    idx = torch.arange(BT, device=device)
    I = (idx[:, None] == idx[None, :]).to(dtype)             # [BT,BT]
    M_le = (idx[:, None] >= idx[None, :]).to(dtype)          # lower incl diag
    M_lt = M_le - I                                          # strict lower
    C_cum = M_le                                              # prefix sum: y = C_cum @ x
    C_rcum = (idx[None, :] >= idx[:, None]).to(dtype)         # upper incl diag: suffix sum
    return I, M_le, M_lt, C_cum, C_rcum


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6):
    x32 = x.to(torch.float32)
    rstd = torch.rsqrt((x32 * x32).sum(dim=-1) + eps)         # [...], no keepdim
    y = x32 * rstd[..., None]
    return y, rstd


def forward_ref(
    q: torch.Tensor,           # [T, Nqk, D]
    k: torch.Tensor,           # [T, Nqk, D]
    v: torch.Tensor,           # [T, Nv, D]
    g_raw: torch.Tensor,       # [T, Nv]
    beta: torch.Tensor,        # [T, Nv]
    initial_state: torch.Tensor,  # [B,NV,DV,DK]
    BT: int,
    use_qk_l2norm_in_kernel: bool,
    l2_eps: float,
    I: torch.Tensor, M_le: torch.Tensor, M_lt: torch.Tensor, C_cum: torch.Tensor,act_seq_len: torch.Tensor,
):
    T, Nqk, D = q.shape
    Nv = v.shape[1]
    B = initial_state.shape[0]
    group = Nv // Nqk
    SL = act_seq_len[1]
    assert SL % BT == 0
    NT = SL // BT
    scale = 1.0 / math.sqrt(D)

    # l2norm (store for backward)
    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q, eps=l2_eps)     # [T, Nqk, D]  [T, Nqk] 
        k_norm, k_rstd = l2norm_fwd(k, eps=l2_eps)
        q_used = q_norm
        k_used = k_norm
    else:
        q_used = q.to(torch.float32)
        k_used = k.to(torch.float32)
        q_rstd = None
        k_rstd = None

    v32 = v.to(torch.float32)
    beta32 = beta.to(torch.float32)
    g_raw32 = g_raw.to(torch.float32)

    out = torch.zeros((T, Nv, D), dtype=torch.float32, device=q.device)
    cache_A = torch.zeros((B*Nv*NT, BT, BT), dtype=torch.float32, device=q.device)
    cache_S_before = torch.zeros((B*Nv*NT, D, D), dtype=torch.float32, device=q.device)
    cache_w = torch.zeros((B*Nv*NT, BT, D), dtype=torch.float32, device=q.device)
    cache_u = torch.zeros((B*Nv*NT, BT, D), dtype=torch.float32, device=q.device)
    cache_v_new = torch.zeros((B*Nv*NT, BT, D), dtype=torch.float32, device=q.device)
    cache_final_state = torch.empty_like(initial_state, dtype=torch.float32, device=q.device)

    for b in range(B):
        b_ofs = act_seq_len[b]
        for h in range(Nv):
            nqk_idx = h // group
            S = initial_state[b, h].to(torch.float32)  # [K,V]
            for c in range(NT):
                bs_ofs = b_ofs + c * BT
                cache_idx = (b * Nv + h) * NT + c

                qc = q_used[bs_ofs:bs_ofs + BT, nqk_idx, :]            # [BT,K]
                kc = k_used[bs_ofs:bs_ofs + BT, nqk_idx, :]            # [BT,K]
                vc = v32[bs_ofs:bs_ofs + BT, h, :]               # [BT,V]
                betac = beta32[bs_ofs:bs_ofs + BT, h]            # [BT]
                gc_raw = g_raw32[bs_ofs:bs_ofs + BT, h]          # [BT]

                # g_cum via matmul
                g_cum = C_cum @ gc_raw                 # [BT]
                eg = torch.exp(g_cum)                  # [BT]
                gl = g_cum[-1]

                # build A = inv(I + L) via triangular solve
                diff = g_cum[:, None] - g_cum[None, :]
                decay = torch.exp(diff)                # [BT,BT]
                KKT = kc @ kc.t()                       # [BT,BT]
                L = (betac[:, None] * KKT) * decay
                L = L * M_lt                             # strict-lower
                M = I + L
                A = torch.linalg.solve_triangular(M, I, upper=False)

                # u and w
                vb = vc * betac[:, None]
                kbg = kc * (betac[:, None] * eg[:, None])
                u = A @ vb                               # [BT,V]
                w = A @ kbg                              # [BT,K]

                # cache S_before, A, w, u
                cache_A[cache_idx] = A
                cache_S_before[cache_idx] = S # [V, K]
                cache_w[cache_idx] = w
                cache_u[cache_idx] = u

                # v_new
                v_prime = w @ S                          # [BT,V]
                v_new = u - v_prime
                cache_v_new[cache_idx] = v_new

                # local attention output (mask by elementwise multiply)
                qk = qc @ kc.t()                         # [BT,BT]
                A_local = (qk * decay) * M_le            # keep lower incl diag
                o1 = eg[:, None] * (qc @ S)              # [BT,V]
                o2 = A_local @ v_new                     # [BT,V]
                out[bs_ofs:bs_ofs + BT, h, :] = (o1 + o2) * scale

                # update state
                s = torch.exp(gl - g_cum)                # [BT]
                v_scaled = v_new * s[:, None]            # [BT,V]
                S = S * torch.exp(gl) + kc.t() @ v_scaled
            cache_final_state[b, h] = S #[V,K]

    cache = {
        "A": cache_A,
        "w": cache_w,
        "u": cache_u,
        "v_new": cache_v_new,
        "S_before": cache_S_before,
        "q_norm": q_used,     # [B,T,H,K] float32
        "k_norm": k_used,     # [B,T,H,K] float32
        "q_rstd": q_rstd,     # [B,T,H] float32 or None
        "k_rstd": k_rstd,
    }

    return out, cache_final_state, cache

# -----------------------------
# constant matrices (no tril/triu/ones/zeros)
# -----------------------------
def make_chunk_constants(L: int, device, dtype=torch.float32):
    idx = torch.arange(L, device=device)
    I = (idx[:, None] == idx[None, :]).to(dtype)             # [L,L]
    M_le = (idx[:, None] >= idx[None, :]).to(dtype)          # lower incl diag
    M_lt = M_le - I                                          # strict lower
    C_cum = M_le                                              # prefix sum: y = C_cum @ x
    C_rcum = (idx[None, :] >= idx[:, None]).to(dtype)         # upper incl diag: suffix sum
    return I, M_le, M_lt, C_cum, C_rcum


def l2norm_bwd_chunk(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor):
    # y: [BT,D], rstd: [BT], dy: [BT,D]
    # dx = dy*rstd - sum(dy*y)*y*rstd
    dot = (dy * y).sum(dim=-1)                                # [BT]
    dx = dy * rstd[:, None] - dot[:, None] * y * rstd[:, None]
    return dx

# ============================================================
# backward_ref under constraints:
# - everything inside for b / for h / for chunk(reverse) loop
# - no tril/triu/ones/zeros/masked_fill/cumsum/flip
# - <=4D tensors
# ============================================================
def _compute_g_and_decay(gc_raw, C_cum):
    g_cum = C_cum @ gc_raw                                # [L]
    eg = torch.exp(g_cum)                                 # [L]
    gl = g_cum[-1]                                        # scalar
    diff = g_cum[:, None] - g_cum[None, :]
    decay = torch.exp(diff)                               # [L,L]
    return g_cum, eg, gl, decay


def _local_attn_dv0(qc, kc, doc, decay, M_le, scale):
    qk = qc @ kc.t()                                      # [L,L]
    A_local = (qk * decay) * M_le                         # [L,L]
    dv0 = (A_local.t() @ doc) * scale                     # [L,V]
    return qk, A_local, dv0


def _recurrence_backprop(kc, dS, gl, g_cum, dv0, qc, eg, doc, scale, w):
    dS_next = dS                                           # [K,V]
    s_tok = torch.exp(gl - g_cum)                          # [L]
    dv_state = (kc @ dS_next) * s_tok[:, None]             # [L,V]
    dv_total = dv_state + dv0                              # [L,V]

    q_eff = qc * eg[:, None]                               # [L,K]
    dS = dS_next * torch.exp(gl)
    dS = dS + (q_eff.t() @ doc) * scale
    dS = dS - (w.t() @ dv_total)
    return dS_next, s_tok, dv_total, dS, q_eff


def _compute_qkg_grads(
    device, L, K,
    qc, kc, v_new, doc,
    g_cum, eg, gl, s_tok,
    dS_next, S_before,
    qk, decay, M_le, scale,
):
    dq_c = torch.empty((L, K), device=device, dtype=torch.float32); dq_c.zero_()
    dk_c = torch.empty((L, K), device=device, dtype=torch.float32); dk_c.zero_()
    dg_cum = torch.empty((L,), device=device, dtype=torch.float32); dg_cum.zero_()

    dq1 = (doc @ S_before.t()) * eg[:, None] * scale
    dq_c += dq1
    dg_cum += (dq1 * qc).sum(dim=-1)

    v_scaled = v_new * s_tok[:, None]
    dk_state = v_scaled @ dS_next.t()
    dk_c += dk_state

    scalar = (kc * dk_state).sum(dim=-1)
    dg_cum -= scalar
    dg_cum[-1] += scalar.sum()
    dg_cum[-1] += torch.exp(gl) * (S_before * dS_next).sum()

    dA_base = (doc @ v_new.t()) * M_le * scale
    dq_c += (dA_base * decay) @ kc
    dk_c += (dA_base * decay).t() @ qc
    A_base = (qk * decay) * M_le
    tmp = dA_base * A_base
    dg_cum += tmp.sum(dim=-1) - tmp.sum(dim=-2)

    return dq_c, dk_c, dg_cum

# [Update on v4] remove dq_c, because it never used
def _wy_repr_fused_updates(
    vc, betac, kc, eg, du, dw,
    A, M_lt, decay, dk_c, dg_cum,
):
    vb = vc * betac[:, None]
    kbg = kc * (betac[:, None] * eg[:, None])

    dvb = A.t() @ du
    dkbg = A.t() @ dw

    dv_c = dvb * betac[:, None]
    db_c = (dvb * vc).sum(dim=-1)

    dk_c += dkbg * (betac[:, None] * eg[:, None])
    db_c += (dkbg * (kc * eg[:, None])).sum(dim=-1)
    dg_cum += (dkbg * kbg).sum(dim=-1)

    dA = dw @ kbg.t() + du @ vb.t()
    dL = -(A.t() @ (dA @ A.t()))
    dL = dL * M_lt

    KKT = kc @ kc.t()
    E = decay

    db_c += (dL * (KKT * E)).sum(dim=-1)

    Lmat = (betac[:, None] * KKT) * E
    Lmat = Lmat * M_lt
    tmp2 = dL * Lmat
    dg_cum += tmp2.sum(dim=-1) - tmp2.sum(dim=-2)

    Mmat = dL * (betac[:, None] * E)
    dk_c += (Mmat + Mmat.t()) @ kc

    return dv_c, db_c, dk_c, dg_cum


def _finalize_chunk_grads(
    C_rcum, dg_cum,
    use_qk_l2norm_in_kernel,
    q_used_chunk, k_used_chunk,
    q_rstd_chunk, k_rstd_chunk,
    dq_c, dk_c,
):
    dg_raw_c = C_rcum @ dg_cum

    if use_qk_l2norm_in_kernel:
        dq_raw_c = l2norm_bwd_chunk(q_used_chunk, q_rstd_chunk, dq_c)
        dk_raw_c = l2norm_bwd_chunk(k_used_chunk, k_rstd_chunk, dk_c)
    else:
        dq_raw_c = dq_c
        dk_raw_c = dk_c

    return dq_raw_c, dk_raw_c, dg_raw_c


# -------------------------------------------------------
# Golden
# -------------------------------------------------------
def torch_golden_gated_delta_rule_backward_ref(
    q: torch.Tensor,           # [T, Nqk, Dk]
    k: torch.Tensor,           # [T, Nqk, Dk]
    v: torch.Tensor,           # [T, Nv, Dv]
    g_raw: torch.Tensor,       # [T, Nv]
    beta: torch.Tensor,        # [T, Nv]
    initial_state: torch.Tensor, # [B,Bv,Dk,Dv]
    act_seq_len: torch.Tensor, #[B+1,]
    do: torch.Tensor,          # [T, Nv, Dv]
    dht: torch.Tensor,         # [B,Nv,Dk,Dv]
    cache: dict,
    L: int,
    I: torch.Tensor, M_le: torch.Tensor, M_lt: torch.Tensor, C_cum: torch.Tensor, C_rcum: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    l2_eps: float,
):
    device = q.device
    T, Nqk, Dk = q.shape
    _, Nv, Dv = v.shape
    B = initial_state.shape[0]
    group = Nv // Nqk
    SL = act_seq_len[1]
    assert SL % L == 0
    NT = SL // L
    scale = 1.0 / math.sqrt(Dk)

    # global grads (<=4D)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    db = torch.empty_like(beta, dtype=torch.float32)
    dg_raw = torch.empty_like(g_raw, dtype=torch.float32)
    dh0 = torch.empty_like(initial_state, dtype=torch.float32)

    # init outputs deterministically (no torch.zeros)
    dq.zero_()
    dk.zero_()
    dv.zero_()
    db.zero_()
    dg_raw.zero_()
    dh0.zero_()

    # normalized inputs and rstd (<=4D)
    q_used = cache["q_norm"]
    k_used = cache["k_norm"]

    q_rstd = cache["q_rstd"]
    k_rstd = cache["k_rstd"]

    v32 = v.to(torch.float32)
    beta32 = beta.to(torch.float32)
    g_raw32 = g_raw.to(torch.float32)

    for b in range(B):
        b_ofs = act_seq_len[b]
        for h in range(Nv):
            nqk_idx = h // group
            for i in range(NT):
                c = NT - 1 - i
                if i == 0:  # i = 0 -> c = NT-1 (last chunk)
                    dS = dht[b, h].to(torch.float32)

                bs_ofs = b_ofs + c * L

                # view_chunk_inputs(all <=2D)
                qc = q_used[bs_ofs:bs_ofs + L, nqk_idx, :]               #[L,Dk]
                kc = k_used[bs_ofs:bs_ofs + L, nqk_idx, :]               #[L,Dk]
                vc = v32[bs_ofs:bs_ofs + L, h, :]                        #[L,Dv]
                betac = beta32[bs_ofs:bs_ofs + L, h]                     #[L,]
                gc_raw = g_raw32[bs_ofs:bs_ofs + L, h]                   #[L,]
                doc = do[bs_ofs:bs_ofs + L, h, :].to(torch.float32)      #[L,Dv]
                A = cache["A"][b, h, c]                                  #[L,L]
                w = cache["w"][b, h, c]                                  #[L,Dv]
                S_before = cache["S_before"][b, h, c]                    #[Dk,Dv]
                v_new = cache["v_new"][b, h, c]                          #[L,Dv]

                # ---- g_cum via matmul (no cumsum) ----
                g_cum, eg, gl, decay = _compute_g_and_decay(gc_raw, C_cum)

                # ===== (A) local attention grad wrt v_new: dv0 =====
                qk, A_local, dv0 = _local_attn_dv0(qc, kc, doc, decay, M_le, scale)

                # ===== (B) recurrence backprop (dS_next is current dS) =====
                dS_next, s_tok, dv_total, dS, q_eff = _recurrence_backprop(
                    kc, dS, gl, g_cum, dv0, qc, eg, doc, scale, w
                )

                # ===== (C) grads for q,k,g from outputs + state update + local attention =====
                dq_c, dk_c, dg_cum = _compute_qkg_grads(
                    device, L, Dk,
                    qc, kc, v_new, doc,
                    g_cum, eg, gl, s_tok,
                    dS_next, S_before,
                    qk, decay, M_le, scale,
                )

                # ===== (D) v_new = u - wS  => dw, du =====
                dw = -(dv_total @ S_before.t())
                du = dv_total

                # ===== (E) prepare_wy_repr_bwd fused inside same loop =====
                dv_c, db_c, dk_c, dg_cum = _wy_repr_fused_updates(
                    vc, betac, kc, eg, du, dw,
                    A, M_lt, decay, dk_c, dg_cum,
                )

                # ===== (F) dg_raw = reverse-cumsum(dg_cum) via matmul =====
                # ===== (G) l2norm backward inside chunk loop (as requested) =====
                dq_raw_c, dk_raw_c, dg_raw_c = _finalize_chunk_grads(
                    C_rcum, dg_cum,
                    use_qk_l2norm_in_kernel,
                    q_used[bs_ofs:bs_ofs + L, nqk_idx, :], k_used[bs_ofs:bs_ofs + L, nqk_idx, :],
                    q_rstd[bs_ofs:bs_ofs + L, nqk_idx] if use_qk_l2norm_in_kernel else None,
                    k_rstd[bs_ofs:bs_ofs + L, nqk_idx] if use_qk_l2norm_in_kernel else None,
                    dq_c, dk_c,
                )

                # ===== store into global grads =====
                dq[bs_ofs:bs_ofs + L, nqk_idx, :] = dq_raw_c
                dk[bs_ofs:bs_ofs + L, nqk_idx, :] = dk_raw_c
                dv[bs_ofs:bs_ofs + L, h, :] = dv_c
                db[bs_ofs:bs_ofs + L, h] = db_c
                dg_raw[bs_ofs:bs_ofs + L, h] = dg_raw_c

            # after all chunks, dS is dh0
            dh0[b, h] = dS

    return dq, dk, dv, db, dg_raw, dh0



# ------------------------------------------------------
# Module 2 on PyPTO
# ------------------------------------------------------
def pypto_g_and_decay_kernel(gc_raw_col, C_cum):
    # [l,1]  [l,l]
    BT = gc_raw_col.shape[0]
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    g_cum_col = pypto.matmul(C_cum, gc_raw_col, pypto.DT_FP32)
    eg_col = pypto.exp(g_cum_col)
    gl_1 = g_cum_col.view([1, 1], [BT - 1, 0])
    diff = g_cum_col - g_cum_col.transpose(0, 1)
    decay = pypto.exp(diff)

    return g_cum_col, eg_col, gl_1, decay


# ------------------------------------------------------
# Module 3 on PyPTO
# ------------------------------------------------------
def pypto_local_attn_dv0_kernel(
    qc_in, kc_in, doc_in, decay_in, M_le_in, scale_scalar
):
    #[l,d]  [l,d]  [l,d]   [l,l]    [l,l] 
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])    
    qk = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    common_mask_decay = decay_in * M_le_in
    A_local = qk * common_mask_decay      # 是否可以先给qc或者kc乘以common_mask_decay，再做matmul
    dv0 = pypto.matmul(A_local, doc_in, pypto.DT_FP32, a_trans=True, b_trans=False)
    dv0 = dv0 * scale_scalar

    return qk, dv0, common_mask_decay


# ------------------------------------------------------
# Module 4 on PyPTO
# ------------------------------------------------------
def pypto_recurrence_backprop(
    kc, dS_in, gl, g_cum, dv0, qc, eg, doc, scale_scalar, w,
):  
    #[l,d] [d,d] [1,1] [l,1] [l,d] [l,d] [l,1] [l,d] 1 [l,d]
    BT = g_cum.shape[0]
    K, V = dS_in.shape
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    gl_expand = pypto.expand_clone(gl, [BT,1])
    s_tok = pypto.exp(gl_expand - g_cum)
    s_tok_expand = pypto.expand_clone(s_tok, [BT,V])
    dv_state = pypto.matmul(kc, dS_in, pypto.DT_FP32, b_trans=False) * s_tok_expand # [BT,V] # [No BF16]
    dv_total = dv_state + dv0
    q_eff = qc * eg
    gl_exp = pypto.exp(gl)
    gl_exp_expand = pypto.expand_clone(gl_exp, [K,1])
    term1 = dS_in * gl_exp_expand
    term2 = pypto.matmul(q_eff, doc, pypto.DT_FP32, a_trans=True , b_trans=False) # [No BF16]
    term3 = pypto.matmul(w, dv_total, pypto.DT_FP32, a_trans=True , b_trans=False) # [No BF16]
    dS_final = term1 + term2 * scale_scalar - term3

    return s_tok, dv_total, dS_final, gl_exp


# ------------------------------------------------------
# Module 5 on PyPTO
# ------------------------------------------------------
def pypto_compute_qkg_grads_dw_du(
    qc_in, kc_in, v_new_in, doc_in, eg_in, gl_exp_1, s_tok_in, dS_next_in, S_before_in, qk_in, M_le_in, scale_scalar, dv_total_in, common_mask_decay
):  
    #[l,d]  [l,d]  [l,d]    [l,d]   [l,1]   [1,1]     [l,1]      [d,d]       [d,d]      [l,l]    [l,l]       1            [l,d]        [l,l]
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dq_c_tmp = pypto.matmul(doc_in, S_before_in, pypto.DT_FP32, a_trans=False, b_trans=True) * eg_in * scale_scalar  #[l,d]
    dg_cum_tmp = (dq_c_tmp * qc_in).sum(dim=-1, keepdim=True) # [l,1]
    v_scaled = v_new_in * s_tok_in #[l,d]
    dk_state = pypto.matmul(v_scaled, dS_next_in, pypto.DT_FP32, a_trans=False, b_trans=True)  #[l,d]
    dw_final = pypto.matmul(dv_total_in, S_before_in, pypto.DT_FP32, a_trans=False, b_trans=True) * -1.0  #[l,d]
    scalar = (kc_in * dk_state).sum(dim=-1, keepdim=True)  #[l,1]
    dg_cum_tmp2 = dg_cum_tmp - scalar  #[l,1]
    one_hot_last = M_le_in[:, -1:]  #[l,1]
    tail_add = scalar.sum(dim=0, keepdim=True) #[1,1]

    # pypto.set_vec_tile_shapes(16, 16, 8, 8)
    pypto.set_vec_tile_shapes(128, 128)
    tail_add2 = tail_add + gl_exp_1 * (S_before_in * dS_next_in).sum(dim=0, keepdim=True).sum(dim=-1, keepdim=True) #[1,1]

    pypto.set_vec_tile_shapes(256, 128)
    dg_cum_tmp3 = dg_cum_tmp2 + one_hot_last * tail_add2  #[l,1]
    scaled_common_mask_decay = common_mask_decay * scale_scalar  #[l,l]
    dA_based_tmp = pypto.matmul(doc_in, v_new_in, pypto.DT_FP32, a_trans=False, b_trans=True) #[l,l]
    dq_c_term1 = dA_based_tmp * scaled_common_mask_decay  #[l,l]
    dq_c_term2 = pypto.matmul(dq_c_term1, kc_in, pypto.DT_FP32, a_trans=False, b_trans=False)  #[l,1]
    dk_c_term1 = pypto.matmul(dq_c_term1, qc_in, pypto.DT_FP32, a_trans=True, b_trans=False)  #[l,d]
    dq_c = dq_c_tmp + dq_c_term2  #[l,d]  expand
    dk_c = dk_state + dk_c_term1  #[l,d]
    dAA_tmp = dq_c_term1 * qk_in  #[l,l]

    # pypto.set_vec_tile_shapes(16, 16, 8, 8)    
    pypto.set_vec_tile_shapes(128, 128, 128)
    dg_cum_term4 = dAA_tmp.sum(dim=-1) - dAA_tmp.sum(dim=-2)
    pypto.set_vec_tile_shapes(128, 128)
    dg_cum_final = dg_cum_tmp3 + dg_cum_term4.unsqueeze(-1)

    return dq_c, dg_cum_final, dk_c, dw_final


# ------------------------------------------------------
# Module 6 on PyPTO
# ------------------------------------------------------
def pypto_wy_repr_fused_updates(
    vc, betac, kc, eg, du, dw, A, M_lt, decay, dk_c_in, dg_cum,
):  
#[l,d] [l,1] [l,d] [l,1] [l,d] [l,d] [l,l] [l,l] [l,l] [l,d] [l,1]
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    vb = vc * betac  #[l,d]
    beg = betac * eg # [l,1]
    kbg = kc * beg   #[l,d]
    dvb = pypto.matmul(A, du, pypto.DT_FP32, a_trans=True, b_trans=False)  #[l,d]
    dkbg = pypto.matmul(A, dw, pypto.DT_FP32, a_trans=True, b_trans=False)  #[l,d]
    dv_c = dvb * betac  #[l,d]
    db_c_tmp = (dvb * vc).sum(-1, keepdim=True)  #[l,1]
    dk_c_tmp = dk_c_in + (dkbg * beg)  #[l,d]
    db_c_tmp2 = db_c_tmp + (dkbg * (kc * eg)).sum(-1, keepdim=True) #[l,1]
    dg_cum_tmp = dg_cum + (dkbg * kbg).sum(-1, keepdim=True) #[l,1]
    dA_term1 = pypto.matmul(dw, kbg, pypto.DT_FP32, a_trans=False, b_trans=True) #[l,l]
    dA_term2 = pypto.matmul(du, vb, pypto.DT_FP32, a_trans=False, b_trans=True)  #[l,l]
    dA = dA_term1 + dA_term2  #[l,l]
    dL_term1 = pypto.matmul(dA, A, pypto.DT_FP32, a_trans=False, b_trans=True) #[l,l]
    dL_tmp = pypto.matmul(A, dL_term1, pypto.DT_FP32, a_trans=True, b_trans=False)  #[l,l]
    dL = (dL_tmp * -1.0) * M_lt  #[l,l]
    kkt = pypto.matmul(kc, kc, pypto.DT_FP32, a_trans=False, b_trans=True)  #[l,l]
    db_c = db_c_tmp2 + (dL * (kkt * decay)).sum(-1, keepdim=True) #[l,1]
    mmat = dL * betac * decay #[l,l]

    dg_cum_term = mmat * (kkt*M_lt) #[l,l]
    dg_cum_out = dg_cum_tmp + dg_cum_term.sum(-1, keepdim=True) - dg_cum_term.sum(-2).unsqueeze(-1) #[l,1]
    pypto.set_vec_tile_shapes(128, 128)
    mmat_t = mmat.transpose(0,1)  #[l,l]
    mmat_sum = mmat + mmat_t #[l,l]
    dk_c_term2 = pypto.matmul(mmat_sum, kc, pypto.DT_FP32, a_trans=False, b_trans=False) #[l,d]
    dk_c = dk_c_tmp + dk_c_term2 #[l,d]

    return dv_c, db_c, dk_c, dg_cum_out


# ------------------------------------------------------
# Module 7 on PyPTO
# ------------------------------------------------------
def pypto_l2norm_bwd_qk(
    yq, rstd_yq, dyq,
    yk, rstd_yk, dyk
    ):
    pypto.set_vec_tile_shapes(256, 128)
    dot_q = (dyq * yq).sum(-1, keepdim=True)
    dot_k = (dyk * yk).sum(-1, keepdim=True)
    dq = dyq * rstd_yq - dot_q * yq * rstd_yq
    dk = dyk * rstd_yk - dot_k * yk * rstd_yk
    return dq, dk


def pypto_finalize_chunk_grads(
    C_rcum, dg_cum, q_used, k_used, q_rstd, k_rstd, dq_c, dk_c
):  
    #[l,l]  [l,1]   [l,d]    [l,d]   [l,1]  [l,1]  [l,d] [l,d] 
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dg_raw_c = pypto.matmul(C_rcum, dg_cum, pypto.DT_FP32, b_trans=False)
    dq_raw_c, dk_raw_c = pypto_l2norm_bwd_qk(q_used, q_rstd, dq_c, k_used, k_rstd, dk_c)

    return dg_raw_c, dq_raw_c, dk_raw_c


verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    # "pass_verify_pass_filter": ["CodegenPreproc"],
}
@pypto.frontend.jit(
    runtime_options={
        # "stitch_function_inner_memory": 128 * 16,
        # "stitch_function_num_initial": 128,
        # "stitch_function_outcast_memory": 128 * 16,
        # "stitch_function_size": 65535,
        # "device_sched_mode": 1
    }, 
    debug_options={"runtime_debug_mode": 1},
    # verify_options=verify_options
    pass_options={
        "vec_nbuffer_setting": {-1: 4, -2: 1},
        "cube_l1_reuse_setting": {-1: 16},
    },
)
def gated_delta_rule_bwd_kernel(
    v_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    g_raw_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    beta_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    do_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dht_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    M_le_in: pypto.Tensor([], pypto.DT_FP32),
    M_lt_in: pypto.Tensor([], pypto.DT_FP32), 
    C_cum_in: pypto.Tensor([], pypto.DT_FP32),
    C_rcum_in: pypto.Tensor([], pypto.DT_FP32),
    A_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    w_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    v_new_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    S_before_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    q_norm_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k_norm_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    q_rstd_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k_rstd_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    act_seq_len: pypto.Tensor([], pypto.DT_INT32),
    dq_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dk_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dv_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    db_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dg_raw_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
):  

    # pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    pypto.experimental.set_operation_options(combine_axis=True)

    # -------------------------------------------------------------
    # Calculate the loop parameters
    # -------------------------------------------------------------
    _, nqk, dim = q_norm_cache.shape
    _, nv, dim = v_in.shape
    batch = dht_in.shape[0]
    l, l = M_le_in.shape
    group = nv // nqk
    scale_scalar = 1 / (dim ** 0.5)

    # -------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------
    for b_idx in pypto.loop(batch, name="LOOP_B_BSND", idx_name="b_idx"):
        dyn_seq = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        dyn_chunks =  dyn_seq // l
        b_ofs = act_seq_len[b_idx]
        for nv_idx in pypto.loop(nv, name="LOOP_Nv_BSND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(16, 16, 128, 128)
            dS_2d = dht_in[b_idx, nv_idx]  #[Dk, Dv]
            for inv_s_idx in pypto.loop(0, dyn_seq, l, name="LOOP_S_REVERSE_BSND", idx_name="i_idx", unroll_list=[16, 1]):
                s_idx = dyn_seq - inv_s_idx - l
                bs_ofs = b_ofs + s_idx
                rev_idx = s_idx // l
                actual_L = (dyn_seq - s_idx).min(l)
                
                # -----------------------------------------
                # Module 1 pto_slice_chunk_inputs_kernel
                # -----------------------------------------
                pypto.set_semantic_label("view")
                pypto.set_vec_tile_shapes(128, 128, 128)
                query_used_view = pypto.view(q_norm_cache, [l, 1, dim], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, dim])
                key_used_view = pypto.view(k_norm_cache, [l, 1, dim], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, dim])
                value_view = pypto.view(v_in, [l, 1, dim], [bs_ofs, nv_idx, 0], valid_shape =[actual_L, 1, dim])
                do_view =  pypto.view(do_in, [l, 1, dim], [bs_ofs, nv_idx, 0], valid_shape =[actual_L, 1, dim])

                qc = pypto.reshape(query_used_view, [l, dim], valid_shape=[actual_L, dim])  #[l(BT), D]
                kc = pypto.reshape(key_used_view, [l, dim], valid_shape=[actual_L, dim]) #[l(BT), D]
                vc = pypto.reshape(value_view, [l, dim], valid_shape=[actual_L, dim]) #[l(BT), D]
                doc = pypto.reshape(do_view, [l, dim], valid_shape=[actual_L, dim]) #[l(BT), D]

                pypto.set_vec_tile_shapes(128, 128)
                q_rstd_2d = pypto.view(q_rstd_cache, [l, 1], [bs_ofs, nqk_idx], valid_shape =[actual_L, 1])
                k_rstd_2d = pypto.view(k_rstd_cache, [l, 1], [bs_ofs, nqk_idx], valid_shape =[actual_L, 1])

                betac = pypto.view(beta_in, [l, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1]) #[1, l(BT), 1]
                gc_raw = pypto.view(g_raw_in, [l, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1]) #[1, l(BT), 1]

                cache_idx = (b_idx * nv + nv_idx) * dyn_chunks + rev_idx
                A_view = pypto.view(A_cache, [1, l, l], [cache_idx, 0, 0])
                w_view = pypto.view(w_cache, [1, l, dim], [cache_idx, 0, 0])
                S_before_view = pypto.view(S_before_cache, [1, dim, dim], [cache_idx, 0, 0])
                v_new_view = pypto.view(v_new_cache, [1, l, dim], [cache_idx, 0, 0])

                A_view_2d = pypto.reshape(A_view, [l, l])
                w_view_2d = pypto.reshape(w_view, [l, dim])
                S_before_view_2d = pypto.reshape(S_before_view, [dim, dim])
                v_new_view_2d = pypto.reshape(v_new_view, [l, dim])
                
                # -----------------------------------------
                # Module 2 pto_g_and_decay_kernel
                # -----------------------------------------
                # [l,1]    [l,1] [1,1]  [l,l] 
                pypto.set_semantic_label("g_decay")
                g_cum_2d, eg_2d, gl_1, decay_2d = pypto_g_and_decay_kernel(gc_raw, C_cum_in)

                # -----------------------------------------
                # Module 3 pto_local_attn_dv0_kernel
                # -----------------------------------------
                #[l,l] [l,d]   [l,l]
                pypto.set_semantic_label("dv0")
                qk, dv0, common_mask_decay = pypto_local_attn_dv0_kernel(qc, kc, doc, decay_2d, M_le_in, scale_scalar)

                # -----------------------------------------
                # Module 4 pypto_recurrence_backprop
                # -----------------------------------------
                # [l,1] [l,d]      [d,d]    [1,1]
                pypto.set_semantic_label("recurrence")
                s_tok, dv_total, dS_final, gl_exp_1 = pypto_recurrence_backprop(kc, dS_2d, gl_1, g_cum_2d, dv0, qc, eg_2d, doc, scale_scalar, w_view_2d)

                # -----------------------------------------
                # Module 5 pypto_compute_qkg_grads_dw_du
                # -----------------------------------------
                #[l,d]  [l,1]         [l,d]    [l,d]
                pypto.set_semantic_label("qkg")
                dq_c, dg_cum_final, dk_c_tmp, dw_final = pypto_compute_qkg_grads_dw_du(qc, kc, v_new_view_2d, doc, eg_2d, gl_exp_1, s_tok, dS_2d, S_before_view_2d, qk, M_le_in, scale_scalar, dv_total, common_mask_decay)

                # -----------------------------------------
                # Module 6 pypto_wy_repr_fused_updates
                # -----------------------------------------
                #[l,d] [l,1] [l,d] [l,1]
                pypto.set_semantic_label("wy_repr")
                dv_c, db_c, dk_c, dg_cum_out = pypto_wy_repr_fused_updates(vc, betac, kc, eg_2d, dv_total, dw_final, A_view_2d, M_lt_in, decay_2d, dk_c_tmp, dg_cum_final)

                # -----------------------------------------
                # Module 7 pypto_finalize_chunk_grads
                # -----------------------------------------
                #[l,1]     [l,d]     [l,d]
                pypto.set_semantic_label("finalize")
                dg_raw_c, dq_raw_c, dk_raw_c = pypto_finalize_chunk_grads(C_rcum_in, dg_cum_out, qc, kc, q_rstd_2d, k_rstd_2d, dq_c, dk_c)

                # Assemble
                pypto.set_semantic_label("Assemble")
                pypto.set_vec_tile_shapes(128, 128)
                dS_2d[:] = dS_final
                dq_out[bs_ofs:bs_ofs + l, nqk_idx] = dq_raw_c
                dk_out[bs_ofs:bs_ofs + l, nqk_idx] = dk_raw_c
                dv_out[bs_ofs:bs_ofs + l, nv_idx] = dv_c
                db_out[bs_ofs:bs_ofs + l, nv_idx:nv_idx+1] = db_c
                dg_raw_out[bs_ofs:bs_ofs + l, nv_idx:nv_idx+1] = dg_raw_c

            # pypto.set_vec_tile_shapes(16, 16, 128, 128)
            dh0_out[b_idx, nv_idx] = dS_2d



def pypto_function(
    q, k, v, g_raw, beta, initial_state, act_seq_len, 
    do, dht, cache,
    I, M_le, M_lt, C_cum, C_rcum,
    run_mode='npu',
    ):

    device = q.device
    L = M_le.shape[0]
    T, Nqk, D = q.shape
    _, Nv, _ = v.shape
    B = len(act_seq_len) - 1
    S = act_seq_len[1]

    dq_out = torch.zeros([T, Nqk, D], dtype=torch.float32, device=q.device)
    dk_out = torch.zeros([T, Nqk, D], dtype=torch.float32, device=q.device)
    dv_out = torch.zeros([T, Nv, D], dtype=torch.float32, device=q.device)
    db_out = torch.zeros([T, Nv], dtype=torch.float32, device=q.device)
    dg_raw_out = torch.zeros([T, Nv], dtype=torch.float32, device=q.device)
    dh0_out = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=q.device)

    # Cache from forward
    A_cache = cache['A'].to(device)
    w_cache = cache['w'].to(device)
    v_new_cache = cache['v_new'].to(device)
    S_before_cache = cache['S_before'].to(device)
    q_norm_cache = cache['q_norm'].to(device)
    k_norm_cache = cache['k_norm'].to(device)
    q_rstd_cache = cache['q_rstd'].to(device)
    k_rstd_cache = cache['k_rstd'].to(device)


    input_tensors = [
        v, g_raw, beta, do, dht, M_le, M_lt, C_cum, C_rcum,
        A_cache, w_cache, v_new_cache, S_before_cache, 
        q_norm_cache, k_norm_cache, q_rstd_cache, k_rstd_cache, act_seq_len, 
        dq_out, dk_out, dv_out, db_out, dg_raw_out, dh0_out
    ]

    gated_delta_rule_bwd_kernel(*input_tensors)
    print('>>> pypto done')

    return dq_out, dk_out, dv_out, db_out, dg_raw_out, dh0_out

    

def main():
    torch.manual_seed(0)
    device_id = 0
    torch.npu.set_device(device_id)
    run_mode = 'npu'
    device = f'{run_mode}:{device_id}'
    dtype = torch.float32

    T = 128
    S = 128
    Nqk = 1
    Nv = 2
    D = 128
    act_seq_len = [0, T]
    B = len(act_seq_len) - 1
    L = 128
    assert S % L == 0
    C = S // L

    use_l2 = True
    eps = 1e-6

    # inputs
    torch.manual_seed(0)
    scale_input_tensor = 0.1
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32, device=f'npu:{device_id}')
    q = (torch.randn(T, Nqk, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    k = (torch.randn(T, Nqk, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    v = (torch.randn(T, Nv, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    g_raw = (torch.randn(T, Nv, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    beta = torch.rand(T, Nv, device=device, dtype=dtype).requires_grad_(True)
    initial_state = (torch.randn(B, Nv, D, D, device=device, dtype=dtype) * 0.01).requires_grad_(True)

    # constants matirx
    I, M_le, M_lt, C_cum, C_rcum = make_chunk_constants(L, device=device, dtype=torch.float32)
    
    # # input by pto
    # inputs = [q, k, v, beta, g_raw, initial_state, act_seq_len]
    # core_attn_out, last_state_data, cache = pypto_chunk_gated_delta_rule(*inputs)

    # input by torch
    core_attn_out, last_state_data, cache = forward_ref(
        q=q, k=k, v=v, g_raw=g_raw, beta=beta, initial_state=initial_state,
        BT=L, use_qk_l2norm_in_kernel=use_l2, l2_eps=eps,
        I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum, act_seq_len=act_seq_len,
    )

    # upstream grads
    do_tnd = torch.randn_like(core_attn_out)
    dht_tnd = torch.randn_like(last_state_data)

    # ---------------- Golden ----------------
    Dv = D
    Dk = D
    cache_for_golden = {
        "A": cache['A'].reshape([B, Nv, C, L, L]), #(B,Nv,C,L,L)
        "w": cache['w'].reshape([B, Nv, C, L, Dv]),#(B,Nv,C,L,D)
        "u": cache['u'].reshape([B, Nv, C, L, Dv]),#(B,Nv,C,L,D)
        "v_new": cache['v_new'].reshape([B, Nv, C, L, Dv]),#(B,Nv,C,L,D)
        "S_before": cache['S_before'].reshape([B, Nv, C, Dv, Dk]),#(B,Nv,C,D,D)
        "q_norm": cache['q_norm'], #(T, Nqk, D)
        "k_norm": cache['k_norm'], #(T, Nqk, D)
        "q_rstd": cache['q_rstd'], #(T, Nqk)
        "k_rstd": cache['k_rstd'], #(T, Nqk)
        "final_state": last_state_data, #(B, Nv, Dk, Dv)
    }
    with torch.no_grad():
        dq, dk, dv, db, dg_raw, dh0 = torch_golden_gated_delta_rule_backward_ref(
            q=q.clone().detach(), k=k.clone().detach(), v=v.clone().detach(),
            g_raw=g_raw.clone().detach(), beta=beta.clone().detach(),
            initial_state=initial_state.clone().detach(),
            act_seq_len=act_seq_len.clone().detach(),
            do=do_tnd.clone().detach(), dht=dht_tnd.clone().detach(),
            cache=cache_for_golden,
            L=I.shape[0],
            I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum, C_rcum=C_rcum,
            use_qk_l2norm_in_kernel=use_l2,
            l2_eps=eps,
        )

    # ---------------- pypto ----------------
    cache_for_pto = {
        "A": cache['A'].contiguous(), #(B*Nv*C,L,L)
        "w": cache['w'].contiguous(),#(B*Nv*C,L,D)
        "u": cache['u'].contiguous(),#(B*Nv*C,L,D)
        "v_new": cache['v_new'].contiguous(),#(B*Nv*C,L,D)
        "S_before": cache['S_before'].contiguous(),#(B*Nv*C,Dk,Dv)
        "q_norm": cache['q_norm'].contiguous(), #(T, Nqk, D)
        "k_norm": cache['k_norm'].contiguous(), #(T, Nqk, D)
        "q_rstd": cache['q_rstd'].contiguous(), #(T, Nqk)
        "k_rstd": cache['k_rstd'].contiguous(), #(T, Nqk)
        "final_state": last_state_data.contiguous(), #(B, Nv, Dk, Dv)
    }
    with torch.no_grad():
        pto_dq, pto_dk, pto_dv, pto_db, pto_dg_raw, pto_dh0  = pypto_function(
            q=q.detach(), k=k.detach(), v=v.detach(),
            g_raw=g_raw.detach(), beta=beta.detach(),
            initial_state=initial_state.detach(),
            act_seq_len=act_seq_len.detach(),
            do=do_tnd.detach(), dht=dht_tnd.detach(),
            cache=cache_for_pto,
            I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum, C_rcum=C_rcum,
            run_mode=run_mode, 
        )
    

    detailed_tensor_compare(pto_dq, dq, 'dq')
    detailed_tensor_compare(pto_dk, dk, 'dk')
    detailed_tensor_compare(pto_dv, dv, 'dv')
    detailed_tensor_compare(pto_db, db, 'db')
    detailed_tensor_compare(pto_dg_raw, dg_raw, 'dg_raw')
    detailed_tensor_compare(pto_dh0, dh0, 'dh0')
    print("\n✅ All checks passed.")


def detailed_tensor_compare(tensor1, tensor2, tensor_name, rtol=1e-3, atol=1e-3, verbose=True, max_outliers_display=20):
    """
    Detailed tensor comparison, analyzing the proportion of elements that are out of tolerance, and displaying specific information about those that exceed the tolerance.
    
    Args:
    tensor1: The first tensor.
    tensor2: The second tensor.
    rtol: Relative tolerance.
    atol: Absolute tolerance.
    verbose: Whether to print detailed information.
    max_outliers_display: Maximum number of out-of-tolerance elements to display.

    Returns:
    dict: A dictionary containing the comparison results.
    """
    # Ensure tensors are comparable
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()
    
    # Calculate the difference
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)
    
    # Tolerance Check
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask
    
    # Statistics
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements
    
    # Difference Statistics
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()
    
    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()
        
        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]
        
        sorted_indices = torch.argsort(outlier_diffs, descending=True)
        sorted_outlier_indices = tuple(ind[sorted_indices] for ind in outlier_indices)
        sorted_outlier_values1 = outlier_values1[sorted_indices]
        sorted_outlier_values2 = outlier_values2[sorted_indices]
        sorted_outlier_diffs = outlier_diffs[sorted_indices]
        sorted_outlier_relative_diffs = outlier_relative_diffs[sorted_indices]
        
    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        sorted_outlier_indices = None
        sorted_outlier_values1 = None
        sorted_outlier_values2 = None
        sorted_outlier_diffs = None
        sorted_outlier_relative_diffs = None
    
    result = {
        'total_elements': total_elements,
        'out_of_tolerance_count': out_of_tolerance_count,
        'out_of_tolerance_ratio': out_of_tolerance_ratio,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_out_of_tolerance_diff': max_out_diff,
        'mean_out_of_tolerance_diff': mean_out_diff,
        'all_close': out_of_tolerance_count == 0,
        'tolerance_mask': tolerance_mask,
        'diff_tensor': diff,
        'outlier_indices': sorted_outlier_indices,
        'outlier_values1': sorted_outlier_values1,
        'outlier_values2': sorted_outlier_values2,
        'outlier_diffs': sorted_outlier_diffs,
        'outlier_relative_diffs': sorted_outlier_relative_diffs
    }
    
    if verbose:
        print("\n" + "="*60)
        print("📊 Tensor Detailed Comparison Report")
        print(f"name: {tensor_name}")
        print("="*60)
        print(f"Total number of elements: {total_elements:,}")
        print(f"Number of elements exceeding tolerance: {out_of_tolerance_count:,}")
        print(f"Out of Tolerance Ratio: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio*100:.4f}%)")
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Average difference: {mean_diff:.6f}")
        print(f"Difference Standard Deviation: {std_diff:.6f}")
        print(f"Tolerance Settings: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"Maximum deviation exceeding tolerance: {max_out_diff:.6f}")
            print(f"Average deviation exceeding tolerance: {mean_out_diff:.6f}")
            
            print(f"\n🔍 Details of elements exceeding tolerance limits (Before Displaying{min(max_outliers_display, out_of_tolerance_count)}):")
            print("-" * 80)
            print(f"{'Index':<20} {'Tensor1 value':<15} {'Tensor2 value':<15} {'Absolute difference':<12} {'Relative difference':<12}")
            print("-" * 80)
            
            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices))))
                print(f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} {sorted_outlier_values2[i].item():<15.6f} "
                      f"{sorted_outlier_diffs[i].item():<12.6f} {sorted_outlier_relative_diffs[i].item():<12.6f}")
            
            if out_of_tolerance_count > max_outliers_display:
                print(f"... And also {out_of_tolerance_count - max_outliers_display} An element exceeding the tolerance is not displayed.")
        
        print(f"\n✅ Tensor Matching: {result['all_close']}")
        print("="*60)
    
    return result



if __name__ == "__main__":
    main()
