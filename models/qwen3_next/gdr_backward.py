import math
import torch
import torch.nn.functional as F
import pypto
import pytest
import numpy as np
from typing import Optional, Tuple

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


# -----------------------------
# l2norm fwd/bwd (no special ops)
# y = x * rstd, rstd = rsqrt(sum(x^2)+eps)
# dx = dy*rstd - sum(dy*y)*y*rstd
# -----------------------------
def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6):
    x32 = x.to(torch.float32)
    rstd = torch.rsqrt((x32 * x32).sum(dim=-1) + eps)         # [...], no keepdim
    y = x32 * rstd[..., None]
    return y, rstd

def l2norm_bwd_chunk(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor):
    # y: [BT,D], rstd: [BT], dy: [BT,D]
    # dx = dy*rstd - sum(dy*y)*y*rstd
    dot = (dy * y).sum(dim=-1)                                # [BT]
    dx = dy * rstd[:, None] - dot[:, None] * y * rstd[:, None]
    return dx


# ============================================================
# forward_ref under constraints (no cumsum/masked_fill/tril/triu etc)
# - input B,T,H
# - caches are python tensor (v5)
# ============================================================
def forward_ref(
    q: torch.Tensor,           # [B,T,H,K]
    k: torch.Tensor,           # [B,T,H,K]
    v: torch.Tensor,           # [B,T,H,V]
    g_raw: torch.Tensor,       # [B,T,H]
    beta: torch.Tensor,        # [B,T,H]
    initial_state: torch.Tensor,  # [B,H,K,V]
    BT: int,
    use_qk_l2norm_in_kernel: bool,
    l2_eps: float,
    I: torch.Tensor, M_le: torch.Tensor, M_lt: torch.Tensor, C_cum: torch.Tensor,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T % BT == 0
    NT = T // BT
    scale = 1.0 / math.sqrt(K)

    # l2norm (store for backward)
    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q, eps=l2_eps)     # q_norm float32
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

    out = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    cache_A = torch.zeros((B * H * NT * BT, BT), dtype=torch.float32, device=q.device)
    cache_S_before = torch.zeros((B * H * NT * K, V), dtype=torch.float32, device=q.device)
    cache_w = torch.zeros((B * H * NT * BT, V), dtype=torch.float32, device=q.device)
    cache_u = torch.zeros((B * H * NT * BT, V), dtype=torch.float32, device=q.device)
    cache_v_new = torch.zeros((B * H * NT * BT, V), dtype=torch.float32, device=q.device)

    for b in range(B):
        for h in range(H):
            S = initial_state[b, h].to(torch.float32)  # [K,V]
            for c in range(NT):
                t0 = c * BT
                t1 = t0 + BT

                cache_idx = (b * H + h) * NT + c

                qc = q_used[b, t0:t1, h, :]            # [BT,K]
                kc = k_used[b, t0:t1, h, :]            # [BT,K]
                vc = v32[b, t0:t1, h, :]               # [BT,V]
                betac = beta32[b, t0:t1, h]            # [BT]
                gc_raw = g_raw32[b, t0:t1, h]          # [BT]

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
                # cache["S_before"][b][h].append(S)
                # cache["A"][b][h].append(A)
                # cache["w"][b][h].append(w)
                # cache["u"][b][h].append(u)
                cache_bt_start = cache_idx * BT
                cache_k_start = cache_idx * K
                cache_A[cache_bt_start:cache_bt_start + BT] = A
                cache_S_before[cache_k_start:cache_k_start + K] = S
                cache_w[cache_bt_start:cache_bt_start + BT] = w
                cache_u[cache_bt_start:cache_bt_start + BT] = u

                # v_new
                v_prime = w @ S                          # [BT,V]
                v_new = u - v_prime
                # cache["v_new"][b][h].append(v_new)
                cache_v_new[cache_bt_start:cache_bt_start + BT] = v_new

                # local attention output (mask by elementwise multiply)
                qk = qc @ kc.t()                         # [BT,BT]
                A_local = (qk * decay) * M_le            # keep lower incl diag
                o1 = eg[:, None] * (qc @ S)              # [BT,V]
                o2 = A_local @ v_new                     # [BT,V]
                out[b, t0:t1, h, :] = (o1 + o2) * scale

                # update state
                s = torch.exp(gl - g_cum)                # [BT]
                v_scaled = v_new * s[:, None]            # [BT,V]
                S = S * torch.exp(gl) + kc.t() @ v_scaled

    final_state = torch.empty_like(initial_state, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            # last chunk updated state is the final; we didn't store it, so recompute cheaply:
            # easiest: run again but only for final_state; keep it simple and deterministic
            # (still within constraints for forward_ref)
            pass
        
    # Instead of recompute, capture final S in a separate pass during loop (no extra ops)
    # We'll store final S per (b,h) in cache during loop:
    # cache["final_state"] = torch.empty_like(initial_state, dtype=torch.float32)
    cache_final_state = torch.empty_like(initial_state, dtype=torch.float32, device=q.device)
    for b in range(B):
        for h in range(H):
            # final S is S_after_last = update of last chunk;
            # we can rebuild by taking last S_before and applying one forward update again,
            # but better: just re-run minimal recurrence with cached per chunk (v_new and g_cum recomputed).
            # For test sizes it's ok.
            S = initial_state[b, h].to(torch.float32)
            for c in range(NT):
                cache_idx = (b * H + h) * NT + c
                t0 = c * BT
                t1 = t0 + BT
                kc = k_used[b, t0:t1, h, :]
                gc_raw = g_raw32[b, t0:t1, h]
                g_cum = C_cum @ gc_raw
                gl = g_cum[-1]
                # v_new = cache["v_new"][b][h][c]
                # v_new = cache_v_new[b,h,c]
                cache_bt_start = cache_idx * BT
                v_new = cache_v_new[cache_bt_start:cache_bt_start + BT]
                s = torch.exp(gl - g_cum)
                v_scaled = v_new * s[:, None]
                S = S * torch.exp(gl) + kc.t() @ v_scaled
            # cache["final_state"][b, h] = S
            cache_final_state[b, h] = S

    cache = {
        "A": cache_A,
        "w": cache_w,
        "u": cache_u,
        "v_new": cache_v_new,
        "S_before": cache_S_before,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "q_norm": q_used.transpose(1, 2).contiguous().reshape(B * H * T, K),
        "k_norm": k_used.transpose(1, 2).contiguous().reshape(B * H * T, K),
        "q_rstd": q_rstd,     # [B,T,H] float32 or None
        "k_rstd": k_rstd,
        "scale": scale,
        "BT": BT,
        "final_state": cache_final_state
    }

    return out, cache_final_state, cache



# -----------------------------
# constant matrices (no tril/triu/ones/zeros)
# -----------------------------
def make_chunk_constants(BT: int, D: int, device, dtype=torch.float32):
    idx = torch.arange(BT, device=device)
    I = (idx[:, None] == idx[None, :]).to(dtype)             # [BT,BT]
    M_le = (idx[:, None] >= idx[None, :]).to(dtype)          # lower incl diag
    M_lt = M_le - I                                          # strict lower
    C_cum = M_le                                              # prefix sum: y = C_cum @ x
    C_rcum = (idx[None, :] >= idx[:, None]).to(dtype)         # upper incl diag: suffix sum
    ones_1l = torch.ones(1,BT).to(dtype).to(device)                     # (1,L) ones
    ones_1d = torch.ones(1, D).to(dtype).to(device)   
    return I, M_le, M_lt, C_cum, C_rcum, ones_1l, ones_1d


# ============================================================
# backward_ref under constraints:
# - everything inside for b / for h / for chunk(reverse) loop
# - no tril/triu/ones/zeros/masked_fill/cumsum/flip
# - <=4D tensors
# ============================================================
def _slice_chunk_inputs(
    q_used, k_used, v32, beta32, g_raw32, do,
    cache, b, h, c, t0, t1,
):
    qc = q_used[b, t0:t1, h, :]                          # [BT,K]
    kc = k_used[b, t0:t1, h, :]                          # [BT,K]
    vc = v32[b, t0:t1, h, :]                             # [BT,V]
    betac = beta32[b, t0:t1, h]                          # [BT]
    gc_raw = g_raw32[b, t0:t1, h]                        # [BT]
    doc = do[b, t0:t1, h, :].to(torch.float32)            # [BT,V]

    # A = cache["A"][b][h][c]                               # [BT,BT]
    # w = cache["w"][b][h][c]                               # [BT,K]
    # S_before = cache["S_before"][b][h][c]                 # [K,V]
    # v_new = cache["v_new"][b][h][c]                       # [BT,V]
    A = cache["A"][b, h, c]
    w = cache["w"][b, h, c]
    S_before = cache["S_before"][b, h, c]
    v_new = cache["v_new"][b, h, c]  
    return qc, kc, vc, betac, gc_raw, doc, A, w, S_before, v_new


def _compute_g_and_decay(gc_raw, C_cum):
    g_cum = C_cum @ gc_raw                                # [BT]
    eg = torch.exp(g_cum)                                 # [BT]
    gl = g_cum[-1]                                        # scalar
    diff = g_cum[:, None] - g_cum[None, :]
    decay = torch.exp(diff)                               # [BT,BT]
    return g_cum, eg, gl, decay


def _local_attn_dv0(qc, kc, doc, decay, M_le, scale):
    qk = qc @ kc.t()                                      # [BT,BT]
    A_local = (qk * decay) * M_le                         # [BT,BT]
    dv0 = (A_local.t() @ doc) * scale                     # [BT,V]
    return qk, A_local, dv0


def _recurrence_backprop(kc, dS, gl, g_cum, dv0, qc, eg, doc, scale, w):
    dS_next = dS                                          # [K,V]
    s_tok = torch.exp(gl - g_cum)                          # [BT]
    dv_state = (kc @ dS_next) * s_tok[:, None]             # [BT,V]
    dv_total = dv_state + dv0                              # [BT,V]

    q_eff = qc * eg[:, None]                               # [BT,K]
    dS = dS_next * torch.exp(gl)
    dS = dS + (q_eff.t() @ doc) * scale
    dS = dS - (w.t() @ dv_total)
    return dS_next, s_tok, dv_total, dS, q_eff


def _compute_qkg_grads(
    device, BT, K,
    qc, kc, v_new, doc,
    g_cum, eg, gl, s_tok,
    dS_next, S_before,
    qk, decay, M_le, scale,
):
    dq_c = torch.empty((BT, K), device=device, dtype=torch.float32); dq_c.zero_()
    dk_c = torch.empty((BT, K), device=device, dtype=torch.float32); dk_c.zero_()
    dg_cum = torch.empty((BT,), device=device, dtype=torch.float32); dg_cum.zero_()

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
    q: torch.Tensor,           # [B,T,H,K]
    k: torch.Tensor,           # [B,T,H,K]
    v: torch.Tensor,           # [B,T,H,V]
    g_raw: torch.Tensor,       # [B,T,H]
    beta: torch.Tensor,        # [B,T,H]
    initial_state: torch.Tensor,  # [B,H,K,V]
    do: torch.Tensor,          # [B,T,H,V]
    dht: torch.Tensor,         # [B,H,K,V]
    cache: dict,
    BT: int,
    I: torch.Tensor, M_le: torch.Tensor, M_lt: torch.Tensor, C_cum: torch.Tensor, C_rcum: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    l2_eps: float,
):
    device = q.device
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert T % BT == 0
    NT = T // BT
    scale = cache["scale"]

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
    q_used = cache["q_norm"]  # [B,T,H,K] float32
    k_used = cache["k_norm"]

    q_rstd = cache["q_rstd"]  # [B,T,H] or None
    k_rstd = cache["k_rstd"]

    v32 = v.to(torch.float32)
    beta32 = beta.to(torch.float32)
    g_raw32 = g_raw.to(torch.float32)

    for b in range(B):
        for h in range(H):
            # [Tomo] For loop for Chunk is updated, becuase pypto gets wrong if reverse loop. 
            for i in range(NT):
                c = NT - 1 - i
                if i == 0:  # i = 0 -> c = NT-1 (last chunk)
                    dS = dht[b, h].to(torch.float32)

                t0 = c * BT
                t1 = t0 + BT

                # slice chunk inputs (all <=2D)
                qc, kc, vc, betac, gc_raw, doc, A, w, S_before, v_new = _slice_chunk_inputs(
                    q_used, k_used, v32, beta32, g_raw32, do,
                    cache, b, h, c, t0, t1,
                )

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
                    device, BT, K,
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
                    q_used[b, t0:t1, h, :], k_used[b, t0:t1, h, :],
                    q_rstd[b, t0:t1, h] if use_qk_l2norm_in_kernel else None,
                    k_rstd[b, t0:t1, h] if use_qk_l2norm_in_kernel else None,
                    dq_c, dk_c,
                )

                # ===== store into global grads =====
                dq[b, t0:t1, h, :] = dq_raw_c
                dk[b, t0:t1, h, :] = dk_raw_c
                dv[b, t0:t1, h, :] = dv_c
                db[b, t0:t1, h] = db_c
                dg_raw[b, t0:t1, h] = dg_raw_c

            # after all chunks, dS is dh0
            dh0[b, h] = dS

    return dq, dk, dv, db, dg_raw, dh0



def prepare_tensor_for_bwd_from_cache(cache, device, dtype):
    
    def to_5d_tensor(data, device, dtype):
        if data is None:
            return None

        if torch.is_tensor(data):
            return data.to(device=device, dtype=dtype)

        if isinstance(data, list):
            def stack_recursive(item):
                if isinstance(item, list):
                    stacked_items = [stack_recursive(sub) for sub in item]
                    return torch.stack(stacked_items)
                elif torch.is_tensor(item):
                    return item.to(device=device, dtype=dtype)
                else:
                    return torch.tensor(item, device=device, dtype=dtype)

            return stack_recursive(data)
        
        return torch.tensor(data, device=device, dtype=dtype)

    A           = to_5d_tensor(cache.get('A'), device, dtype)
    w           = to_5d_tensor(cache.get('w'), device, dtype)
    u           = to_5d_tensor(cache.get('u'), device, dtype)
    v_new       = to_5d_tensor(cache.get('v_new'), device, dtype)
    S_before    = to_5d_tensor(cache.get('S_before'), device, dtype)

    def to_scalar_tensor(data, name):
        if data is None: return None
        t = torch.as_tensor(data, device=device, dtype=dtype)
        if name in ['scale', 'use_qk_l2norm_in_kernel']:
            return t.view(1, 1)
        return t

    q_norm      = to_scalar_tensor(cache.get('q_norm'), 'q_norm').squeeze(0)
    k_norm      = to_scalar_tensor(cache.get('k_norm'), 'k_norm').squeeze(0)
    q_rstd      = to_scalar_tensor(cache.get('q_rstd'), 'q_rstd').squeeze(0)
    k_rstd      = to_scalar_tensor(cache.get('k_rstd'), 'k_rstd').squeeze(0)
    final_state = to_scalar_tensor(cache.get('final_state'), 'final_state')
    scale       = to_scalar_tensor(cache.get('scale'), 'scale')
    use_qk_l2norm_in_kernel = to_scalar_tensor(cache.get('use_qk_l2norm_in_kernel'), 'use_qk_l2norm_in_kernel')


    # print out the shape
    debug_list = [
        ("A", A),
        ("w", w),
        ("u", u),
        ("v_new", v_new),
        ("S_before", S_before),
        ("q_norm", q_norm),
        ("k_norm", k_norm),
        ("q_rstd", q_rstd),
        ("k_rstd", k_rstd),
        ("final_state", final_state),
        ("scale", scale),
        ("use_qk_l2norm", use_qk_l2norm_in_kernel)
    ]

    print(f"\n{'='*85}")
    print(f"{'Cache Key':<25} | {'Shape':<35} | {'Dtype':<15}")
    print(f"{'-'*85}")

    for name, t in debug_list:
        if t is not None:
            shape_str = str(list(t.shape))
            dtype_str = str(t.dtype).replace('torch.', '')
            print(f"{name:<25} | {shape_str:<35} | {dtype_str:<15}")
        else:
            print(f"{name:<25} | {'None':<35} | {'-'*15}")

    print(f"{'='*85}\n")

    cache_tensors = [A, w, u, v_new, S_before, q_norm, k_norm, q_rstd, k_rstd, final_state, scale, use_qk_l2norm_in_kernel]
    return cache_tensors


# ------------------------------------------------------
# Module 1 on PyPTO
# ------------------------------------------------------
def pypto_slice_chunk_inputs(
    q_norm_cache, k_norm_cache, v_in, beta_in, g_raw_in, do_in,
    A_cache, w_cache, S_before_cache, v_new_cache,
    L, D, b_idx, s_idx, rev_idx, nv_idx, nqk_idx, actual_L, nqk_heads, seq_len, H, NT
    ):
    # pypto.set_vec_tile_shapes(16, 16, 128, 128)
    qkv_cache_row = (b_idx * nqk_heads + nqk_idx) * seq_len + s_idx
    data_cache_row = (b_idx * H + nv_idx) * seq_len + s_idx
    query_used_2d = pypto.view(q_norm_cache, [L, D], [qkv_cache_row, 0], valid_shape =[actual_L, D])
    key_used_2d = pypto.view(k_norm_cache, [L, D], [qkv_cache_row, 0], valid_shape =[actual_L, D])
    value_2d = pypto.view(v_in, [L, D], [data_cache_row, 0], valid_shape =[actual_L, D])
    do_2d =  pypto.view(do_in, [L, D], [data_cache_row, 0], valid_shape =[actual_L, D])
    betac_2d = pypto.view(beta_in, [L, D], [data_cache_row, 0], valid_shape =[actual_L, D])

    # pypto.set_vec_tile_shapes(16, 16, 128)
    gc_raw_2d = pypto.view(g_raw_in, [1, L], [data_cache_row, 0], valid_shape =[1, actual_L])
    
    # pypto.set_vec_tile_shapes(16, 128, 128)
    cache_idx = (b_idx * H + nv_idx) * NT + rev_idx
    cache_bt_row = cache_idx * L
    cache_d_row = cache_idx * D
    A_view_2d = pypto.view(A_cache, [L, L], [cache_bt_row, 0])
    w_view_2d = pypto.view(w_cache, [L, D], [cache_bt_row, 0])
    S_before_view_2d = pypto.view(S_before_cache, [D, D], [cache_d_row, 0])
    v_new_view_2d = pypto.view(v_new_cache, [L, D], [cache_bt_row, 0])

    return query_used_2d, key_used_2d, value_2d, do_2d, betac_2d, gc_raw_2d, A_view_2d, w_view_2d, S_before_view_2d, v_new_view_2d


# ------------------------------------------------------
# Module 2 on PyPTO
# ------------------------------------------------------
def pypto_g_and_decay_kernel(gc_raw_2d, C_cum_2d):
    pypto.set_vec_tile_shapes(256, 128)
    g_cum_2d = pypto.matmul(C_cum_2d, gc_raw_2d, pypto.DT_FP32, a_trans=False, b_trans=True)
    pypto.set_pass_options(sg_set_scope=1)
    diff = g_cum_2d - g_cum_2d.transpose(0, 1)
    decay = pypto.exp(diff)
    pypto.set_pass_options(sg_set_scope=-1)

    return g_cum_2d, decay


# ------------------------------------------------------
# Fused Module 3 + 4
# ------------------------------------------------------
def pypto_fused_34_local_attn_and_recurrence_backprop(
    query_used_2d, key_used_2d, doc, decay_2d, M_le_in, scale_scalar,
    dS_in, g_cum_2d, w_view_2d, dim, l
):
    pypto.set_vec_tile_shapes(256, 128)
    qk = pypto.matmul(query_used_2d, key_used_2d, pypto.DT_FP32, a_trans=False, b_trans=True)
    kkt = pypto.matmul(key_used_2d, key_used_2d, pypto.DT_FP32, a_trans=False, b_trans=True)

    pypto.set_pass_options(sg_set_scope=1)
    A_local = qk * decay_2d * M_le_in
    eg = pypto.exp(g_cum_2d)
    eg_ld = pypto.expand_clone(eg, [l, dim])
    gl_exp_1d = eg_ld[-1:,:]
    s_tok_ld = gl_exp_1d/eg_ld
    q_eff = query_used_2d * eg_ld
    pypto.set_pass_options(sg_set_scope=-1)

    dv_from_state = pypto.matmul(key_used_2d, dS_in, pypto.DT_FP32, a_trans=False, b_trans=False)
    dv_from_local = pypto.matmul(A_local, doc, pypto.DT_FP32, a_trans=True, b_trans=False)
    dv_total = dv_from_state * s_tok_ld + dv_from_local * scale_scalar
    term2 = pypto.matmul(q_eff, doc, pypto.DT_FP32, a_trans=True, b_trans=False)
    term3 = pypto.matmul(w_view_2d, dv_total, pypto.DT_FP32, a_trans=True, b_trans=False)
    dS_final = term2 * scale_scalar - term3 + dS_in * gl_exp_1d 
    return qk, kkt, s_tok_ld, dv_total, dS_final, gl_exp_1d, eg_ld, doc


# ------------------------------------------------------
# Fused Module 5 + 6
# ------------------------------------------------------
def pypto_fused_56_qkg_and_wy_repr(
    qc_in, kc_in, v_new_in, doc_in, gl_exp_1d, s_tok_in, dS_next_in, S_before_in,
    qk_in, M_le_in, scale_scalar, dv_total_in,
    vc, betac, eg, A_view_2d, M_lt_in, decay_2d, kkt, ones_1l, ones_1d, C_rcum, L
):
    pypto.set_vec_tile_shapes(256, 128)
    v_scaled = v_new_in * s_tok_in
    dq_c_tmp = pypto.matmul(doc_in, S_before_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    dk_state = pypto.matmul(v_scaled, dS_next_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    dw_pos = pypto.matmul(dv_total_in, S_before_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    
    
    dA_based_tmp = pypto.matmul(doc_in, v_new_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    dq_c_term1 = dA_based_tmp * decay_2d * M_le_in * scale_scalar
    dq_c_term2 = pypto.matmul(dq_c_term1, kc_in, pypto.DT_FP32, a_trans=False, b_trans=False)
    dk_c_term1 = pypto.matmul(dq_c_term1, qc_in, pypto.DT_FP32, a_trans=True, b_trans=False)
    

    pypto.set_pass_options(sg_set_scope=1)
    dqg_scale = dq_c_tmp * eg * scale_scalar
    dq_c = dqg_scale + dq_c_term2
    vb = vc * betac
    beg = betac * eg
    kbg = kc_in * beg
    pypto.set_pass_options(sg_set_scope=-1)

    dvb = pypto.matmul(A_view_2d, dv_total_in, pypto.DT_FP32, a_trans=True, b_trans=False)
    dkbg = pypto.matmul(A_view_2d, dw_pos, pypto.DT_FP32, a_trans=True, b_trans=False)
    dv_c = dvb * betac
    dA_term1_pos = pypto.matmul(dw_pos, kbg, pypto.DT_FP32, a_trans=False, b_trans=True)
    dA_term2 = pypto.matmul(dv_total_in, vb, pypto.DT_FP32, a_trans=False, b_trans=True)
    dL_part1 = pypto.matmul(A_view_2d, dA_term2, pypto.DT_FP32, a_trans=True, b_trans=False)
    dL_part1 = pypto.matmul(dL_part1, A_view_2d, pypto.DT_FP32, a_trans=False, b_trans=True)
    dL_part2 = pypto.matmul(A_view_2d, dA_term1_pos, pypto.DT_FP32, a_trans=True, b_trans=False)
    dL_part2 = pypto.matmul(dL_part2, A_view_2d, pypto.DT_FP32, a_trans=False, b_trans=True)
    dL_masked = (dL_part1 - dL_part2) * M_lt_in
    db_c_tmp = dvb * vc - dkbg * kc_in * eg

    pypto.set_pass_options(sg_set_scope=2)
    mmat_pos = dL_masked * betac[:, 0:L] * decay_2d
    kkt_M = kkt * M_lt_in
    dg_cum_term_pos = mmat_pos * kkt_M
    mmat_sum = mmat_pos + mmat_pos.transpose(0, 1)
    pypto.set_pass_options(sg_set_scope=-1)

    bc_left = pypto.matmul(db_c_tmp, ones_1d, pypto.DT_FP32, a_trans=False, b_trans=True)
    bc_right = pypto.matmul(kkt * decay_2d * dL_masked, ones_1l, pypto.DT_FP32, a_trans=False, b_trans=True)
    db_c = bc_left - bc_right
    dk_c_term2 = pypto.matmul(mmat_sum, kc_in, pypto.DT_FP32, a_trans=False, b_trans=False)
    dk_c = dk_state + dk_c_term1 - dk_c_term2 - dkbg * beg
    term_row = pypto.matmul(dg_cum_term_pos, ones_1l, pypto.DT_FP32, a_trans=False, b_trans=True)
    term_col = pypto.matmul(dg_cum_term_pos, ones_1l, pypto.DT_FP32, a_trans=True, b_trans=True)
    dg_cum_presum = dq_c_tmp * qc_in * eg * scale_scalar - kc_in * dk_state - dkbg * kbg
    dg_cum_tmp6 = pypto.matmul(dg_cum_presum, ones_1d, pypto.DT_FP32, a_trans=False, b_trans=True)

    pypto.set_pass_options(sg_set_scope=4)
    dAA_tmp = dA_based_tmp * decay_2d * M_le_in * qk_in * scale_scalar
    dg_cum_tmp4 = dAA_tmp.sum(dim=-1, keepdim=True) - dAA_tmp.sum(dim=-2, keepdim=True).transpose(-1, -2)
    tail_add2 = (kc_in * dk_state).sum(dim=-1, keepdim=True).sum(dim=0, keepdim=True) + gl_exp_1d[:,0:1] * (S_before_in * dS_next_in).sum(dim=-1, keepdim=True).sum(dim=0, keepdim=True)
    dg_cum = term_col - term_row + dg_cum_tmp6 + M_le_in[:, -1:] * tail_add2 + dg_cum_tmp4
    pypto.set_pass_options(sg_set_scope=-1)

    dg_cum_out = pypto.matmul(C_rcum, dg_cum, pypto.DT_FP32, b_trans=False)
    
    return dq_c, dg_cum_out, dk_c, dv_c, db_c


# ------------------------------------------------------
# Module 7 on PyPTO
# ------------------------------------------------------
def pypto_l2norm_bwd(
    yq, rstd_yq, dyq,
    ):
    pypto.set_pass_options(sg_set_scope=3)
    dot_q = (dyq * yq).sum(-1, keepdim=True)
    d = dyq * rstd_yq - dot_q * yq * rstd_yq
    pypto.set_pass_options(sg_set_scope=-1)
    return d

def pypto_finalize_chunk_grads(
    q_used, k_used, q_rstd, k_rstd, dq_c, dk_c, use_qk_l2norm_in_kernel_cache
):  
    pypto.set_vec_tile_shapes(256, 128)
    use_l2norm_sym = pypto.SymbolicScalar(use_qk_l2norm_in_kernel_cache[0])
    if pypto.cond(use_l2norm_sym == 1):
        dq_raw_c = pypto_l2norm_bwd(q_used, q_rstd, dq_c)
        dk_raw_c = pypto_l2norm_bwd(k_used, k_rstd, dk_c)
    else:
        # F5: pass through gradients; use * 1.0 to avoid PyPTO A=B memory alias
        dq_raw_c = dq_c * 1.0
        dk_raw_c = dk_c * 1.0

    return dq_raw_c, dk_raw_c


def _gated_delta_rule_bwd_kernel_impl(
    v_in, g_raw_in, beta_in, do_in, dht_in,
    M_le_in, M_lt_in, C_cum_in_2d, C_rcum_in, ones_1l, ones_1d,
    A_cache, w_cache, v_new_cache, S_before_cache,
    q_norm_cache, k_norm_cache, q_rstd_cache, k_rstd_cache,
    use_qk_l2norm_in_kernel_cache,
    dq_out, dk_out, dv_out, db_out, dg_raw_out, dh0_out,
):
    batch = dh0_out.shape[0]
    nv = dh0_out.shape[1]
    dim = dh0_out.shape[2]
    nqk = dq_out.shape[2]
    dyn_seq = dq_out.shape[1]
    l = M_le_in.shape[0]
    dyn_chunks = dyn_seq // l
    group = nv // nqk
    scale_scalar = 1 / (dim ** 0.5)

    dS_2d = pypto.tensor([dim, dim], pypto.DT_FP32)

    for b_idx in pypto.loop(batch, name="LOOP_B_BSND", idx_name="b_idx"):
        for nv_idx in pypto.loop(nv, name="LOOP_Nv_BSND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(256, 128)
            dht_cache_row = (b_idx * nv + nv_idx) * dim
            dS_2d = pypto.view(dht_in, [dim, dim], [dht_cache_row, 0])
            for inv_s_idx in pypto.loop(0, dyn_seq, l, name="LOOP_S_REVERSE_BSND", idx_name="i_idx", unroll_list=[32]):
                s_idx = dyn_seq - inv_s_idx - l
                rev_idx = s_idx // l
                actual_L = (dyn_seq - s_idx).min(l)
                # pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

                pypto.set_vec_tile_shapes(256, 128)
                qkv_cache_row = (b_idx * nqk + nqk_idx) * dyn_seq + s_idx
                q_rstd_2d = pypto.view(q_rstd_cache, [l, dim], [qkv_cache_row, 0], valid_shape=[actual_L, dim])
                k_rstd_2d = pypto.view(k_rstd_cache, [l, dim], [qkv_cache_row, 0], valid_shape=[actual_L, dim])

                query_used_2d, key_used_2d, value_2d, do_2d, betac_2d, gc_raw_2d, A_view_2d, w_view_2d, S_before_view_2d, v_new_view_2d  = pypto_slice_chunk_inputs(
                    q_norm_cache, k_norm_cache, v_in, beta_in, g_raw_in, do_in,
                    A_cache, w_cache, S_before_cache, v_new_cache,
                    l, dim, b_idx, s_idx, rev_idx, nv_idx, nqk_idx, actual_L, nqk, dyn_seq, nv, dyn_chunks
                )

                g_cum_2d, decay_2d = pypto_g_and_decay_kernel(gc_raw_2d, C_cum_in_2d)

                qk, kkt, s_tok, dv_total, dS_final, gl_exp_1, eg, doc = pypto_fused_34_local_attn_and_recurrence_backprop(
                    query_used_2d, key_used_2d, do_2d, decay_2d, M_le_in, scale_scalar,
                    dS_2d, g_cum_2d, w_view_2d, dim, l
                )

                dq_c, dg_raw_c, dk_c, dv_c, db_c = pypto_fused_56_qkg_and_wy_repr(
                    query_used_2d, key_used_2d, v_new_view_2d, doc, gl_exp_1, s_tok, dS_2d, S_before_view_2d,
                    qk, M_le_in, scale_scalar, dv_total,
                    value_2d, betac_2d, eg, A_view_2d, M_lt_in, decay_2d, kkt, ones_1l, ones_1d, C_rcum_in, l
                )

                dq_raw_c, dk_raw_c = pypto_finalize_chunk_grads(
                    query_used_2d, key_used_2d, q_rstd_2d, k_rstd_2d, dq_c, dk_c, use_qk_l2norm_in_kernel_cache
                )

                dS_2d[:] = dS_final

                pypto.set_vec_tile_shapes(16, 128, 128, 128)
                dq_out[b_idx, s_idx:s_idx + l, nqk_idx] = dq_raw_c
                dk_out[b_idx, s_idx:s_idx + l, nqk_idx] = dk_raw_c
                dv_out[b_idx, s_idx:s_idx + l, nv_idx] = dv_c
                db_out[b_idx, s_idx:s_idx + l, nv_idx:nv_idx + 1] = db_c
                dg_raw_out[b_idx, s_idx:s_idx + l, nv_idx:nv_idx + 1] = dg_raw_c

            dh0_out[b_idx, nv_idx] = dS_2d


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 16,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 16,
        "device_sched_mode": 1,
        "run_mode": pypto.RunMode.NPU},
    debug_options={"runtime_debug_mode": 1}
)
def gated_delta_rule_bwd_kernel_npu(
    v_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    g_raw_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    beta_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    do_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    dht_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    M_le_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    M_lt_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    C_cum_in_2d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    C_rcum_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ones_1l: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ones_1d: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    A_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    w_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    v_new_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    S_before_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    q_norm_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    k_norm_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    q_rstd_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    k_rstd_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    use_qk_l2norm_in_kernel_cache: pypto.Tensor([pypto.STATIC], pypto.DT_INT32),
    dq_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    dk_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    dv_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    db_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    dg_raw_out: pypto.Tensor([pypto.STATIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    dh0_out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    _gated_delta_rule_bwd_kernel_impl(
        v_in, g_raw_in, beta_in, do_in, dht_in,
        M_le_in, M_lt_in, C_cum_in_2d, C_rcum_in, ones_1l, ones_1d,
        A_cache, w_cache, v_new_cache, S_before_cache,
        q_norm_cache, k_norm_cache, q_rstd_cache, k_rstd_cache,
        use_qk_l2norm_in_kernel_cache,
        dq_out, dk_out, dv_out, db_out, dg_raw_out, dh0_out,
    )


def pypto_function(
    q, v, g_raw, beta, do, dht,
    cache,
    M_le, M_lt, C_cum, C_rcum, ones_1l, ones_1d,
    act_seq_len
    ):

    device = q.device
    B, S, Nqk, D = q.shape
    Nv = cache['final_state'].shape[1]
    B = len(act_seq_len)

    # Cache from forward
    A_cache = cache['A'].to(device)
    w_cache = cache['w'].to(device)
    v_new_cache = cache['v_new'].to(device)
    S_before_cache = cache['S_before'].to(device)
    q_norm_cache = cache['q_norm'].to(device)
    k_norm_cache = cache['k_norm'].to(device)
    q_rstd_cache = cache['q_rstd'].to(device)
    k_rstd_cache = cache['k_rstd'].to(device)
    use_qk_l2norm_in_kernel_cache =  torch.tensor([cache['use_qk_l2norm_in_kernel']], dtype=torch.int).to(device)

    input_tensors = [
        v, g_raw, beta, do, dht, M_le, M_lt, C_cum, C_rcum, ones_1l, ones_1d,
        A_cache, w_cache, v_new_cache, S_before_cache, 
        q_norm_cache, k_norm_cache, q_rstd_cache, k_rstd_cache, 
        use_qk_l2norm_in_kernel_cache
    ]

    dq = torch.empty((B, S, Nqk, D), dtype=torch.float32, device=device)
    dk = torch.empty((B, S, Nqk, D), dtype=torch.float32, device=device)
    dv = torch.empty((B, S, Nv, D), dtype=torch.float32, device=device)
    db = torch.empty((B, S, Nv), dtype=torch.float32, device=device)
    dg_raw = torch.empty((B, S, Nv), dtype=torch.float32, device=device)
    dh0 = torch.empty((B, Nv, D, D), dtype=torch.float32, device=device)

    output_tensors = [dq, dk, dv, db, dg_raw, dh0]

    gated_delta_rule_bwd_kernel_npu(*input_tensors, *output_tensors)
    print('>>> pypto done')

    return dq, dk, dv, db, dg_raw, dh0

def main():
    torch.manual_seed(0)
    device_id = 0
    torch.npu.set_device(device_id)
    run_mode = 'npu'
    device = f'{run_mode}:{device_id}'
    dtype = torch.float32

    # -----------------------------------------
    # Input Shape Config
    # -----------------------------------------
    S = 4096
    Nqk = 4
    Nv = 4
    D = 128 
    BT = 128
    act_seq_len = [S]
    B = len(act_seq_len)
    use_l2 = True
    eps = 1e-6
    assert S % BT == 0

    # -----------------------------------------
    # Init input tensor
    # -----------------------------------------
    torch.manual_seed(0)
    scale_input_tensor = 0.1
    q = (torch.randn(B, S, Nqk, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    k = (torch.randn(B, S, Nqk, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    v = (torch.randn(B, S, Nv, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    g_raw = (torch.randn(B, S, Nv, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    beta = torch.rand(B, S, Nv, device=device, dtype=dtype).requires_grad_(True)
    initial_state = (torch.randn(B, Nv, D, D, device=device, dtype=dtype) * 0.01).requires_grad_(True)

    # -----------------------------------------
    # constants (provided as inputs to ref impls)
    # -----------------------------------------
    I, M_le, M_lt, C_cum, C_rcum, ones_1l, ones_1d = make_chunk_constants(BT, D, device=device, dtype=torch.float32)

    # -----------------------------------------
    # Preparation (Forward Pass)
    # -----------------------------------------
    o_ref_bsnd, ht_ref_bsnd, cache = forward_ref(
        q=q, k=k, v=v, g_raw=g_raw, beta=beta, initial_state=initial_state,
        BT=BT, use_qk_l2norm_in_kernel=use_l2, l2_eps=eps,
        I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum,
    )

    # -----------------------------------------
    # Preparation (upstream grads)
    # -----------------------------------------
    do_bsnd = torch.randn_like(o_ref_bsnd)
    dht_bsnd = torch.randn_like(ht_ref_bsnd)

    # -----------------------------------------
    # Golden function by PyTorch
    # -----------------------------------------
    Bg, Tg, Hg, Kg = q.shape
    _, _, _, Vg = v.shape
    NTg = Tg // BT
    cache_for_golden = {
        "A": cache['A'].reshape([Bg, Hg, NTg, BT, BT]),
        "w": cache['w'].reshape([Bg, Hg, NTg, BT, Vg]),
        "u": cache['u'].reshape([Bg, Hg, NTg, BT, Vg]),
        "v_new": cache['v_new'].reshape([Bg, Hg, NTg, BT, Vg]),
        "S_before": cache['S_before'].reshape([Bg, Hg, NTg, Kg, Vg]),
        "use_qk_l2norm_in_kernel": cache['use_qk_l2norm_in_kernel'],
        "q_norm": cache['q_norm'].reshape(Bg, Hg, Tg, Kg).permute(0, 2, 1, 3).contiguous(),
        "k_norm": cache['k_norm'].reshape(Bg, Hg, Tg, Kg).permute(0, 2, 1, 3).contiguous(),
        "q_rstd": cache['q_rstd'],
        "k_rstd": cache['k_rstd'],
        "scale": cache['scale'],
        "BT": BT,
        "final_state": cache['final_state'],
    }
    with torch.no_grad():
        dq, dk, dv, db, dg_raw, dh0 = torch_golden_gated_delta_rule_backward_ref(
            q=q.clone().detach(), k=k.clone().detach(), v=v.clone().detach(),
            g_raw=g_raw.clone().detach(), beta=beta.clone().detach(),
            initial_state=initial_state.clone().detach(),
            do=do_bsnd.clone().detach(), dht=dht_bsnd.clone().detach(),
            cache=cache_for_golden,
            BT=I.shape[0],
            I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum, C_rcum=C_rcum,
            use_qk_l2norm_in_kernel=use_l2,
            l2_eps=eps,
        )

    # -----------------------------------------
    # Preparation for PyPTO Function
    # Data adaptation
    # -----------------------------------------
    cache_for_pypto = {
        "A": cache['A'],
        "w": cache['w'],
        "u": cache['u'],
        "v_new": cache['v_new'],
        "S_before": cache['S_before'],
        "use_qk_l2norm_in_kernel": cache['use_qk_l2norm_in_kernel'],
        "q_norm": cache['q_norm'],
        "k_norm": cache['k_norm'],
        "q_rstd": cache['q_rstd'].permute(0, 2, 1).contiguous().reshape(B * Nqk * S, 1).expand(B * Nqk * S, D).contiguous(),
        "k_rstd": cache['k_rstd'].permute(0, 2, 1).contiguous().reshape(B * Nqk * S, 1).expand(B * Nqk * S, D).contiguous(),
        "scale": cache['scale'],
        "BT": BT,
        "final_state": cache['final_state'],
    }
    C_cum_2d = C_cum.to(dtype).contiguous()
    gate_for_pypto = g_raw.transpose(1, 2).contiguous().reshape(B * Nv * S, 1)
    beta_for_pypto = beta.transpose(1, 2).contiguous().reshape(B * Nv * S, 1).expand(B * Nv * S, D).contiguous()
    v_for_pypto = v.transpose(1, 2).contiguous().reshape(B * Nv * S, D)
    do_for_pypto = do_bsnd.transpose(1, 2).contiguous().reshape(B * Nv * S, D)
    dht_for_pypto = dht_bsnd.contiguous().reshape(B * Nv * D, D)

    # -----------------------------------------
    # PyPTO Gated Delta Rule Backward Kernel
    # -----------------------------------------
    with torch.no_grad():
        pto_dq, pto_dk, pto_dv, pto_db, pto_dg_raw, pto_dh0  = pypto_function(
            q=q.detach(), v=v_for_pypto.detach(),
            g_raw=gate_for_pypto.detach(), beta=beta_for_pypto.detach(),
            do=do_for_pypto.detach(), dht=dht_for_pypto.detach(),
            cache=cache_for_pypto,
            M_le=M_le, M_lt=M_lt, C_cum=C_cum_2d, C_rcum=C_rcum, ones_1l=ones_1l, ones_1d=ones_1d,
            act_seq_len=act_seq_len
        )
    
    # -----------------------------------------
    # Assesment
    # -----------------------------------------
    detailed_tensor_compare(pto_dq, dq, 'dq')
    detailed_tensor_compare(pto_dk, dk, 'dk')
    detailed_tensor_compare(pto_dv, dv, 'dv')
    detailed_tensor_compare(pto_db, db, 'db')
    detailed_tensor_compare(pto_dg_raw, dg_raw, 'dg_raw')
    detailed_tensor_compare(pto_dh0, dh0, 'dh0')
    print("\n✅ All checks (Gated Delta Rule Backward).")
    print("Batch:{} | Seq:{} | Nv:{} | Nqk:{} | D:{} | Chunksize:{}".format(B,S,Nv,Nqk,D,BT))


if __name__ == "__main__":
    main()

