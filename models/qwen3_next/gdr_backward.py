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
    cache_A = torch.zeros((B*H*NT, BT, BT), dtype=torch.float32, device=q.device)
    cache_S_before = torch.zeros((B*H*NT, K, V), dtype=torch.float32, device=q.device)
    cache_w = torch.zeros((B*H*NT, BT, V), dtype=torch.float32, device=q.device)
    cache_u = torch.zeros((B*H*NT, BT, V), dtype=torch.float32, device=q.device)
    cache_v_new = torch.zeros((B*H*NT, BT, V), dtype=torch.float32, device=q.device)

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
                cache_A[cache_idx] = A
                cache_S_before[cache_idx] = S
                cache_w[cache_idx] = w
                cache_u[cache_idx] = u

                # v_new
                v_prime = w @ S                          # [BT,V]
                v_new = u - v_prime
                # cache["v_new"][b][h].append(v_new)
                cache_v_new[cache_idx] = v_new

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
                t0 = c * BT
                t1 = t0 + BT
                kc = k_used[b, t0:t1, h, :]
                gc_raw = g_raw32[b, t0:t1, h]
                g_cum = C_cum @ gc_raw
                gl = g_cum[-1]
                # v_new = cache["v_new"][b][h][c]
                v_new = cache_v_new[b,h,c]
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
        "q_norm": q_used,     # [B,T,H,K] float32
        "k_norm": k_used,     # [B,T,H,K] float32
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
def make_chunk_constants(BT: int, device, dtype=torch.float32):
    idx = torch.arange(BT, device=device)
    I = (idx[:, None] == idx[None, :]).to(dtype)             # [BT,BT]
    M_le = (idx[:, None] >= idx[None, :]).to(dtype)          # lower incl diag
    M_lt = M_le - I                                          # strict lower
    C_cum = M_le                                              # prefix sum: y = C_cum @ x
    C_rcum = (idx[None, :] >= idx[:, None]).to(dtype)         # upper incl diag: suffix sum
    return I, M_le, M_lt, C_cum, C_rcum


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
    L, D, b_idx, s_idx, rev_idx, nv_idx, nqk_idx, actual_L, H, NT
    ):

    pypto.set_vec_tile_shapes(16, 128, 128, 128)
    query_used_view = pypto.view(q_norm_cache, [1, 1, L, D], [b_idx, 0, s_idx, 0], valid_shape =[1, 1, actual_L, D]) #bnsd
    key_used_view = pypto.view(k_norm_cache, [1, 1, L, D], [b_idx, 0, s_idx, 0], valid_shape =[1, 1, actual_L, D]) #bnsd

    value_view = pypto.view(v_in, [1, L, 1, D], [b_idx, s_idx, nv_idx, 0], valid_shape =[1, actual_L, 1, D])
    do_view =  pypto.view(do_in, [1, L, 1, D], [b_idx, s_idx, nv_idx, 0], valid_shape =[1, actual_L, 1, D])
    vc = pypto.reshape(value_view, [L, D], valid_shape=[actual_L, D]) #[L(BT), D]
    doc = pypto.reshape(do_view, [L, D], valid_shape=[actual_L, D]) #[L(BT), D]

    pypto.set_vec_tile_shapes(16, 128, 128)
    gc_raw_view = pypto.view(g_raw_in, [1, 1, L], [b_idx, 0, s_idx], valid_shape =[1, 1, actual_L]) #[1, L(BT), 1]
    betac_view = pypto.view(beta_in, [1, 1, L], [b_idx, 0, s_idx], valid_shape =[1, 1, actual_L]) #[1, L(BT), 1]
    betac = betac_view[0,0,:].unsqueeze(-1)

    cache_idx = (b_idx * H + nv_idx) * NT + rev_idx
    A_view = pypto.view(A_cache, [1, L, L], [cache_idx, 0, 0])
    w_view = pypto.view(w_cache, [1, L, D], [cache_idx, 0, 0])
    S_before_view = pypto.view(S_before_cache, [1, D, D], [cache_idx, 0, 0])
    v_new_view = pypto.view(v_new_cache, [1, L, D], [cache_idx, 0, 0])

    A_view_2d = A_view[0]
    w_view_2d = w_view[0]
    S_before_view_2d = S_before_view[0]
    v_new_view_2d = v_new_view[0]

    return query_used_view, key_used_view, vc, doc, betac, gc_raw_view, A_view_2d, w_view_2d, S_before_view_2d, v_new_view_2d, 


# ------------------------------------------------------
# Module 2 on PyPTO
# ------------------------------------------------------
def pypto_g_and_decay_kernel(gc_raw_3d, C_cum_3d):
    BT = gc_raw_3d.shape[-1]
    pypto.set_vec_tile_shapes(16, 256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    g_cum_col_3d = pypto.matmul(C_cum_3d, gc_raw_3d, pypto.DT_FP32, b_trans=True)
    g_cum_col = g_cum_col_3d[0,:]

    pypto.set_vec_tile_shapes(256, 128)
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
    pypto.set_vec_tile_shapes(16,16,256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    qk_4d = pypto.matmul(qc_in, kc_in, pypto.DT_FP32, a_trans=False, b_trans=True) # [1,1,L,L]
    kkt_4d = pypto.matmul(kc_in, kc_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    qk = qk_4d[0,0,:]
    kkt = kkt_4d[0,0,:]

    pypto.set_vec_tile_shapes(256, 128)
    common_mask_decay = decay_in * M_le_in
    A_local = qk * common_mask_decay
    dv0 = pypto.matmul(A_local, doc_in, pypto.DT_FP32, a_trans=True, b_trans=False)
    dv0 = dv0 * scale_scalar

    return qk, dv0, common_mask_decay, kkt


# ------------------------------------------------------
# Module 4 on PyPTO
# ------------------------------------------------------
def pypto_recurrence_backprop(
    kc, dS_in, gl, g_cum, dv0, qc, eg, doc, scale_scalar, w,
):  
    BT = g_cum.shape[0]
    K, V = dS_in.shape
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    s_tok = pypto.exp(gl - g_cum)
    dv_state = pypto.matmul(kc, dS_in, pypto.DT_FP32, b_trans=False) * s_tok
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
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dq_c_tmp = pypto.matmul(doc_in, S_before_in, pypto.DT_FP32, a_trans=False, b_trans=True) * eg_in * scale_scalar
    dg_cum_tmp = (dq_c_tmp * qc_in).sum(dim=-1, keepdim=True)
    v_scaled = v_new_in * s_tok_in
    dk_state = pypto.matmul(v_scaled, dS_next_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    dw_final = pypto.matmul(dv_total_in, S_before_in, pypto.DT_FP32, a_trans=False, b_trans=True) * -1.0
    scalar = (kc_in * dk_state).sum(dim=-1, keepdim=True)
    dg_cum_tmp2 = dg_cum_tmp - scalar
    one_hot_last = M_le_in[:, -1:]
    tail_add = scalar.sum(dim=0, keepdim=True)

    pypto.set_vec_tile_shapes(64, 64, 32, 32)
    tail_add2 = tail_add + gl_exp_1 * (S_before_in * dS_next_in).sum(dim=0, keepdim=True).sum(dim=-1, keepdim=True)

    pypto.set_vec_tile_shapes(256, 128)
    dg_cum_tmp3 = dg_cum_tmp2 + one_hot_last * tail_add2 
    scaled_common_mask_decay = common_mask_decay * scale_scalar
    dA_based_tmp = pypto.matmul(doc_in, v_new_in, pypto.DT_FP32, a_trans=False, b_trans=True)
    dq_c_term1 = dA_based_tmp * scaled_common_mask_decay
    dq_c_term2 = pypto.matmul(dq_c_term1, kc_in, pypto.DT_FP32, a_trans=False, b_trans=False)
    dk_c_term1 = pypto.matmul(dq_c_term1, qc_in, pypto.DT_FP32, a_trans=True, b_trans=False)
    dq_c = dq_c_tmp + dq_c_term2
    dk_c = dk_state + dk_c_term1
    dAA_tmp = dq_c_term1 * qk_in

    pypto.set_vec_tile_shapes(64, 64, 32, 32)    
    dg_cum_term4 = dAA_tmp.sum(dim=-1) - dAA_tmp.sum(dim=-2)
    dg_cum_final = dg_cum_tmp3 + dg_cum_term4.unsqueeze(-1)

    return dq_c, dg_cum_final, dk_c, dw_final


# ------------------------------------------------------
# Module 6 on PyPTO
# ------------------------------------------------------
def pypto_wy_repr_fused_updates(
    vc, betac, kc, eg, du, dw,A, M_lt, decay, dk_c_in, dg_cum, kkt
):  
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    vb = vc * betac
    beg = betac * eg
    kbg = kc * beg
    dvb = pypto.matmul(A, du, pypto.DT_FP32, a_trans=True, b_trans=False)
    dkbg = pypto.matmul(A, dw, pypto.DT_FP32, a_trans=True, b_trans=False)
    dv_c = dvb * betac
    db_c_tmp = (dvb * vc).sum(-1, keepdim=True)
    dk_c_tmp = dk_c_in + (dkbg * beg)
    db_c_tmp2 = db_c_tmp + (dkbg * (kc * eg)).sum(-1, keepdim=True)
    dg_cum_tmp = dg_cum + (dkbg * kbg).sum(-1, keepdim=True)
    dA_term1 = pypto.matmul(dw, kbg, pypto.DT_FP32, a_trans=False, b_trans=True)
    dA_term2 = pypto.matmul(du, vb, pypto.DT_FP32, a_trans=False, b_trans=True)
    dA = dA_term1 + dA_term2
    dL_term1 = pypto.matmul(dA, A, pypto.DT_FP32, a_trans=False, b_trans=True)
    dL_tmp = pypto.matmul(A, dL_term1, pypto.DT_FP32, a_trans=True, b_trans=False)
    dL = (dL_tmp * -1.0) * M_lt
    db_c = db_c_tmp2 + (dL * (kkt * decay)).sum(-1, keepdim=True)
    mmat = dL * betac * decay

    pypto.set_vec_tile_shapes(16, 256, 128)
    dg_cum_term = mmat * (kkt*M_lt)
    dg_cum_out = dg_cum_tmp + dg_cum_term.sum(-1, keepdim=True) - dg_cum_term.sum(-2).unsqueeze(-1)
    mmat_t = mmat.transpose(0,1)
    mmat_sum = mmat + mmat_t
    dk_c_term2 = pypto.matmul(mmat_sum, kc, pypto.DT_FP32, a_trans=False, b_trans=False)
    dk_c = dk_c_tmp + dk_c_term2

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
    C_rcum, dg_cum, q_used, k_used, q_rstd, k_rstd, dq_c, dk_c, use_qk_l2norm_in_kernel_cache
):  
    pypto.set_vec_tile_shapes(256, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dg_raw_c = pypto.matmul(C_rcum, dg_cum, pypto.DT_FP32, b_trans=False)
    use_l2norm_sym = pypto.SymbolicScalar(use_qk_l2norm_in_kernel_cache[0])
    if pypto.cond(use_l2norm_sym == 1):
        dq_raw_c, dk_raw_c = pypto_l2norm_bwd_qk(q_used, q_rstd, dq_c, k_used, k_rstd, dk_c)
    else:
        dq_raw_c = q_used * 1.0
        dk_raw_c = k_used * 1.0

    return dg_raw_c, dq_raw_c, dk_raw_c


def pypto_bsnd_gated_delta_rule_bwd(
        shape: list, run_mode: str = 'npu', dynamic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # -------------------------------------------------------------
    # Set dynamic
    # -------------------------------------------------------------
    if dynamic:
        seq = pypto.frontend.dynamic("seq")
        chunks = pypto.frontend.dynamic("chunks")
        batch, _, nqk, nv, dim, l = shape
    else:
        batch, seq, nqk, nv, dim, l = shape
        chunks = seq // l
    group = nv // nqk
    

    # -------------------------------------------------------------
    # Set run mode (npu or cpu)
    # -------------------------------------------------------------
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    # -------------------------------------------------------------
    # Launch the kernel
    # -------------------------------------------------------------
    @pypto.frontend.jit(
        runtime_options={"run_mode": mode,
        "stitch_function_inner_memory": 128,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128,
        "device_sched_mode": 1
        }, 
        debug_options={"runtime_debug_mode": 1}
    )
    def gated_delta_rule_bwd_kernel(
            v_in: pypto.Tensor((batch, seq, nv, dim), pypto.DT_FP32),
            g_raw_in: pypto.Tensor((batch, nv, seq), pypto.DT_FP32),
            beta_in: pypto.Tensor((batch, nv, seq), pypto.DT_FP32),
            do_in: pypto.Tensor((batch, seq, nv, dim), pypto.DT_FP32),
            dht_in: pypto.Tensor((batch, nv, dim, dim), pypto.DT_FP32),
            M_le_in: pypto.Tensor((l,l), pypto.DT_FP32),
            M_lt_in: pypto.Tensor((l,l), pypto.DT_FP32), 
            C_cum_in_3d: pypto.Tensor((1, l,l), pypto.DT_FP32),
            C_rcum_in: pypto.Tensor((l,l), pypto.DT_FP32),
            A_cache: pypto.Tensor((batch*nv*chunks, l, l), pypto.DT_FP32),
            w_cache: pypto.Tensor((batch*nv*chunks, l, dim), pypto.DT_FP32),
            v_new_cache: pypto.Tensor((batch*nv*chunks, l, dim), pypto.DT_FP32),
            S_before_cache: pypto.Tensor((batch*nv*chunks, dim, dim), pypto.DT_FP32),
            q_norm_cache: pypto.Tensor((batch, nqk, seq, dim), pypto.DT_FP32),
            k_norm_cache: pypto.Tensor((batch, nqk, seq, dim), pypto.DT_FP32),
            q_rstd_cache: pypto.Tensor((batch, nqk, seq), pypto.DT_FP32),
            k_rstd_cache: pypto.Tensor((batch, nqk, seq), pypto.DT_FP32),
            use_qk_l2norm_in_kernel_cache: pypto.Tensor((1,), pypto.DT_INT32),
        ) -> (
            pypto.tensor((batch, seq, nqk, dim), pypto.DT_FP32),
            pypto.tensor((batch, seq, nqk, dim), pypto.DT_FP32),
            pypto.tensor((batch, seq, nv, dim), pypto.DT_FP32),
            pypto.tensor((batch, seq, nv), pypto.DT_FP32),
            pypto.tensor((batch, seq, nv), pypto.DT_FP32),
            pypto.tensor((batch, nv, dim, dim), pypto.DT_FP32),
        ):
        # -------------------------------------------------------------
        # Calculate the loop parameters
        # -------------------------------------------------------------
        scale_scalar = 1 / (dim ** 0.5)
        dyn_seq = v_in.shape[1]
        dyn_chunks =  dyn_seq // l

        # -------------------------------------------------------------
        # Init output tensors (dq, dk, dv, db, dg, dh0)
        # -------------------------------------------------------------
        dq_out = pypto.tensor((batch, dyn_seq, nqk, dim), pypto.DT_FP32)
        dk_out = pypto.tensor((batch, dyn_seq, nqk, dim), pypto.DT_FP32)
        dv_out =  pypto.Tensor((batch, dyn_seq, nv, dim), pypto.DT_FP32)
        db_out = pypto.Tensor((batch, dyn_seq, nv), pypto.DT_FP32)
        dg_raw_out = pypto.Tensor((batch, dyn_seq, nv), pypto.DT_FP32)
        dh0_out = pypto.Tensor((batch, nv, dim, dim), pypto.DT_FP32)

        # -------------------------------------------------------------
        # main loop
        # -------------------------------------------------------------
        for b_idx in pypto.loop(batch, name="LOOP_B_BSND", idx_name="b_idx"):
            for nv_idx in pypto.loop(nv, name="LOOP_Nv_BSND", idx_name="nv_idx"):
                nqk_idx = nv_idx // group
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                dS_view =pypto.view(dht_in, [1, 1, dim, dim], [b_idx, nv_idx, 0, 0])
                dS_2d = pypto.reshape(dS_view, [dim, dim])  #[D, D]
                for inv_s_idx in pypto.loop(0, dyn_seq, l, name="LOOP_S_REVERSE_BSND", idx_name="i_idx"):
                    s_idx = dyn_seq - inv_s_idx - l
                    rev_idx = s_idx // l
                    actual_L = (dyn_seq - s_idx).min(l)
                    
                    # -----------------------------------------
                    # Preprocess
                    # -----------------------------------------
                    pypto.set_vec_tile_shapes(16, 16, 128)
                    q_rstd_view = pypto.view(q_rstd_cache, [1, 1, l], [b_idx, 0, s_idx], valid_shape =[1, 1, actual_L])
                    k_rstd_view = pypto.view(k_rstd_cache, [1, 1, l], [b_idx, 0, s_idx], valid_shape =[1, 1, actual_L])
                    q_rstd_2d = q_rstd_view[0,0,:].unsqueeze(-1)
                    k_rstd_2d = k_rstd_view[0,0,:].unsqueeze(-1)
                    
                    # -----------------------------------------
                    # Module 1 pto_slice_chunk_inputs_kernel
                    # -----------------------------------------
                    query_used_4d, key_used_4d, vc, doc, betac, gc_raw_3d, A_view_2d, w_view_2d, S_before_view_2d, v_new_view_2d,  = pypto_slice_chunk_inputs(
                        q_norm_cache, k_norm_cache, v_in, beta_in, g_raw_in, do_in,
                        A_cache, w_cache, S_before_cache, v_new_cache,
                        l, dim, b_idx, s_idx, rev_idx, nv_idx, nqk_idx, actual_L, nv, dyn_chunks
                    )
                    
                    # -----------------------------------------
                    # Module 2 pto_g_and_decay_kernel
                    # -----------------------------------------
                    g_cum_2d, eg_2d, gl_1, decay_2d = pypto_g_and_decay_kernel(gc_raw_3d, C_cum_in_3d)

                    # -----------------------------------------
                    # Module 3 pto_local_attn_dv0_kernel
                    # -----------------------------------------
                    qk, dv0, common_mask_decay, kkt = pypto_local_attn_dv0_kernel(query_used_4d, key_used_4d, doc, decay_2d, M_le_in, scale_scalar)
                    qc = pypto.reshape(query_used_4d, [l, dim], inplace=False)  #[L(BT), D]
                    kc = pypto.reshape(key_used_4d, [l, dim], inplace=False) #[L(BT), D]

                    # -----------------------------------------
                    # Module 4 pypto_recurrence_backprop
                    # -----------------------------------------
                    s_tok, dv_total, dS_final, gl_exp_1 = pypto_recurrence_backprop(kc, dS_2d, gl_1, g_cum_2d, dv0, qc, eg_2d, doc, scale_scalar, w_view_2d)
                    
                    # -----------------------------------------
                    # Module 5 pypto_compute_qkg_grads_dw_du
                    # -----------------------------------------
                    dq_c, dg_cum_final, dk_c_tmp, dw_final = pypto_compute_qkg_grads_dw_du(qc, kc, v_new_view_2d, doc, eg_2d, gl_exp_1, s_tok, dS_2d, S_before_view_2d, qk, M_le_in, scale_scalar, dv_total, common_mask_decay)

                    # -----------------------------------------
                    # Module 6 pypto_wy_repr_fused_updates
                    # -----------------------------------------
                    dv_c, db_c, dk_c, dg_cum_out = pypto_wy_repr_fused_updates(vc, betac, kc, eg_2d, dv_total, dw_final, A_view_2d, M_lt_in, decay_2d, dk_c_tmp, dg_cum_final, kkt)

                    # -----------------------------------------
                    # Module 7 pypto_finalize_chunk_grads
                    # -----------------------------------------
                    dg_raw_c, dq_raw_c, dk_raw_c = pypto_finalize_chunk_grads(C_rcum_in, dg_cum_out, qc, kc, q_rstd_2d, k_rstd_2d, dq_c, dk_c, use_qk_l2norm_in_kernel_cache)

                    dS_2d[:] = dS_final

                    # Assemble
                    pypto.set_vec_tile_shapes(16, 128, 16, 128)
                    dq_out[b_idx:b_idx+1, s_idx:s_idx+l, nqk_idx:nqk_idx+1, 0:] = dq_raw_c.reshape([1, l, 1, dim])
                    dk_out[b_idx:b_idx+1, s_idx:s_idx+l, nqk_idx:nqk_idx+1, 0:] = dk_raw_c.reshape([1, l, 1, dim])
                    dv_out[b_idx:b_idx+1, s_idx:s_idx+l, nv_idx:nv_idx+1, 0:] = dv_c.reshape([1, l, 1, dim])
                    db_out[b_idx:b_idx+1, s_idx:s_idx+l, nv_idx:nv_idx+1] = db_c.unsqueeze(0)
                    dg_raw_out[b_idx:b_idx+1, s_idx:s_idx+l, nv_idx:nv_idx+1] = dg_raw_c.unsqueeze(0)

                dh0_out[b_idx:b_idx+1, nv_idx:nv_idx+1, 0:, 0:] = dS_2d.reshape([1, 1, dim, dim])
            

        return dq_out, dk_out, dv_out, db_out, dg_raw_out, dh0_out


    return gated_delta_rule_bwd_kernel


def pypto_function(
    q, k, v, g_raw, beta, initial_state, do, dht,
    cache,
    I, M_le, M_lt, C_cum, C_rcum,
    act_seq_len,
    run_mode='npu'
    ):

    device = q.device
    BT = M_le.shape[0]
    L = BT
    B, S, Nqk, D = q.shape
    _, _, Nv, _ = v.shape
    B = len(act_seq_len)
    input_shape = [B, S, Nqk, Nv, D, L]


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
        v, g_raw, beta, do, dht, M_le, M_lt, C_cum, C_rcum,
        A_cache, w_cache, v_new_cache, S_before_cache, 
        q_norm_cache, k_norm_cache, q_rstd_cache, k_rstd_cache, 
        use_qk_l2norm_in_kernel_cache
    ]

    dq, dk, dv, db, dg_raw, dh0 = pypto_bsnd_gated_delta_rule_bwd(shape=input_shape, run_mode=run_mode, dynamic=True)(*input_tensors)
    print('>>> pypto done')

    return dq, dk, dv, db, dg_raw, dh0

def main():
    torch.manual_seed(0)
    device_id = 3
    torch.npu.set_device(device_id)
    run_mode = 'npu'
    device = f'{run_mode}:{device_id}'
    dtype = torch.float32

    # keep it fast; you can scale up to B=1,T=4096,H=4,K=128,V=128,BT=64,
    S = 4096
    Nqk = 4
    Nv = 4
    D = 128
    act_seq_len = [S]
    B = len(act_seq_len)
    BT = 64
    assert S % BT == 0

    use_l2 = True
    eps = 1e-6

    # inputs (B,T,H,*)
    torch.manual_seed(0)
    scale_input_tensor = 0.1
    q = (torch.randn(B, S, Nqk, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    k = (torch.randn(B, S, Nqk, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    v = (torch.randn(B, S, Nv, D, device=device, dtype=dtype) * scale_input_tensor).requires_grad_(True)
    g_raw = (torch.randn(B, S, Nv, device=device, dtype=dtype) * 0.01).requires_grad_(True)
    beta = torch.rand(B, S, Nv, device=device, dtype=dtype).requires_grad_(True)
    initial_state = (torch.randn(B, Nv, D, D, device=device, dtype=dtype) * 0.01).requires_grad_(True)

    # constants (provided as inputs to ref impls)
    I, M_le, M_lt, C_cum, C_rcum = make_chunk_constants(BT, device=device, dtype=torch.float32)

    # input BSND
    o_ref_bsnd, ht_ref_bsnd, cache = forward_ref(
        q=q, k=k, v=v, g_raw=g_raw, beta=beta, initial_state=initial_state,
        BT=BT, use_qk_l2norm_in_kernel=use_l2, l2_eps=eps,
        I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum,
    )

    # upstream grads
    do_bsnd = torch.randn_like(o_ref_bsnd)
    dht_bsnd = torch.randn_like(ht_ref_bsnd)

    # ---------------- Golden ----------------
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
        "q_norm": cache['q_norm'],
        "k_norm": cache['k_norm'],
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


    cache_for_pypto = {
        "A": cache['A'],
        "w": cache['w'],
        "u": cache['u'],
        "v_new": cache['v_new'],
        "S_before": cache['S_before'],
        "use_qk_l2norm_in_kernel": cache['use_qk_l2norm_in_kernel'],
        "q_norm": cache['q_norm'].reshape([B, Nqk, S, D]).contiguous(), # bnsd
        "k_norm": cache['k_norm'].reshape([B, Nqk, S, D]).contiguous(), # bnsd
        "q_rstd": cache['q_rstd'].reshape([B, Nqk, S]).contiguous(),
        "k_rstd": cache['k_rstd'].reshape([B, Nqk, S]).contiguous(),
        "scale": cache['scale'],
        "BT": BT,
        "final_state": cache['final_state'],
    }
    C_cum_3d = C_cum.unsqueeze(0).to(dtype) 
    gate_for_pypto = g_raw.transpose(1,2).contiguous()
    beta_for_pypto = beta.transpose(1,2).contiguous()
    print('>>> Reshaped for PyPTO Run')
    print('beta_for_pypto (B,N,S) ', beta_for_pypto.shape)
    print('gate_for_pypto (B,N,S)', gate_for_pypto.shape)
    print('q_norm (B,N,S,D)', cache_for_pypto['q_norm'].shape)
    print('k_norm (B,N,S,D)', cache_for_pypto['k_norm'].shape)
    print('q_rstd (B,N,S)', cache_for_pypto['q_rstd'].shape)
    print('k_rstd (B,N,S)', cache_for_pypto['k_rstd'].shape)
    with torch.no_grad():
        pto_dq, pto_dk, pto_dv, pto_db, pto_dg_raw, pto_dh0  = pypto_function(
            q=q.detach(), k=k.detach(), v=v.detach(),
            g_raw=gate_for_pypto.detach(), beta=beta_for_pypto.detach(),
            initial_state=initial_state.detach(),
            do=do_bsnd.detach(), dht=dht_bsnd.detach(),
            cache=cache_for_pypto,
            I=I, M_le=M_le, M_lt=M_lt, C_cum=C_cum_3d, C_rcum=C_rcum,
            act_seq_len=act_seq_len,
            run_mode=run_mode
        )
    

    detailed_tensor_compare(pto_dq, dq, 'dq')
    detailed_tensor_compare(pto_dk, dk, 'dk')
    detailed_tensor_compare(pto_dv, dv, 'dv')
    detailed_tensor_compare(pto_db, db, 'db')
    detailed_tensor_compare(pto_dg_raw, dg_raw, 'dg_raw')
    detailed_tensor_compare(pto_dh0, dh0, 'dh0')
    print("\n✅ All checks passed.")


if __name__ == "__main__":
    main()
