import os
import time

import pypto
import torch
import torch_npu
import numpy as np
import pytest

import torch.nn.functional as F

def gen_dims(params):
    """Generate dimension parameters from input params."""
    dims = {}
    dims["T"] = params["T"]
    dims["B"] = params["B"]
    dims["Nqk"] = params["Nqk"]
    dims["Nv"] = params["Nv"]
    dims["D"] = 128
    dims["L"] = 128
    return dims


def gen_inputs(dims, dtype=torch.float32):
    """Generate input tensors for testing."""
    t = dims["T"]
    b = dims["B"]
    nqk = dims["Nqk"]
    nv = dims["Nv"]
    d = dims["D"]
    l = dims["L"]

    # Generate input data
    query = (torch.rand([t, nqk, d], dtype=dtype) * (1.3655 + 0.2785) - (1.3655 + 0.2785)) * 1.
    key = (torch.rand([t, nqk, d], dtype=dtype) * (1.4664 + 0.2785) - (1.4664 + 0.2785)) * 1.
    value = (torch.rand([t, nv, d], dtype=dtype) * (1.6488 + 0.2785) - (1.6488 + 0.2785)) * 1.
    beta = (torch.rand([t, nv], dtype=dtype) * (0.8927 - 0.0889) - (0.8927 - 0.0889)) * 1.
    gate = (torch.rand([t, nv], dtype=dtype) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)) * 1.
    states = torch.zeros([b, nv, d, d], dtype=dtype)

    # Generate act_seq_len based on B and T
    seq_len_per_batch = t // b
    act_seq_len = [i * seq_len_per_batch for i in range(b + 1)]
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32)

    # Helper tensors
    mask = torch.tril(-torch.ones([l, l], dtype=dtype), diagonal=-1)
    tril_mask = torch.ones([l, l], dtype=dtype).tril()
    eye = torch.eye(16, dtype=dtype).repeat(1, 8)

    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "gate": gate,
        "states": states,
        "act_seq_len": act_seq_len,
        "mask": mask,
        "tril_mask": tril_mask,
        "eye": eye,
    }


def compare(**kwargs):
    """Compare two tensors with tolerance."""
    actual = kwargs.get("actual")
    expected = kwargs.get("expected")
    name = kwargs.get("name")
    rtol = kwargs.get("rtol")
    atol_abs = kwargs.get("atol_abs")
    atol_rel = kwargs.get("atol_rel")
    print_diff_indices = kwargs.get("print_diff_indices", False)
    max_diff_print = kwargs.get("max_diff_print", 50)
    diff_epsilon = kwargs.get("diff_epsilon", 1e-7)  # 小于此视为 0，避免浮点噪声

    diff = torch.abs(actual.float() - expected.float())

    non_zero_mask = diff > diff_epsilon
    non_zero_count = non_zero_mask.sum().item()
    max_diff = torch.max(diff).item()
    min_diff = torch.min(diff).item()
    print(f"🔥 {name} comparing diff: {non_zero_count} non-zero elements, max diff: {max_diff}, min diff: {min_diff}")
    mean_diff = torch.mean(diff).item()
    print(f"🔥 {name} comparing mean diff: {mean_diff}")

    if print_diff_indices and non_zero_count > 0:
        flat_actual = actual.float().flatten()
        flat_expected = expected.float().flatten()
        flat_diff = diff.flatten()
        indices = torch.nonzero(non_zero_mask.flatten(), as_tuple=False).squeeze(-1)
        
        n_print = min(int(indices.numel()), max_diff_print)
        print(f"🔥 {name} first {n_print} diff indices (flat) [index] actual, expected, diff:")
        for i in range(n_print):
            idx = int(indices[i].item())
            print(f"    [{idx}] actual={flat_actual[idx].item():.8f}, expected={flat_expected[idx].item():.8f}, diff={flat_diff[idx].item():.8f}")
        if indices.numel() > max_diff_print:
            print(f"    ... and {indices.numel() - max_diff_print} more.")

    tolerance = atol_abs + atol_rel * torch.abs(expected.float())
    out_of_tolerance = (diff > tolerance).sum().item()
    total = actual.numel()

    if out_of_tolerance > 0:
        ratio = out_of_tolerance / total
        if ratio > 0.01:  # More than 1% out of tolerance
            raise AssertionError(f"{name} comparison failed: {out_of_tolerance}/{total} elements out of tolerance")
    print(f"✅ {name} comparison passed: {total - out_of_tolerance}/{total} elements within tolerance")


def debug_tensor_stats(name, tensor):
            if tensor is None:
                print(f"[GDN_BWD_DEBUG] {name}: None")
                return

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

            t = tensor.detach().to(torch.float32)
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            amax = t.abs().max().item()
            # 均值、方差
            mean = t.mean().item()
            std = t.std().item()
            print(
                f"[GDN_BWD_DEBUG][rank{rank}] {name}: "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"has_nan={has_nan}, has_inf={has_inf}, amax={amax}, mean={mean}, std={std}"
            )


### torch ###
def segs_chunk_gated_delta_rule_sub_inverse_differentiable(attn, chunk_size):
    """可微版本：不修改 attn，用 stack 构建新矩阵，保证梯度可回传到 attn。"""
    L = chunk_size
    device, dtype = attn.device, attn.dtype
    # 按行构造，每行只依赖原始 attn
    rows = []
    for i in range(L):
        if i == 0:
            row = attn[..., 0, :]  # [..., L]
        else:
            line = attn[..., i, :i]   # [..., i]
            sub = attn[..., :i, :i]   # [..., i, i]
            correction = (line.unsqueeze(-1) * sub).sum(dim=-2)  # [..., i]
            new_line = line + correction
            row = torch.cat([new_line, attn[..., i, i:i + 1], attn[..., i, i + 1:]], dim=-1)  # [..., L]
        rows.append(row)
    result = torch.stack(rows, dim=-2)  # [..., L, L]
    return result + torch.eye(L, dtype=dtype, device=device)


def segs_chunk_gated_delta_rule_sub_cycle_differentiable(**kwargs):
    """可微版本：用 list + torch.cat 替代对 attn_out 的 in-place 赋值，masked_fill 用非 in-place。"""
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    decay_mask = kwargs.get("decay_mask")
    k_cumdecay = kwargs.get("k_cumdecay")
    g = kwargs.get("g")
    last_recurrent_state = kwargs.get("last_recurrent_state")
    total_sequence_length = kwargs.get("total_sequence_length")
    chunk_size = kwargs.get("chunk_size")
    diff_G = kwargs.get("diff_G", None)
    return_attn_tmp = kwargs.get("return_attn_tmp", True)

    attn_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    output_list = []
    captured_attn = None
    for index in range(0, total_sequence_length // chunk_size):
        q_index = query[:, :, index]
        k_index = key[:, :, index]
        v_index = value[:, :, index]
        attn = (q_index @ k_index.transpose(-1, -2) * decay_mask[:, :, index]).masked_fill(attn_mask, 0)
        if return_attn_tmp:
            attn.retain_grad()
            captured_attn = attn  # 只保留最后一次（单 chunk 即 index=0）
        v_new = v_index - (k_cumdecay[:, :, index]) @ last_recurrent_state
        cur_out = (q_index * g[:, :, index, :, None].exp()) @ last_recurrent_state + attn @ v_new
        output_list.append(cur_out.unsqueeze(2))

        if diff_G is not None:
            kgexp_factor = (k_index * diff_G[:, :, index, :].exp()[..., None]).transpose(-1, -2) @ v_new
        else:
            kgexp_factor = (k_index * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]).transpose(-1, -2) @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, index, -1, None, None].exp()
            + kgexp_factor
        )

    attn_out = torch.cat(output_list, dim=2)
    return attn_out, last_recurrent_state, captured_attn


def segs_chunk_gated_delta_rule_sub_differentiable(**kwargs):
    """PyTorch 可微参考实现：仅将 sub_inverse 与 sub_cycle 换为可微版本，其余与原 sub 一致。"""
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    g = kwargs.get("g")
    beta = kwargs.get("beta")
    chunk_size = kwargs.get("chunk_size")
    initial_state = kwargs.get("initial_state")
    output_final_state = kwargs.get("output_final_state")
    use_qk_l2norm_in_kernel = kwargs.get("use_qk_l2norm_in_kernel")
    return_intermediates = kwargs.get("return_intermediates", True)

    b, n, s, d = value.shape
    initial_state = initial_state.transpose(3, 2)
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, pad_size)) for x in (beta, g)]

    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    gate_before_cum = g

    g_after_cum = gate_before_cum.cumsum(dim=-1)
    # 显式写成 gate_cum = tril @ gate_before_cum，与 PyPTO 一致，反向即 d_gate_before_cum = tril.T @ d_gate_cum
    # tril = torch.tril(torch.ones(chunk_size, chunk_size, dtype=g.dtype, device=g.device))
    # g_after_cum = torch.einsum('ij,...j->...i', tril, gate_before_cum)
    if return_intermediates:
        gate_before_cum.retain_grad()
        g_after_cum.retain_grad()
    decay_mask = ((g_after_cum.unsqueeze(-1) - g_after_cum.unsqueeze(-2)).tril().exp().float()).tril()
    if return_intermediates:
        
        decay_mask.retain_grad()
    

    # Path B: 保留「乘 decay、mask 后、sub_inverse 前」的 attn，对应 PyPTO 的 attn（d_x = d_attn * attn）
    attn_before_inverse = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    attn = segs_chunk_gated_delta_rule_sub_inverse_differentiable(attn_before_inverse, chunk_size)

    value = attn @ v_beta
    g_exp = g_after_cum.exp()
    if return_intermediates:
        g_exp.retain_grad()
    # Path C 中间量: w = key_beta * gate_exp（对应 PyPTO d_gate_exp_C）
    w_pathC = k_beta * g_exp.unsqueeze(-1)
    if return_intermediates:
        w_pathC.retain_grad()
    k_cumdecay = attn @ w_pathC
   
    if return_intermediates:
        k_cumdecay.retain_grad()
    diff_G = None
    
    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device).to(value)
    else:
        last_recurrent_state = initial_state.to(value)
    assert k_cumdecay.requires_grad, "k_cumdecay should require grad"

    if return_intermediates:
        attn_out, last_recurrent_state, attn_tmp_subcycle = segs_chunk_gated_delta_rule_sub_cycle_differentiable(
        query=query, key=key, value=value,
        decay_mask=decay_mask,
        k_cumdecay=k_cumdecay, g=g_after_cum, last_recurrent_state=last_recurrent_state,
        total_sequence_length=total_sequence_length, chunk_size=chunk_size,
        diff_G=diff_G if return_intermediates else None,
        return_attn_tmp=True,
    )
    else:
        attn_out, last_recurrent_state, _ = segs_chunk_gated_delta_rule_sub_cycle_differentiable(
            query=query, key=key, value=value,
            decay_mask=decay_mask, k_cumdecay=k_cumdecay, g=g_after_cum, last_recurrent_state=last_recurrent_state,
            total_sequence_length=total_sequence_length, chunk_size=chunk_size,
            diff_G=diff_G if return_intermediates else None,
        )

    if not output_final_state:
        last_recurrent_state = None
    attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1, attn_out.shape[-1])
    attn_out = attn_out[:, :, :sequence_length].transpose(1, 2).contiguous()
    last_recurrent_state = last_recurrent_state.transpose(3, 2) if last_recurrent_state is not None else None
    if return_intermediates:
        return attn_out, last_recurrent_state, {
            "gate (cumsum 前输入)": gate_before_cum,   # 新增：看 d(gate)
            "gate_cum (总)": g_after_cum,
            "decay_mask (path B 总)": decay_mask,
            "g_exp (path A+C, D 最后行)": g_exp,
            "w_pathC (path C 中间 w=k_beta*g_exp)": w_pathC,
            "diff_G (path G diff=g_last-g)": diff_G,
            "k_cumdecay": k_cumdecay,
            "attn_tmp_subcycle": attn_tmp_subcycle,
        }
    return attn_out, last_recurrent_state, None


def segs_chunk_gated_delta_rule_differentiable(**kwargs):
    """Segmented chunk gated delta rule for batch processing."""
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    g = kwargs.get("gate")
    beta = kwargs.get("beta")
    act_seq_len = kwargs.get("act_seq_len")
    chunk_size = kwargs.get("chunk_size")
    initial_state = kwargs.get("initial_state")
    output_final_state = kwargs.get("output_final_state")
    use_qk_l2norm_in_kernel = kwargs.get("use_qk_l2norm_in_kernel")
    return_intermediates = kwargs.get("return_intermediates", True)

    t, n1, d = query.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1

    query = query.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)

    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)

    query, key, value, beta, g = \
        [x.transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)

    intermediates = None
    for b_idx in range(batch):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        seg_s = 128
        pad_size = (chunk_size - s % chunk_size) % chunk_size
        pad_seq_length = s + pad_size
        batch_query, batch_key, batch_value = \
            [F.pad(x[:, b_ofs:b_ofs + s], (0, 0, 0, pad_size)) for x in (query, key, value)]
        batch_beta, batch_g = [F.pad(x[:, b_ofs:b_ofs + s], (0, pad_size)) for x in (beta, g)]
        result_list = []
        recurrent_state = initial_state[b_idx:b_idx + 1, ...]
        for s_idx in range(0, pad_seq_length, seg_s):
            chunk_query, chunk_key, chunk_value = \
                [x[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d) for x in (batch_query, batch_key, batch_value)]
            chunk_gate, chunk_beta = [x[:, s_idx:s_idx + seg_s].reshape(1, n, seg_s) for x in (batch_g, batch_beta)]
            want_chunk_intermediates = return_intermediates and (s_idx == 0)
            if want_chunk_intermediates:
                cur_attn, cur_state, chunk_intermediates = segs_chunk_gated_delta_rule_sub_differentiable(
                    query=chunk_query, key=chunk_key, value=chunk_value,
                    g=chunk_gate, beta=chunk_beta, chunk_size=chunk_size, initial_state=recurrent_state,
                    output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    return_intermediates=True,
                )
                intermediates = chunk_intermediates
            else:
                cur_attn, cur_state, _ = segs_chunk_gated_delta_rule_sub_differentiable(
                    query=chunk_query, key=chunk_key, value=chunk_value,
                    g=chunk_gate, beta=chunk_beta, chunk_size=chunk_size, initial_state=recurrent_state,
                    output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    return_intermediates=False,
                )
            result_list.append(cur_attn.squeeze(0))
            recurrent_state = cur_state
        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs:b_ofs + s] = batch_attn
        final_state[b_idx:b_idx + 1, ...] = recurrent_state
    if return_intermediates:
        return final_attn, final_state, intermediates
    return final_attn, final_state, None


### pypto ###

def pre_attn_v2(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum, decay_mask, beta_k and kkt.

    Parameters
    ---------
    gate: [L, 1]
    key: [L, D]
    beta: [L, 1]
    tril: [L, L]
    mask: [L, L]

    Return
    ---------
    gate_cum: [L, 1]
    decay_mask: [L, L]
    A: [L, L]
    key_beta: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # cal_cumsum
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # [L,1]
    # cal_decay_mask
    decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()  # [L,L]
    # beta_k
    key_beta = key_view_2d * beta_view  # [L,D]
    # kkt
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)  # [L,L]
    a = kkt * decay_mask * mask  # [L,L]

    return gate_cum, decay_mask, a, key_beta


def inverse_pto_v2(attn: pypto.Tensor, eye: pypto.Tensor, size: int) -> pypto.Tensor:
    """
    Calculate inverse of big matrix.

    Parameters
    ---------
    attn: [L, L]
    eye: [L, L]
    size: matrix size

    Return
    ---------
    attn_inv: [L, L]
    """
    min_length = size // 8
    pypto.set_vec_tile_shapes(128, 128)

    attn_8_8_list = []
    for i in range(8):
        attn_8_8_list.append(attn.view([min_length, min_length], [min_length * i, min_length * i]) + 0.0)
    attn_tmp_dim0 = pypto.concat(attn_8_8_list, dim=0)
    attn_tmp_dim1 = pypto.concat(attn_8_8_list, dim=1)

    attn_tmp_dim1_inv = inverse_pto_min_length_v2(attn_tmp_dim0, attn_tmp_dim1, eye, min_length, min_length * 8)

    attn_8_8_inv_list = []
    for i in range(8):
        attn_8_8_inv_list.append(attn_tmp_dim1_inv[:, min_length * i:min_length * (i + 1)] + 0.0)

    attn_4_inv_list = []
    for i in range(4):
        attn_4_inv_list.append(inverse_matmul_v2(attn=attn, attn_1_1_inv=attn_8_8_inv_list[i * 2],
            attn_2_2_inv=attn_8_8_inv_list[i * 2 + 1], x_ofs=min_length * i * 2, y_ofs=min_length * i * 2,
            m_len=min_length))

    attn_2_inv_list = []
    for i in range(2):
        attn_2_inv_list.append(inverse_matmul_v2(attn=attn, attn_1_1_inv=attn_4_inv_list[i * 2],
            attn_2_2_inv=attn_4_inv_list[i * 2 + 1], x_ofs=min_length * i * 4, y_ofs=min_length * i * 4,
            m_len=min_length * 2))

    attn_inv = inverse_matmul_v2(attn=attn, attn_1_1_inv=attn_2_inv_list[0],
        attn_2_2_inv=attn_2_inv_list[1], x_ofs=0, y_ofs=0, m_len=min_length * 4)
    return attn_inv


def inverse_pto_min_length_v2(
    attn_dim0: pypto.Tensor,
    attn_dim1: pypto.Tensor,
    eye: pypto.Tensor,
    row_num: int,
    col_num: int,
) -> pypto.Tensor:
    """
    Calculate inverse of matrix with tail concat optimization.

    Parameters
    ---------
    attn_dim0: [L, L // 8]
    attn_dim1: [L // 8, L]
    eye: [L, L]
    row_num: L // 8
    col_num: L

    Return
    ---------
    res: [L, L]
    """
    size = col_num // row_num  # L // (L // 8) = 8

    attn_inv_list = {}
    attn_inv_list[1] = attn_dim1[:2, :]  # [2, L]
    pypto.set_vec_tile_shapes(128, 128)

    attn_dim0_trans = attn_dim0.transpose(0, 1).reshape([col_num, row_num])

    for i in range(2, row_num, 1):
        # Add 0.0 to enable attn_inv_cur to enter the UB in advance
        attn_inv_cur = attn_inv_list.get(i - 1) + 0.0
        row = attn_dim1.view([1, col_num], [i, 0])
        row_expand = attn_dim0_trans.view([size * i, 1], [0, i])  # [size * i, 1]
        attn_inv_cur_reshape = attn_inv_cur.reshape([size * i, row_num])  # [size * i, row_num] = [size * i, L // 8] = [8 * i, L // 8]
        prod_mul = (row_expand * attn_inv_cur_reshape).reshape([i, col_num])  # [i, col_num] = [i, L]

        prod = prod_mul.sum(0, keepdim=True)
        attn_update = row + prod

        attn_inv_list[i] = pypto.concat([attn_inv_cur, attn_update], dim=0)

    res = attn_inv_list.get(row_num - 1) + eye

    return res


def inverse_matmul_v2(**kwargs) -> pypto.Tensor:
    """
    Calculate inverse of small matrix.

    Parameters
    ---------
    attn: [L, L]
    attn_1_1_inv: attn upper left matrix
    attn_2_2_inv: attn bottom right matrix
    x_ofs: row offset
    y_ofs: column offset
    len: matrix length

    Return
    ---------
    attn_inv: [len * 2, len * 2]
    """
    attn = kwargs.get("attn")
    attn_1_1_inv = kwargs.get("attn_1_1_inv")
    attn_2_2_inv = kwargs.get("attn_2_2_inv")
    x_ofs = kwargs.get("x_ofs")
    y_ofs = kwargs.get("y_ofs")
    m_len = kwargs.get("m_len")

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    attn_2_1 = attn.view([m_len, m_len], [x_ofs + m_len, y_ofs])

    attn_2_1_inv = (attn_2_2_inv @ attn_2_1) @ attn_1_1_inv

    attn_inv = pypto.tensor([m_len * 2, m_len * 2], dtype=attn_1_1_inv.dtype)
    attn_inv[0:m_len, 0:m_len] = attn_1_1_inv
    attn_inv[m_len:m_len * 2, 0:m_len] = attn_2_1_inv
    attn_inv[m_len:m_len * 2, m_len:m_len * 2] = attn_2_2_inv

    return attn_inv


def cal_value_and_key_cumdecay_v2(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate value and k cumdecay.

    Parameters
    ---------
    attn: [L, L]
    value_view: [L, D]
    beta_view: [L, D]
    key_beta: [L, D]
    gate_cum: [L, 1]

    Return
    ---------
    value_out: [L, D]
    key_cum_out: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # value_out
    value_beta_view = value_view * beta_view  # [L, D]
    value_out = pypto.matmul(attn, value_beta_view, pypto.DT_FP32)  # [L, D]
    # k_cumdecay_out
    g_exp = pypto.exp(gate_cum)  # [L, 1]
    weighted_k_beta_view = key_beta * g_exp  # [L, D]
    key_cum_out = pypto.matmul(attn, weighted_k_beta_view, pypto.DT_FP32)  # [L, D]

    return value_out, key_cum_out


def recurrent_state_attn_all_v2(**kwargs) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate attention.

    Parameters
    ---------
    query: [L, D]
    key: [L, D]
    value:[L, Dv]
    k_cumdecay:[L, Dk]
    gate: [L, 1]
    state: [D, D]
    decay_mask: [L, L]
    tril: [L, L]

    Return
    ---------
    chunk_attn_out: [L, D]
    state_new:[Dv, Dk]
    """
    query = kwargs.get("query")
    key = kwargs.get("key")
    value = kwargs.get("value")
    k_cumdecay = kwargs.get("k_cumdecay")
    gate = kwargs.get("gate")
    state = kwargs.get("state")
    decay_mask = kwargs.get("decay_mask")
    tril = kwargs.get("tril")

    dv = value.shape[-1]
    l = gate.shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)
    _last_gate_1 = gate[l - 1:l, :]
    kgexp = key * (_last_gate_1 - gate).exp()  # [L, Dk]
    qgexp = query * gate_exp
    # pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    # pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [Dv, Dk]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_value = pypto.matmul(value, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [L, Dk]
    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, L] = [L, L]
    _last_gate_2 = pypto.expand_clone(gate_exp[l - 1:l, :], (dv, 1))  # [Dv, 1]
    final_state_1 = state * _last_gate_2
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    attn_tmp = attn * decay_mask * tril  # [L, L]
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    # pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new


@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 32,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 32,
    },
)
def chunk_gated_delta_rule_fwd_clear_wo_l2norm(*args):
    """
    Chunk Gated Delta Rule fused operator.

    This is the main entry point for the Gated Delta Rule attention computation.
    It processes input sequences in chunks of size L=128, maintaining recurrent
    state across chunks for efficient long sequence modeling.

    Parameters
    ----------
    query : Input query tensor, shape [T, Nqk, D], dtype float32 after l2norm
    key : Input key tensor, shape [T, Nqk, D], dtype float32 after l2norm
    value : Input value tensor, shape [T, Nv, D], dtype float32
    beta : Beta scaling factor, shape [T, Nv], dtype float32
    gate : Gate signal, shape [T, Nv], dtype float32
    states : Initial recurrent states, shape [B, Nv, D, D], dtype float32
    mask : Attention mask (lower triangular negative), shape [L, L], dtype float32
    tril_mask : Lower triangular mask, shape [L, L], dtype float32
    eye : Identity matrix (specially processed), shape [16, 128], dtype float32
    act_seq_len : Cumulative sequence length indices, shape [B+1], dtype int32
    core_attn_out : Output attention tensor, shape [T, Nv, D], dtype float32
    last_state_data : Output updated states, shape [B, Nv, D, D], dtype float32
    query_norm_res : Output query normalization tensor, shape [T, Nqk, D], dtype float32
    key_norm_res : Output key normalization tensor, shape [T, Nqk, D], dtype float32
    inverse_a_block_res : Output inverse A block tensor, shape [B, Nv, max_chunks, L, L], dtype float32
    all_states_res : Output all chunks state tensor, shape [B, Nv, max_chunks + 1, D, D], dtype float32
    """
    query = args[0]
    key = args[1]
    value = args[2]
    beta = args[3]
    gate = args[4]
    states = args[5]
    mask = args[6]
    tril_mask = args[7]
    eye = args[8]
    act_seq_len = args[9]
    core_attn_out = args[10]
    last_state_data = args[11]

    inverse_a_block_res = args[12]
    all_states_res = args[13]
    _, nqk, d = query.shape
    _, nv, d = value.shape
    b = states.shape[0]
    l, l = mask.shape
    group = nv // nqk
    for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(16, 16, 128, 128)
            last_state = states[b_idx, nv_idx]

            all_states_res[b_idx, nv_idx, 0] = states[b_idx, nv_idx]
            for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx"):
                bs_ofs = b_ofs + s_idx
                actual_l = (s - s_idx).min(l)
                chunk_idx = s_idx // l
                ## view
                query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])

                pypto.set_vec_tile_shapes(128, 128, 128)
                query_norm = pypto.reshape(query_view, [l, d], valid_shape=[actual_l, d])
                key_norm = pypto.reshape(key_view, [l, d], valid_shape=[actual_l, d])
                value_view_2d = pypto.reshape(value_view, [l, d], valid_shape=[actual_l, d])

                # query_norm, key_norm = l2norm_v2(query_view_2d, key_view_2d)
                # if nv_idx % group == 0:
                    # query_norm_res[bs_ofs:bs_ofs + l, nqk_idx:nqk_idx + 1, :] = pypto.reshape(query_norm, [l, 1, d], valid_shape=[actual_l, 1, d])  # 存储中间结果 用于 bwd 计算
                    # key_norm_res[bs_ofs:bs_ofs + l, nqk_idx:nqk_idx + 1, :] = pypto.reshape(key_norm, [l, 1, d], valid_shape=[actual_l, 1, d])

                scale = 1 / d**0.5
                query_scale = query_norm * scale

                # kv_beta & g_cumsum & decay_mask & pre_attn
                gate_cum, decay_mask, a_block, key_beta = pre_attn_v2(gate_view, key_norm, beta_view, tril_mask, mask)
                # gate_cum_res[bs_ofs:bs_ofs + l, nv_idx:nv_idx + 1] = gate_cum  # 存储中间结果 用于 bwd 计算
                # inverse
                a_block_inverse = inverse_pto_v2(a_block, eye, 128)
                inverse_a_block_res[b_idx:b_idx + 1, nv_idx:nv_idx + 1, chunk_idx:chunk_idx + 1, :, :] = pypto.reshape(a_block_inverse, [1, 1, 1, l, l])     # 存储中间结果 用于 bwd 计算
                # cal_value_and_keycumdecay
                pypto.set_vec_tile_shapes(128, 128, 128)
                value_out, key_cum_out = cal_value_and_key_cumdecay_v2(a_block_inverse, value_view_2d,
                    beta_view, key_beta, gate_cum)

                chunk_attn_out, cur_state = recurrent_state_attn_all_v2(query=query_scale, key=key_norm, value=value_out,
                    k_cumdecay=key_cum_out, gate=gate_cum, state=last_state, decay_mask=decay_mask, tril=tril_mask)

                # assemble
                pypto.set_vec_tile_shapes(16, 16, 128, 128)

                last_state[:] = cur_state
                core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out
                last_state_data[b_idx, nv_idx] = last_state
                all_states_res[b_idx, nv_idx, chunk_idx + 1] = last_state


def pypto_chunk_gated_delta_rule_fwd_dyn_clear_wo_l2norm(inputs: dict, outputs: dict):
    """调用 chunk_gated_delta_rule_fwd_clear_wo_l2norm core_attn_out、last_state_data 及 bwd 所需中间结果。"""
    L = inputs["mask"].shape[0]

    input_tensors = [
        (inputs["query"], [0]),  # after l2norm
        (inputs["key"], [0]),  # after l2norm
        (inputs["value"], [0]),
        (inputs["beta"], [0]),
        (inputs["gate"], [0]),
        (inputs["states"], []),
        (inputs["mask"], []),
        (inputs["tril_mask"], []),
        (inputs["eye"], []),
        (inputs["act_seq_len"], []),
    ]
    output_tensors = [
        (outputs["core_attn_out"], [0]),
        (outputs["final_state"], []),
        (outputs["inverse_a_block_res"], []),
        (outputs["all_states_res"], []),
    ]
    pto_inputs = [pypto.from_torch(t, dynamic_axis=ax) for t, ax in input_tensors]
    pto_outputs = [pypto.from_torch(t, dynamic_axis=ax) for t, ax in output_tensors]
    chunk_gated_delta_rule_fwd_clear_wo_l2norm(*pto_inputs, *pto_outputs)
    torch_npu.npu.synchronize()


def _recompute_pre_attn_v3(gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum, decay_mask, beta_k and kkt.

    Parameters
    ---------
    gate: [L, 1]
    key: [L, D]
    beta: [L, 1]
    tril: [L, L]
    mask: [L, L]

    Return
    ---------
    gate_cum: [L, 1]
    decay_mask: [L, L]
    key_beta: [L, D]
    """
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # [L,1]
    decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()  # [L,L]

    key_beta = key_view_2d * beta_view  # [L,D]
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)  # [L,L]

    return gate_cum, decay_mask, key_beta, kkt


# ---------------------------------------------------------------------------
# 复算 cal_value_and_key_cumdecay 的产出
# ---------------------------------------------------------------------------
def _recompute_value_key_cumdecay(a_block_inverse, beta_view, key_beta, gate_cum):
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    g_exp = pypto.exp(gate_cum)  # [L, 1]

    weighted_k_beta_view = key_beta * g_exp  # [L, D]
    key_cum_out = pypto.matmul(a_block_inverse, weighted_k_beta_view, pypto.DT_FP32)

    return key_cum_out


# ---------------------------------------------------------------------------
# 同时计算 d_value、d_query_norm 与 d_key_norm 、d_gate 、d_beta 的反向
# 每个 chunk 的「起始 state」= initial_state（c=0）或 all_states_res[b,nv,c-1]（c>0）
# ---------------------------------------------------------------------------
@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 32,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 32,
    },
)
def chunk_gated_delta_rule_backward_all_clear_test(*args, act_seq_len_int: int = 1024):
    """
    Backward of chunk_gated_delta_rule.
    前向需保存并传入的量: q_norm, k_norm, v, g, beta, A, initial_state, mask, tril_mask, act_seq_len, all_states_res。

    Parameters (args)
    ----------
    do              : [T, Nv, D]   core_attn_out 的梯度
    d_last_state    : [B, Nv, D, D]
    q_norm          : [T, Nqk, D]  query_norm
    k_norm          : [T, Nqk, D]  key_norm
    all_states_res  : [B, Nv, max_chunks + 1, D, D]
    v               : [T, Nv, D]   value
    g               : [T, Nv]      gate only not gate_cum
    beta            : [T, Nv]
    A               : [B, Nv, max_chunks, L, L]  a_block_inverse
    mask            : [L, L]
    tril_mask       : [L, L]
    act_seq_len     : [B+1] int32
    d_value         : [T, Nv, D] 输出
    d_query_norm    : [T, Nqk, D] 输出
    d_key_norm      : [T, Nqk, D] 输出
    d_gate          : [T, Nv]
    d_beta          : [T, Nv]
    """
    do = args[0]
    d_last_state = args[1]
    q_norm = args[2]
    k_norm = args[3]
    all_states_res = args[4]
    v = args[5]
    g = args[6]
    beta = args[7]
    A = args[8]
    mask = args[9]
    tril_mask = args[10]
    act_seq_len = args[11]
    d_value = args[12]
    d_query_norm = args[13]
    d_key_norm = args[14]
    d_gate = args[15]
    d_beta = args[16]

    # ---------- 变量初始化（shape 与前向一致） ----------
    _, nqk, d = q_norm.shape
    nv = v.shape[1]

    assert nqk == nv, "nqk must be equal to nv"
    b = A.shape[0]

    l = mask.shape[0]

    scale = 1.0 / (d ** 0.5)

    num_chunks_int = act_seq_len_int // l

    for b_idx in pypto.loop(b, name="LOOP_B_BWD", idx_name="b_idx"):

        for nv_idx in pypto.loop(nv, name="LOOP_Nv_BWD", idx_name="nv_idx"):
            pypto.set_vec_tile_shapes(128, 128)
            zeros = pypto.full([d, d], 0.0, dtype=pypto.DT_FP32)

            d_state_next = zeros
            def cal_d_value(c_pass2_idx, d_state_next):
                bs_ofs = b_idx * act_seq_len_int + (num_chunks_int - 1 - c_pass2_idx) * l
                actual_l = l

                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                q_view = pypto.view(q_norm, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                k_view = pypto.view(k_norm, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                v_view = pypto.view(v, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                g_view = pypto.view(g, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                d_chunk_out = pypto.view(do, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])

                pypto.set_vec_tile_shapes(1, 1, 1, 128, 128)
                a_chunk = pypto.reshape(
                    pypto.view(A, [1, 1, 1, l, l], [b_idx, nv_idx, num_chunks_int - 1 - c_pass2_idx, 0, 0], valid_shape=[1, 1, 1, l, l]),
                    [l, l]
                )
                state_chunk = pypto.reshape(
                    pypto.view(all_states_res, [1, 1, 1, d, d], [b_idx, nv_idx, num_chunks_int - 1 - c_pass2_idx, 0, 0], valid_shape=[1, 1, 1, d, d]),
                    [d, d]
                )

                pypto.set_vec_tile_shapes(128, 128, 128)
                query_view_2d = pypto.reshape(q_view, [l, d], valid_shape=[actual_l, d])
                key_view_2d = pypto.reshape(k_view, [l, d], valid_shape=[actual_l, d])
                value_view_2d = pypto.reshape(v_view, [l, d], valid_shape=[actual_l, d])
                d_chunk_out_2d = pypto.reshape(d_chunk_out, [l, d], valid_shape=[actual_l, d])

                gate_cum, decay_mask, key_beta, kkt = _recompute_pre_attn_v3(g_view, key_view_2d, beta_view, tril_mask)
                key_cum_out = _recompute_value_key_cumdecay(a_chunk, beta_view, key_beta, gate_cum)
                query_scale = pypto.mul(query_view_2d, scale)
                gate_exp = gate_cum.exp()
                _last_gate_1 = gate_cum[actual_l - 1 : actual_l, :]
                _last_gate_2 = pypto.expand_clone(gate_exp[actual_l - 1 : actual_l, :], (d, 1))
                kgexp = key_view_2d * (_last_gate_1 - gate_cum).exp()
                qgexp = pypto.mul(query_scale, gate_exp)

                value_beta = value_view_2d * beta_view
                pypto.set_vec_tile_shapes(128, 128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                value_out = pypto.matmul(a_chunk, value_beta, pypto.DT_FP32)
                v_prime = pypto.matmul(key_cum_out, state_chunk, pypto.DT_FP32, b_trans=True)

                attn = pypto.matmul(query_scale, key_view_2d, pypto.DT_FP32, b_trans=True)
                attn_tmp = attn * decay_mask * tril_mask

                # ========== d_value ==========
                pypto.set_vec_tile_shapes(128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                d_value_out_from_state = pypto.matmul(kgexp, d_state_next, pypto.DT_FP32, b_trans=True)
                
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                d_value_out_from_attn = pypto.matmul(attn_tmp, d_chunk_out_2d, pypto.DT_FP32, a_trans=True)

                d_value_out = d_value_out_from_attn + d_value_out_from_state
                d_value_beta = pypto.matmul(a_chunk, d_value_out, pypto.DT_FP32, a_trans=True)
                d_value_chunk = d_value_beta * beta_view
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                d_value[bs_ofs : bs_ofs + l, nv_idx] = d_value_chunk


                # ========== d_query_norm（Step 1～8） ==========
                pypto.set_vec_tile_shapes(128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

                d_attn_inter = d_chunk_out_2d
                d_chunk_attn_value = d_chunk_out_2d
                d_chunk_attn_vprime = pypto.neg(d_chunk_out_2d)

                d_attn_tmp_from_value_out = pypto.matmul(d_chunk_attn_value, value_out, pypto.DT_FP32, b_trans=True)
                d_attn_tmp_from_v_prime = pypto.matmul(d_chunk_attn_vprime, v_prime, pypto.DT_FP32, b_trans=True)
                d_attn_tmp_total = d_attn_tmp_from_value_out + d_attn_tmp_from_v_prime
                d_attn = d_attn_tmp_total * decay_mask * tril_mask

                d_query_scale = pypto.matmul(d_attn, key_view_2d, pypto.DT_FP32)
                d_qgexp = pypto.matmul(d_attn_inter, state_chunk, pypto.DT_FP32)
                d_query_scale = d_query_scale + d_qgexp * gate_exp
                d_query_norm_chunk = d_query_scale * scale
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                d_query_norm[bs_ofs : bs_ofs + l, nv_idx] = d_query_norm_chunk


                # ========== d_key_norm ==========
                pypto.set_vec_tile_shapes(128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                # 路径 D（v_prime 支）先算，再与路径 B 的 d_v_prime 合并为 d_v_prime_total，供路径 B 与 d_state_prev 使用。
                # --- Path D: d_state_next -> temp_matmul_vprime -> v_prime ---
                # 前向: cur_state 含 - temp_matmul_vprime; temp_matmul_vprime = v_prime^T @ kgexp. 形状 v_prime^T [D,L], kgexp [L,D] => [D,D]
                # 反向: d_temp_matmul_vprime = - d_state_next. 形状 [D,D]
                # 反向: d_v_prime = kgexp @ d_temp_matmul_vprime^T = kgexp @ (-d_state_next)^T = - kgexp @ d_state_next^T. 形状 [L,D]@[D,D]=[L,D]
                # d_v_prime_from_state = pypto.matmul(kgexp, d_state_next, pypto.DT_FP32, b_trans=True) * (-1.0)
                d_v_prime_from_state = d_value_out_from_state * (-1.0)
                # 形状: d_v_prime_from_state [L,D]

                # --- Path B: d_chunk_attn_out -> chunk_attn_vprime -> v_prime ---
                # 前向: chunk_attn_vprime = attn_tmp @ v_prime. 形状 [L,L]@[L,D]=[L,D]
                # 反向: d_v_prime_from_attn = attn_tmp^T @ d_chunk_attn_vprime. 形状 [L,L]^T@[L,D]=[L,D]
                d_v_prime_from_attn = pypto.matmul(attn_tmp, d_chunk_attn_vprime, pypto.DT_FP32, a_trans=True)
                # 形状: d_v_prime_from_attn [L,D]

                # 合并，路径 B 与 d_state_prev 均用 d_v_prime_total
                # d_v_prime_total = d_v_prime_from_attn + d_v_prime_from_state. 形状 [L,D]
                d_v_prime_total = d_v_prime_from_state + d_v_prime_from_attn
                # d_v_prime_from_attn + d_v_prime_from_state


                # ---------- 路径 B: d_v_prime_total -> key_cum_out -> (w/key_beta/kkt) -> key_norm ----------
                # Step B2: 前向 v_prime = key_cum_out @ state_chunk^T. 形状 [L,D]@[D,D]=[L,D]
                # 反向 d_key_cum_out_total = d_v_prime_total @ state_chunk. 形状 [L,D]@[D,D]=[L,D]
                d_key_cum_out_total = pypto.matmul(d_v_prime_total, state_chunk, pypto.DT_FP32)

                # Step B3: 前向 key_cum_out = a_chunk @ w, w = key_beta * gate_exp. 形状 a_chunk [L,L], w [L,D]
                # 反向 d_w_from_cum = a_chunk^T @ d_key_cum_out_total. 形状 [L,L]^T@[L,D]=[L,D]
                d_w_from_cum = pypto.matmul(a_chunk, d_key_cum_out_total, pypto.DT_FP32, a_trans=True)

                # Step B4: 前向 w = key_beta * gate_exp. 反向 d_key_beta_from_w = d_w_from_cum * gate_exp. 形状 [L,D]*[L,1]=[L,D]
                d_key_beta_from_w = d_w_from_cum * gate_exp

                # Step B5: 前向 key_beta = key_norm * beta_view. 反向 d_key_norm_pathB_from_key_beta = d_key_beta_from_w * beta_view. 形状 [L,D]
                # （此支与 Step B10 合并为 d_key_beta_pathB_total * beta_view，见后）

                # Step B6: 前向 key_cum_out = A_inv @ w. 反向 d(A_inv) = d_key_cum_out_total @ w^T. 形状 [L,D]@[D,L]=[L,L]
                w = key_beta * gate_exp  # [L,D]
                d_a_block_inv_from_cum = pypto.matmul(d_key_cum_out_total, w, pypto.DT_FP32, b_trans=True)

                # Step B6b: 前向 value_out = A_inv @ value_beta. 反向 d(A_inv) += d_value_out @ value_beta^T. 形状 [L,D]@[D,L]=[L,L]
                d_a_block_inv_from_value = pypto.matmul(d_value_out, value_beta, pypto.DT_FP32, b_trans=True)

                d_a_block_inv_total = d_a_block_inv_from_cum + d_a_block_inv_from_value

                # Step B7: 逆微分 d_a_block = - A_inv^T @ d(A_inv) @ A_inv^T. 形状 [L,L]
                
                d_a_block_from_inv_1 = pypto.matmul(a_chunk, d_a_block_inv_total, pypto.DT_FP32, a_trans=True)
                # d_a_block_from_inv_tmp = pypto.cast(d_a_block_from_inv_1, pypto.DT_FP32) # pypto.neg(
                d_a_block_from_inv = pypto.matmul(d_a_block_from_inv_1, a_chunk, pypto.DT_FP32, b_trans=True)

                # Step B8: 前向 a_block = kkt * decay_mask * mask. 反向 d_kkt_from_a = d_a_block_from_inv * decay_mask * mask. 形状 [L,L]
                d_kkt_from_a = d_a_block_from_inv * decay_mask * mask
                # d_kkt_from_a = pypto.cast(d_kkt_from_a_tmp, pypto.DT_FP32)

                # Step B9: 前向 kkt = key_beta @ key_norm^T. 形状 [L,D]@[D,L]=[L,L]
                # 反向 d_key_norm_pathB_from_kkt = d_kkt_from_a^T @ key_beta. 形状 [L,L]^T@[L,D]=[L,D]
                # 反向 d_key_beta_from_kkt = d_kkt_from_a @ key_norm. 形状 [L,L]@[L,D]=[L,D]
                d_key_norm_pathB_from_kkt = pypto.matmul(d_kkt_from_a, key_beta, pypto.DT_FP32, a_trans=True)
                d_key_beta_from_kkt = pypto.matmul(d_kkt_from_a, key_view_2d, pypto.DT_FP32)

                # Step B10: 路径 B 经 key_beta 合并. 前向 key_beta = key_norm * beta_view.
                # d_key_beta_pathB_total = d_key_beta_from_w + d_key_beta_from_kkt. 形状 [L,D]
                # 反向 d_key_norm_pathB_via_beta = d_key_beta_pathB_total * beta_view. 形状 [L,D]
                d_key_beta_pathB_total = d_key_beta_from_w + d_key_beta_from_kkt
                d_key_norm_pathB_via_beta = d_key_beta_pathB_total * beta_view

                # ---------- 路径 A: d_chunk_attn_out -> chunk_attn_value -> attn_tmp -> attn -> key_norm ----------
                # 前向: key_norm -> attn = query_scale @ key_norm^T -> attn_tmp -> chunk_attn_value -> chunk_attn_out
                # Step A1: 前向 chunk_attn_value = attn_tmp @ value_out. 反向 d_attn_tmp_pathA = d_chunk_attn_value @ value_out^T. 形状 [L,D]@[D,L]=[L,L]
                d_attn_tmp_pathA = pypto.matmul(d_chunk_attn_value, value_out, pypto.DT_FP32, b_trans=True)
                # Step A2: 前向 attn_tmp = attn * decay_mask * tril_mask. 反向 d_attn_pathA = d_attn_tmp_pathA * decay_mask * tril_mask. 形状 [L,L]
                d_attn_pathA = d_attn_tmp_pathA * decay_mask * tril_mask

                # Step A3: 前向 attn = query_scale @ key_norm^T. 反向 d_key_norm_pathA = d_attn_pathA^T @ query_scale. 形状 [L,L]^T@[L,D]=[L,D]
                d_key_norm_pathA = pypto.matmul(d_attn_pathA, query_scale, pypto.DT_FP32, a_trans=True)

                # ---------- 路径 C: d_state_next -> temp_matmul_value -> kgexp -> key_norm ----------
                # Step C1: 前向 cur_state 含 + temp_matmul_value = value_out^T @ kgexp. 反向 d_temp_matmul_value = d_state_next. 形状 [D,D]
                # Step C2: 反向 d_kgexp_from_value = value_out @ d_state_next^T. 形状 [L,D]@[D,D]=[L,D]
                d_kgexp_from_value = pypto.matmul(value_out, d_state_next, pypto.DT_FP32, b_trans=True)
                # # Step C3: 前向 kgexp = key_norm * exp_term. 反向 d_key_norm_pathC = d_kgexp_from_value * exp_term. 形状 [L,D]*[L,1]=[L,D]
                exp_term = (_last_gate_1 - gate_cum).exp()  # [L,1]
                d_key_norm_pathC = d_kgexp_from_value * exp_term
                # d_key_norm_pathC = pypto.cast(d_key_norm_pathC, pypto.DT_FP32)

                # ---------- 路径 D（kgexp 支）: d_state_next -> temp_matmul_vprime -> kgexp -> key_norm ----------
                # Step D2: 前向 temp_matmul_vprime = v_prime^T @ kgexp. 反向 d_kgexp_from_v_prime = v_prime @ (-d_state_next)^T = - v_prime @ d_state_next^T. 形状 [L,D]
                d_kgexp_from_v_prime = pypto.matmul(v_prime, d_state_next, pypto.DT_FP32, b_trans=True) * (-1.0)
                # Step D3: 前向 kgexp = key_norm * exp_term. 反向 d_key_norm_pathD = d_kgexp_from_v_prime * exp_term. 形状 [L,D]
                d_key_norm_pathD = d_kgexp_from_v_prime * exp_term
                # d_key_norm_pathD = pypto.cast(d_key_norm_pathD, pypto.DT_FP32)

                # ---------- 合并 4 路 d_key_norm，形状均为 [L,D] ----------
                # 路径 A: d_key_norm_pathA; 路径 B: d_key_norm_pathB_via_beta, d_key_norm_pathB_from_kkt; 路径 C: d_key_norm_pathC; 路径 D: d_key_norm_pathD
                d_key_norm_chunk = d_key_norm_pathA + d_key_norm_pathB_via_beta + d_key_norm_pathB_from_kkt + d_key_norm_pathC + d_key_norm_pathD
                # d_key_norm_chunk = pypto.cast(d_key_norm_chunk, pypto.DT_FP32)
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                d_key_norm[bs_ofs : bs_ofs + l, nv_idx] = d_key_norm_chunk


                # ========== d_gate ==========
                pypto.set_vec_tile_shapes(128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                # 目标: d_gate [L,1]. 前向 gate_cum = tril_mask @ gate => 反向 d_gate = tril_mask^T @ d_gate_cum.
                # 先累加各路径对 d_gate_cum [L,1] 的贡献，再左乘 tril_mask^T.

                # ---------- 路径 A: d_chunk_out -> attn_inter -> qgexp -> gate_exp -> gate_cum ----------
                # 前向: attn_inter = qgexp @ state^T; qgexp = query_scale * gate_exp. 形状 qgexp [L,D], gate_exp [L,1]
                # 反向: d_qgexp = d_attn_inter @ state（已有）[L,D]; d_gate_exp += (d_qgexp * query_scale).sum(-1, keepdim=True) [L,1]
                # 前向: gate_exp = exp(gate_cum) => 反向 d_gate_cum += d_gate_exp * gate_exp
                d_gate_exp_A = (d_qgexp * query_scale).sum(-1, keepdim=True)

                # ---------- 路径 C: d_chunk_out -> chunk_attn_vprime -> v_prime -> key_cum_out -> w -> gate_exp -> gate_cum ----------
                # 前向: w = key_beta * gate_exp; d_w_from_cum 已算. 反向 d_gate_exp += (d_w_from_cum * key_beta).sum(-1, keepdim=True) [L,1]
                d_gate_exp_C = (d_w_from_cum * key_beta).sum(-1, keepdim=True)

                # 合并路径 A+C 对 gate_exp 的梯度，再通过 gate_exp = exp(gate_cum) 得到对 gate_cum 的贡献
                # 前向: gate_exp = exp(gate_cum). 反向: d_gate_cum += d_gate_exp * gate_exp（逐元素）
                d_gate_exp_AC = d_gate_exp_A + d_gate_exp_C
                d_gate_cum_AC = d_gate_exp_AC * gate_exp

                # ---------- 路径 B: d_chunk_out -> chunk_attn_value/chunk_attn_vprime -> attn_tmp -> decay_mask -> gate_cum ----------
                # 前向: attn_tmp = attn * decay_mask * tril_mask; decay_mask = exp(x), 
                # x = (gate_cum - gate_cum^T) * tril_mask
                # 反向: d_decay_mask = d_attn_tmp_total * attn * tril_mask [L,L]（对 decay_mask 求导时 attn、tril_mask 为常数）; d_x = d_decay_mask * decay_mask（exp 链式）
                # 路径 1：a_block = kkt * decay_mask * mask => d_decay_mask += d_a_block_from_inv * kkt * mask
                d_decay_mask_from_a_block = d_a_block_from_inv * kkt * mask

                # 路径 2: d_decay_mask = d_attn_tmp_total * attn * tril_mask 
                d_decay_mask_from_attn_tmp = d_attn_tmp_total * attn * tril_mask
                d_decay_mask = d_decay_mask_from_a_block + d_decay_mask_from_attn_tmp
                
                # 反向: x_ij = (gate_cum_i - gate_cum_j)*tril_ij => d_gate_cum_i += sum_j (d_x_ij * tril_ij) - sum_j (d_x_ji * tril_ji)
                # 即 d_gate_cum_B = (d_x * tril_mask).sum(1, keepdim=True) - (d_x * tril_mask).sum(0, keepdim=True).T 形状 [L,1]
                # 因为 d_x = d_decay_mask * decay_mask（exp 链式） 
                d_x = d_decay_mask * decay_mask
               
                d_x_tril_mask = d_x * tril_mask
                d_gate_cum_B_row = pypto.sum(d_x_tril_mask, 1, keepdim=True)

                d_gate_cum_B_col = pypto.sum(d_x_tril_mask, 0, keepdim=True)
                d_gate_cum_B = d_gate_cum_B_row - d_gate_cum_B_col.transpose(0, 1)

                # ---------- 路径 G: d_state_next -> temp_matmul_value/temp_matmul_vprime -> kgexp -> diff -> gate_cum ----------
                # 前向: kgexp = key * exp(_last_gate_1 - gate_cum) = key * exp(diff), diff = _last_gate_1 - gate_cum [L,1]
                # 反向: d_diff = (d_kgexp * kgexp).sum(-1, keepdim=True) [L,1]; diff 对 gate_cum 得 d_gate_cum += -d_diff; 对 _last_gate_1 得 d_last_gate_1 = d_diff.sum()
                d_kgexp_total = d_kgexp_from_value + d_kgexp_from_v_prime

                d_diff = (d_kgexp_total * kgexp).sum(-1, keepdim=True)
    
                d_gate_cum_G = pypto.neg(d_diff)
                d_last_gate_1_scalar = pypto.sum(d_diff, 0, keepdim=True)          # [1, 1]

                # ---------- 路径 D: d_state_next -> final_state_1 -> _last_gate_2 -> gate_exp[L-1] -> gate_cum[L-1] ----------
                # 前向: final_state_1 = state * _last_gate_2（逐元素）; _last_gate_2 = expand(gate_exp[L-1]) [D,1]
                # 反向: d_last_gate_2 = (d_state_next * state_chunk).sum(1, keepdim=True) [D,1]; d_gate_exp[L-1] += d_last_gate_2.sum() 这里 pypto 不支持这种直接sum 成scalar 的操作，需要 keep dim
                # 前向: gate_exp[L-1] = exp(gate_cum[L-1]) => d_gate_cum[L-1] += d_gate_exp[L-1] * gate_exp[L-1]
                d_last_gate_2_grad = pypto.sum(d_state_next * state_chunk, 1, keepdim=True)
                d_gate_exp_last_scalar = pypto.sum(d_last_gate_2_grad, 0, keepdim=True)        # [1, 1]

                d_gate_cum_D_last = d_gate_exp_last_scalar * gate_exp[actual_l - 1 : actual_l, :]
                
                # ---------- 汇总 d_gate_cum：路径 A+C、B、G 全行；路径 D 与路径 G 的 d_last_gate_1 只加在最后一行 ----------
                # 避免原地切片赋值（会导致 mix assemble and common operation for same output），改为构造“仅最后一行非零”的 [L,1] 再相加
                # tril_mask 下三角，tril_mask[:, L-1] 为 [0,...,0,1]^T，用作 mask
                d_gate_cum_last_add = d_gate_cum_D_last + d_last_gate_1_scalar

                mask_last_row = tril_mask[:, actual_l - 1 : actual_l] # [L, 1]，仅最后行为 1
                d_gate_cum_last_row_tensor = pypto.mul(d_gate_cum_last_add, mask_last_row) # [L,1]，仅最后行非零

                d_gate_cum = d_gate_cum_AC + d_gate_cum_B + d_gate_cum_G + d_gate_cum_last_row_tensor

                # ---------- 前向 gate_cum = tril_mask @ gate；反向 d_gate = tril_mask^T @ d_gate_cum ----------
                # 形状 tril_mask [L,L], d_gate_cum [L,1] => tril_mask^T @ d_gate_cum = [L,1]
                # pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                d_gate_chunk = pypto.matmul(tril_mask, d_gate_cum, pypto.DT_FP32, a_trans=True)

                pypto.set_vec_tile_shapes(128, 1)
                d_gate[bs_ofs : bs_ofs + l, nv_idx : nv_idx + 1] = d_gate_chunk


                # ========== d_beta ==========
                pypto.set_vec_tile_shapes(128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                # ---------- 前向（本 chunk）----------
                # 路径 1 (value 支): beta_view [L,1] -> value_beta = value_view_2d * beta_view [L,D]
                #   -> value_out = A_inv @ value_beta [L,D] -> chunk_attn_value / temp_matmul_value -> chunk_attn_out 与 cur_state.
                # 路径 2 (key_beta 支): beta_view -> key_beta = key_view_2d * beta_view [L,D]
                #   -> kkt = key_beta @ key_view_2d^T [L,L] -> a_block; key_beta -> key_cum_out = A_inv @ (key_beta * gate_exp) -> v_prime -> chunk_attn_vprime / temp_matmul_vprime.
                # ---------- 反向（复用已有中间量）----------
                # 路径 1: d_value_out 已算 (d_value_out_from_attn + d_value_out_from_state); d_value_beta = A^T @ d_value_out 已在 d_value 段求得 [L,D].
                #   对 value_beta = value * beta 求导 => d_beta += (d_value_beta * value_view_2d).sum(dim=-1, keepdim=True), 形状 [L,1].
                # 路径 2: d_key_beta_pathB_total = d_key_beta_from_w + d_key_beta_from_kkt 已在本段求得 [L,D].
                #   对 key_beta = key_norm * beta 求导 => d_beta += (d_key_beta_pathB_total * key_view_2d).sum(dim=-1, keepdim=True), 形状 [L,1].
                # 合并两路后写入 d_beta，形状 [L,1] -> 写回 d_beta[bs_ofs : bs_ofs + l, nv_idx : nv_idx + 1]. 需在 args 中增加 d_beta（例如 args[17]）并传入输出 buffer.
                d_beta_from_value = pypto.sum(d_value_beta * value_view_2d, 1, keepdim=True)   # [L,D] 对最后一维求和 -> [L,1]
                d_beta_from_key_beta = pypto.sum(d_key_beta_pathB_total * key_view_2d, 1, keepdim=True)   # [L,D] -> [L,1]
                d_beta_chunk = d_beta_from_value + d_beta_from_key_beta
                pypto.set_vec_tile_shapes(128, 1)
                d_beta[bs_ofs : bs_ofs + l, nv_idx : nv_idx + 1] = d_beta_chunk


                # ---------- d_state_prev（跨 chunk 传递） ----------
                # 使用上面算好的 d_v_prime_total
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                d_state_from_attn_inter = pypto.matmul(qgexp, d_attn_inter, pypto.DT_FP32, a_trans=True)
                d_state_from_v_prime = pypto.matmul(key_cum_out, d_v_prime_total, pypto.DT_FP32, a_trans=True)
                d_state_prev = d_state_from_attn_inter + d_state_from_v_prime
                d_state_prev = d_state_prev + _last_gate_2 * d_state_next
                d_state_next[:] = d_state_prev


            for c_pass2_idx in pypto.loop(num_chunks_int, name="LOOP_C_PASS2_IDX", idx_name="c_pass2_idx"):
                cal_d_value(c_pass2_idx, d_state_next)


def pypto_chunk_gated_delta_rule_backward_all_clear(
    do, d_last_state, q, k, all_states_res, v, g, beta, A,
    mask, tril_mask, act_seq_len,
    d_value, d_query_norm, d_key_norm, 
    d_gate, d_beta
):
    """调用 chunk_gated_delta_rule_backward_all_clear"""

    
    act_seq_len_int = int(act_seq_len[1].item())
    
    input_tensors = [
        (do, [0]), (d_last_state, []), (q, [0]), (k, [0]), (all_states_res, []), (v, [0]), (g, [0]), (beta, [0]), (A, []),
        (mask, []), (tril_mask, []), (act_seq_len, []),
        (d_value, [0]),
        (d_query_norm, [0]),
        (d_key_norm, [0]),
        (d_gate, [0]),
        (d_beta, [0]),
        # (dgate_path1, [0]),
        # (dgate_path2, [0]),
    ]
    pto_inputs = [pypto.from_torch(t, dynamic_axis=ax) for t, ax in input_tensors]
    # chunk_gated_delta_rule_backward_all_clear(*pto_inputs, act_seq_len_int=act_seq_len_int) # , act_seq_len=act_seq_len

    chunk_gated_delta_rule_backward_all_clear_test(*pto_inputs, act_seq_len_int=act_seq_len_int) # , act_seq_len=act_seq_len
    torch_npu.npu.synchronize()


class FusedChunkGatedDeltaRulePyptoWoL2norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        beta,
        gate,
        states,
        mask,
        tril_mask,
        eye,
        act_seq_len,
        eps=1e-6,
    ):
        """
        PyPTO 封装版 forward。

        输入
        ----
        query/key/value : [T, N, D], float32
        beta/gate       : [T, N], float32
        states          : [B, N, D, D], float32
        mask/tril_mask  : [L, L], float32
        eye             : [16, 128], float32
        act_seq_len     : [B+1], int32

        输出
        ----
        core_attn_out : [T, N, D]
        final_state   : [B, N, D, D]
        """
        assert query.dtype == torch.float32
        assert key.dtype == torch.float32
        assert value.dtype == torch.float32
        assert beta.dtype == torch.float32
        assert gate.dtype == torch.float32
        assert states.dtype == torch.float32

        T, _, D = query.shape
        _, Nv, _ = value.shape
        B = states.shape[0]
        L = mask.shape[0]
        max_chunks = (T + L - 1) // L
        device = query.device

        outputs = {
            "core_attn_out": torch.zeros(T, Nv, D, dtype=torch.float32, device=device),
            "final_state": torch.zeros(B, Nv, D, D, dtype=torch.float32, device=device),
            "inverse_a_block_res": torch.zeros(B, Nv, max_chunks, L, L, dtype=torch.float32, device=device),
            "all_states_res": torch.zeros(B, Nv, max_chunks + 1, D, D, dtype=torch.float32, device=device),
        }
        debug_tensor_stats("query", query)
        debug_tensor_stats("key", key)
        debug_tensor_stats("value", value)
        debug_tensor_stats("beta", beta)
        debug_tensor_stats("gate", gate)
        debug_tensor_stats("states", states)
        debug_tensor_stats("mask", mask)
        debug_tensor_stats("tril_mask", tril_mask)
        debug_tensor_stats("eye", eye)

        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "beta": beta,
            "gate": gate,
            "states": states,
            "mask": mask,
            "tril_mask": tril_mask,
            "eye": eye,
            "act_seq_len": act_seq_len,
        }

        pypto_chunk_gated_delta_rule_fwd_dyn_clear_wo_l2norm(inputs, outputs)

        debug_tensor_stats("core_attn_out", outputs["core_attn_out"])
        debug_tensor_stats("final_state", outputs["final_state"])
        debug_tensor_stats("inverse_a_block_res", outputs["inverse_a_block_res"])
        debug_tensor_stats("all_states_res", outputs["all_states_res"])

        # ctx.eps = eps
        ctx.save_for_backward(
            query,
            key,
            value,
            beta,
            gate,
            states,
            mask,
            tril_mask,
            act_seq_len,
            outputs["inverse_a_block_res"],
            outputs["all_states_res"],
        )
        return outputs["core_attn_out"], outputs["final_state"]

    @staticmethod
    def backward(ctx, grad_core_attn_out, grad_final_state):
        (
            query,
            key,
            value,
            beta,
            gate,
            states,
            mask,
            tril_mask,
            act_seq_len,
            inverse_a_block_res,
            all_states_res,
        ) = ctx.saved_tensors

        device = query.device
        # eps = ctx.eps

        do = grad_core_attn_out.contiguous()
        debug_tensor_stats("do", do)
        if grad_final_state is None:
            d_last_state = torch.zeros_like(states, device=device)
        else:
            d_last_state = grad_final_state.contiguous()
        debug_tensor_stats("d_last_state", d_last_state)
        d_query = torch.zeros_like(query, device=device)
        d_key = torch.zeros_like(key, device=device)
        d_value = torch.zeros_like(value, device=device)
        d_gate = torch.zeros_like(gate, device=device)
        d_beta = torch.zeros_like(beta, device=device)
        
        pypto_chunk_gated_delta_rule_backward_all_clear(
            do,
            d_last_state,
            query,
            key,
            all_states_res,
            value,
            gate,
            beta,
            inverse_a_block_res,
            mask,
            tril_mask,
            act_seq_len,
            d_value,
            d_query,
            d_key,
            d_gate,
            d_beta,
        )
        debug_tensor_stats("d_value", d_value)
        debug_tensor_stats("d_query", d_query)
        debug_tensor_stats("d_key", d_key)
        debug_tensor_stats("d_gate", d_gate)
        debug_tensor_stats("d_beta", d_beta)



        # 当前这套 PyPTO backward 没有显式产出 d_states（对 initial_state 的梯度），这里返回 None。
        return (
            d_query,   # query
            d_key,     # key
            d_value,   # value
            d_beta,    # beta
            d_gate,    # gate
            None,      # states
            None,      # mask
            None,      # tril_mask
            None,      # eye
            None,      # act_seq_len
            None,      # eps
        )


def pypto_chunk_gated_delta_rule_autograd_wo_l2norm(
    query,
    key,
    value,
    beta,
    gate,
    states,
    mask,
    tril_mask,
    eye,
    act_seq_len,
    eps=1e-6,
):
    """
    对外调用接口，返回：
    - core_attn_out
    - final_state
    """
    return FusedChunkGatedDeltaRulePyptoWoL2norm.apply(
        query,
        key,
        value,
        beta,
        gate,
        states,
        mask,
        tril_mask,
        eye,
        act_seq_len,
        eps,
    )

def test_chunk_gated_delta_rule_pypto_autograd_performance_wo_l2norm(
    device_id: int = None,
    num_rounds: int = 10,
    warmup_rounds: int = 3,
    params: dict = None,
):
    """
    使用 PyPTO autograd wo_l2norm 封装 vs Torch 可微实现，做多轮前后向性能与正确性测试。
    gen_inputs 之后先对 query/key 做 L2 norm，再分别喂给 PyPTO 与 Torch（Torch 使用 use_qk_l2norm_in_kernel=False）。
    统计项同 test_chunk_gated_delta_rule_pypto_autograd_performance。
    """
    if device_id is None:
        device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    if params is None:
        params = {"T": 4096, "B": 1, "Nqk": 32, "Nv": 32}

    dims = gen_dims(params)
    T = dims["T"]
    B = dims["B"]
    Nqk = dims["Nqk"]
    Nv = dims["Nv"]
    D = dims["D"]
    L = dims["L"]

    rtol, atol_abs, atol_rel = 1e-3, 1e-3, 1e-3
    eps = 1e-6

    pypto_fwd_times = []
    pypto_bwd_times = []
    pypto_total_times = []
    pypto_fwd_mems = []
    pypto_bwd_mems = []
    pypto_total_mems = []

    torch_fwd_times = []
    torch_bwd_times = []
    torch_total_times = []
    torch_fwd_mems = []
    torch_bwd_mems = []
    torch_total_mems = []

    total_rounds = warmup_rounds + num_rounds

    print(
        f"\n🚀 PyPTO Autograd (wo_l2norm) vs Torch 多轮前后向测试 (warmup={warmup_rounds}, 测试={num_rounds}) on {device}"
    )
    print(f"Config: B={B}, T={T}, Nqk={Nqk}, Nv={Nv}, D={D}, L={L} (gen_input 后先 L2 norm 再分别过 PyPTO/Torch)")

    for r in range(total_rounds):
        torch.manual_seed(r + 300)
        inputs_data = gen_inputs(dims, torch.float32)

        # gen_input 之后先 L2 norm，再 .npu() 给两路用
        query_raw = inputs_data["query"]
        key_raw = inputs_data["key"]
        q_norm = query_raw * torch.rsqrt((query_raw * query_raw).sum(dim=-1, keepdim=True) + eps)
        k_norm = key_raw * torch.rsqrt((key_raw * key_raw).sum(dim=-1, keepdim=True) + eps)

        query = q_norm.npu()
        key = k_norm.npu()
        value = inputs_data["value"].npu()
        beta = inputs_data["beta"].npu()
        gate = inputs_data["gate"].npu()
        states = inputs_data["states"].npu()
        act_seq_len = inputs_data["act_seq_len"].npu()
        mask = inputs_data["mask"].npu()
        tril_mask = inputs_data["tril_mask"].npu()
        eye = inputs_data["eye"].npu()

        do = (torch.rand(T, Nv, D, dtype=torch.float32, device=device) - 0.5) * 0.15

        dstate = torch.rand(B, Nv, D, D, dtype=torch.float32, device=device) 

        # PyPTO autograd wo_l2norm 路径（输入已是 q_norm/k_norm）
        q_pypto = query.clone().detach().requires_grad_(True)
        k_pypto = key.clone().detach().requires_grad_(True)
        v_pypto = value.clone().detach().requires_grad_(True)
        b_pypto = beta.clone().detach().requires_grad_(True)
        g_pypto = gate.clone().detach().requires_grad_(True)
        s_pypto = states.clone().detach()

        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        core_pypto, state_pypto = pypto_chunk_gated_delta_rule_autograd_wo_l2norm(
            q_pypto, k_pypto, v_pypto, b_pypto, g_pypto, s_pypto,
            mask, tril_mask, eye, act_seq_len, eps=eps,
        )
        torch.npu.synchronize()
        t1 = time.perf_counter()
        pypto_fwd_time = t1 - t0
        pypto_fwd_mem = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        loss_pypto = (core_pypto * do).sum() 
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        loss_pypto.backward()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        pypto_bwd_time = t1 - t0
        pypto_bwd_mem = torch.npu.max_memory_allocated(device) / (1024 ** 2)
        pypto_total_time = pypto_fwd_time + pypto_bwd_time
        pypto_total_mem = max(pypto_fwd_mem, pypto_bwd_mem)

        # Torch 可微路径（输入已是 q_norm/k_norm，内部不再 norm）
        q_torch = query.clone().detach().requires_grad_(True)
        k_torch = key.clone().detach().requires_grad_(True)
        v_torch = value.clone().detach().requires_grad_(True)
        b_torch = beta.clone().detach().requires_grad_(True)
        g_torch = gate.clone().detach().requires_grad_(True)
        s_torch = states.clone().detach()

        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        core_torch, state_torch, _ = segs_chunk_gated_delta_rule_differentiable(
            query=q_torch, key=k_torch, value=v_torch, gate=g_torch, beta=b_torch,
            act_seq_len=act_seq_len, chunk_size=L, initial_state=s_torch,
            output_final_state=True, use_qk_l2norm_in_kernel=False, return_intermediates=False,
        )
        torch.npu.synchronize()
        t1 = time.perf_counter()
        torch_fwd_time = t1 - t0
        torch_fwd_mem = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        loss_torch = (core_torch * do).sum() 
        # + (state_torch * dstate).sum()
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        loss_torch.backward()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        torch_bwd_time = t1 - t0
        torch_bwd_mem = torch.npu.max_memory_allocated(device) / (1024 ** 2)
        torch_total_time = torch_fwd_time + torch_bwd_time
        torch_total_mem = max(torch_fwd_mem, torch_bwd_mem)

        # 正确性对比
        compare(actual=core_pypto.detach().cpu(), expected=core_torch.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch core_attn_out",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)
        compare(actual=state_pypto.detach().cpu(), expected=state_torch.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch final_state",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)
        compare(actual=q_pypto.grad.detach().cpu(), expected=q_torch.grad.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch d_query",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)
        compare(actual=k_pypto.grad.detach().cpu(), expected=k_torch.grad.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch d_key",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)
        compare(actual=v_pypto.grad.detach().cpu(), expected=v_torch.grad.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch d_value",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)
        compare(actual=b_pypto.grad.detach().cpu(), expected=b_torch.grad.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch d_beta",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)
        compare(actual=g_pypto.grad.detach().cpu(), expected=g_torch.grad.detach().cpu(),
                name=f"Round{r+1} pypto_autograd_wo_l2norm_vs_torch d_gate",
                rtol=rtol, atol_abs=atol_abs, atol_rel=atol_rel)

        if r >= warmup_rounds:
            pypto_fwd_times.append(pypto_fwd_time)
            pypto_bwd_times.append(pypto_bwd_time)
            pypto_total_times.append(pypto_total_time)
            pypto_fwd_mems.append(pypto_fwd_mem)
            pypto_bwd_mems.append(pypto_bwd_mem)
            pypto_total_mems.append(pypto_total_mem)
            torch_fwd_times.append(torch_fwd_time)
            torch_bwd_times.append(torch_bwd_time)
            torch_total_times.append(torch_total_time)
            torch_fwd_mems.append(torch_fwd_mem)
            torch_bwd_mems.append(torch_bwd_mem)
            torch_total_mems.append(torch_total_mem)
            print(
                f"Round{r+1} | "
                f"PyPTO fwd: {pypto_fwd_time*1000:.2f} ms / {pypto_fwd_mem:.1f} MB, "
                f"bwd: {pypto_bwd_time*1000:.2f} ms / {pypto_bwd_mem:.1f} MB, "
                f"total: {pypto_total_time*1000:.2f} ms / {pypto_total_mem:.1f} MB | "
                f"Torch fwd: {torch_fwd_time*1000:.2f} ms / {torch_fwd_mem:.1f} MB, "
                f"bwd: {torch_bwd_time*1000:.2f} ms / {torch_bwd_mem:.1f} MB, "
                f"total: {torch_total_time*1000:.2f} ms / {torch_total_mem:.1f} MB"
            )

        del (q_pypto, k_pypto, v_pypto, b_pypto, g_pypto, s_pypto,
             q_torch, k_torch, v_torch, b_torch, g_torch, s_torch,
             core_pypto, state_pypto, loss_pypto, core_torch, state_torch, loss_torch, do, dstate)
        torch.npu.empty_cache()

    def _mean(x):
        return sum(x) / len(x) if x else 0.0

    avg_pypto_fwd_t = _mean(pypto_fwd_times)
    avg_pypto_bwd_t = _mean(pypto_bwd_times)
    avg_pypto_total_t = _mean(pypto_total_times)
    avg_pypto_fwd_m = _mean(pypto_fwd_mems)
    avg_pypto_bwd_m = _mean(pypto_bwd_mems)
    avg_pypto_total_m = _mean(pypto_total_mems)
    avg_torch_fwd_t = _mean(torch_fwd_times)
    avg_torch_bwd_t = _mean(torch_bwd_times)
    avg_torch_total_t = _mean(torch_total_times)
    avg_torch_fwd_m = _mean(torch_fwd_mems)
    avg_torch_bwd_m = _mean(torch_bwd_mems)
    avg_torch_total_m = _mean(torch_total_mems)

    print("\n" + "=" * 70)
    print(f"📈 PyPTO Autograd (wo_l2norm) vs Torch 汇总 (平均 {num_rounds} 轮, 不含 warmup {warmup_rounds} 轮)")
    print(f"    Config: B={B}, T={T}, Nqk={Nqk}, Nv={Nv}, D={D}, L={L}")
    print("=" * 70)
    print("  【前向】")
    print(f"    Torch:  {avg_torch_fwd_t*1000:.2f} ms  |  显存: {avg_torch_fwd_m:.1f} MB")
    print(f"    PyPTO:  {avg_pypto_fwd_t*1000:.2f} ms  |  显存: {avg_pypto_fwd_m:.1f} MB")
    if avg_pypto_fwd_t > 0:
        print(f"    Speedup: {avg_torch_fwd_t / avg_pypto_fwd_t:.2f}x")
    print("  【反向】")
    print(f"    Torch:  {avg_torch_bwd_t*1000:.2f} ms  |  显存: {avg_torch_bwd_m:.1f} MB")
    print(f"    PyPTO:  {avg_pypto_bwd_t*1000:.2f} ms  |  显存: {avg_pypto_bwd_m:.1f} MB")
    if avg_pypto_bwd_t > 0:
        print(f"    Speedup: {avg_torch_bwd_t / avg_pypto_bwd_t:.2f}x")
    print("  【合并：时间=前向+反向均值，显存=逐轮 max(fwd,bwd) 后取均值】")
    print(f"    Torch:  {avg_torch_total_t*1000:.2f} ms  |  显存: {avg_torch_total_m:.1f} MB")
    print(f"    PyPTO:  {avg_pypto_total_t*1000:.2f} ms  |  显存: {avg_pypto_total_m:.1f} MB")
    if avg_pypto_total_t > 0:
        print(f"    Speedup: {avg_torch_total_t / avg_pypto_total_t:.2f}x")
    print("=" * 70)


def test_chunk_gated_delta_rule_pypto_autograd_perf_default_wo_l2norm():
    """PyPTO autograd wo_l2norm vs Torch 多轮测试：gen_input 后先 L2 norm，再分别过 PyPTO/Torch。"""
    test_chunk_gated_delta_rule_pypto_autograd_performance_wo_l2norm(
        device_id=int(os.environ.get("TILE_FWK_DEVICE_ID", 0)),
        num_rounds=10,
        warmup_rounds=3,
        params={"T": 4096, "B": 1, "Nqk": 32, "Nv": 32},
    )
    


@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 32,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 32,
    },
)
def chunk_gated_delta_rule_fwd_clear_wo_l2norm_test_ablock(*args):
    """
    Chunk Gated Delta Rule fused operator.

    This is the main entry point for the Gated Delta Rule attention computation.
    It processes input sequences in chunks of size L=128, maintaining recurrent
    state across chunks for efficient long sequence modeling.

    Parameters
    ----------
    query : Input query tensor, shape [T, Nqk, D], dtype float32 after l2norm
    key : Input key tensor, shape [T, Nqk, D], dtype float32 after l2norm
    value : Input value tensor, shape [T, Nv, D], dtype float32
    beta : Beta scaling factor, shape [T, Nv], dtype float32
    gate : Gate signal, shape [T, Nv], dtype float32
    states : Initial recurrent states, shape [B, Nv, D, D], dtype float32
    mask : Attention mask (lower triangular negative), shape [L, L], dtype float32
    tril_mask : Lower triangular mask, shape [L, L], dtype float32
    eye : Identity matrix (specially processed), shape [16, 128], dtype float32
    act_seq_len : Cumulative sequence length indices, shape [B+1], dtype int32
    core_attn_out : Output attention tensor, shape [T, Nv, D], dtype float32
    last_state_data : Output updated states, shape [B, Nv, D, D], dtype float32
    query_norm_res : Output query normalization tensor, shape [T, Nqk, D], dtype float32
    key_norm_res : Output key normalization tensor, shape [T, Nqk, D], dtype float32
    inverse_a_block_res : Output inverse A block tensor, shape [B, Nv, max_chunks, L, L], dtype float32
    all_states_res : Output all chunks state tensor, shape [B, Nv, max_chunks + 1, D, D], dtype float32
    """
    query = args[0]
    key = args[1]
    value = args[2]
    beta = args[3]
    gate = args[4]
    states = args[5]
    mask = args[6]
    tril_mask = args[7]
    eye = args[8]
    act_seq_len = args[9]
    core_attn_out = args[10]
    last_state_data = args[11]

    inverse_a_block_res = args[12]
    all_states_res = args[13]
    a_block_out = args[14]
    _, nqk, d = query.shape
    _, nv, d = value.shape
    b = states.shape[0]
    l, l = mask.shape
    group = nv // nqk
    for b_idx in pypto.loop(b, name="LOOP_B_TND", idx_name="b_idx"):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        for nv_idx in pypto.loop(nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(16, 16, 128, 128)
            last_state = states[b_idx, nv_idx]
            # print(f"last_state shape: {last_state.shape}")
            all_states_res[b_idx, nv_idx, 0] = states[b_idx, nv_idx]
            for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx"):
                bs_ofs = b_ofs + s_idx
                actual_l = (s - s_idx).min(l)
                chunk_idx = s_idx // l
                ## view
                query_view = pypto.view(query, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                key_view = pypto.view(key, [l, 1, d], [bs_ofs, nqk_idx, 0], valid_shape=[actual_l, 1, d])
                value_view = pypto.view(value, [l, 1, d], [bs_ofs, nv_idx, 0], valid_shape=[actual_l, 1, d])
                beta_view = pypto.view(beta, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])
                gate_view = pypto.view(gate, [l, 1], [bs_ofs, nv_idx], valid_shape=[actual_l, 1])

                pypto.set_vec_tile_shapes(128, 128, 128)
                query_norm = pypto.reshape(query_view, [l, d], valid_shape=[actual_l, d])
                key_norm = pypto.reshape(key_view, [l, d], valid_shape=[actual_l, d])
                value_view_2d = pypto.reshape(value_view, [l, d], valid_shape=[actual_l, d])

                scale = 1 / d**0.5
                query_scale = query_norm * scale

                # kv_beta & g_cumsum & decay_mask & pre_attn
                gate_cum, decay_mask, a_block_chunk, key_beta = pre_attn_v2(gate_view, key_norm, beta_view, tril_mask, mask)
                # gate_cum_res[bs_ofs:bs_ofs + l, nv_idx:nv_idx + 1] = gate_cum  # 存储中间结果 用于 bwd 计算
                a_block_out[b_idx:b_idx + 1, nv_idx:nv_idx + 1, chunk_idx:chunk_idx + 1, :, :] = pypto.reshape(a_block_chunk, [1, 1, 1, l, l])
                # inverse
                a_block_inverse = inverse_pto_v2(a_block_chunk, eye, 128)
                inverse_a_block_res[b_idx:b_idx + 1, nv_idx:nv_idx + 1, chunk_idx:chunk_idx + 1, :, :] = pypto.reshape(a_block_inverse, [1, 1, 1, l, l])     # 存储中间结果 用于 bwd 计算
                # cal_value_and_keycumdecay
                pypto.set_vec_tile_shapes(128, 128, 128)
                value_out, key_cum_out = cal_value_and_key_cumdecay_v2(a_block_inverse, value_view_2d,
                    beta_view, key_beta, gate_cum)

                chunk_attn_out, cur_state = recurrent_state_attn_all_v2(query=query_scale, key=key_norm, value=value_out,
                    k_cumdecay=key_cum_out, gate=gate_cum, state=last_state, decay_mask=decay_mask, tril=tril_mask)

                # assemble
                pypto.set_vec_tile_shapes(16, 16, 128, 128)
                # print(f"cur_state shape: {cur_state.shape}")
                last_state[:] = cur_state
                core_attn_out[bs_ofs:bs_ofs + l, nv_idx] = chunk_attn_out
                last_state_data[b_idx, nv_idx] = last_state
                all_states_res[b_idx, nv_idx, chunk_idx + 1] = last_state


def pypto_chunk_gated_delta_rule_fwd_dyn_clear_wo_l2norm_test_ablock(inputs: dict, outputs: dict):
    """调用 chunk_gated_delta_rule_fwd_clear_wo_l2norm core_attn_out、last_state_data 及 bwd 所需中间结果。"""
    L = inputs["mask"].shape[0]

    input_tensors = [
        (inputs["query"], [0]),  # after l2norm
        (inputs["key"], [0]),  # after l2norm
        (inputs["value"], [0]),
        (inputs["beta"], [0]),
        (inputs["gate"], [0]),
        (inputs["states"], []),
        (inputs["mask"], []),
        (inputs["tril_mask"], []),
        (inputs["eye"], []),
        (inputs["act_seq_len"], []),
    ]
    output_tensors = [
        (outputs["core_attn_out"], [0]),
        (outputs["final_state"], []),
        (outputs["inverse_a_block_res"], []),
        (outputs["all_states_res"], []),
        (outputs["a_block"], []),
    ]
    pto_inputs = [pypto.from_torch(t, dynamic_axis=ax) for t, ax in input_tensors]
    pto_outputs = [pypto.from_torch(t, dynamic_axis=ax) for t, ax in output_tensors]
    chunk_gated_delta_rule_fwd_clear_wo_l2norm_test_ablock(*pto_inputs, *pto_outputs)
    torch_npu.npu.synchronize()


def test_multi_round_fwd_a_block(
    device_id: int = None,
    num_rounds: int = 5,
    params: dict = None,
):
    """
    多轮前向测试：每轮 gen_inputs + L2 norm q/k 后调用 test_ablock 前向，打印每轮的 a_block 与 inverse_a_block_res。
    不修改 gen_input 范围，仅用于观察 a_block 是否随轮次/输入变化。
    """
    if device_id is None:
        device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    if params is None:
        params = {"T": 4096, "B": 1, "Nqk": 32, "Nv": 32}

    dims = gen_dims(params)
    T = dims["T"]
    B = dims["B"]
    Nqk = dims["Nqk"]
    Nv = dims["Nv"]
    D = dims["D"]
    L = dims["L"]
    max_chunks = (T + L - 1) // L
    eps = 1e-6

    print(
        f"\n🚀 多轮前向测试 (a_block 打印) num_rounds={num_rounds} on {device}"
    )
    print(f"Config: B={B}, T={T}, Nqk={Nqk}, Nv={Nv}, D={D}, L={L}, max_chunks={max_chunks}")

    for r in range(num_rounds):
        torch.manual_seed(r + 300)
        inputs_data = gen_inputs(dims, torch.float32)

        query_raw = inputs_data["query"]
        key_raw = inputs_data["key"]
        q_norm = query_raw * torch.rsqrt((query_raw * query_raw).sum(dim=-1, keepdim=True) + eps)
        k_norm = key_raw * torch.rsqrt((key_raw * key_raw).sum(dim=-1, keepdim=True) + eps)

        query = q_norm.to(device)
        key = k_norm.to(device)
        value = inputs_data["value"].to(device)
        beta = inputs_data["beta"].to(device)
        gate = inputs_data["gate"].to(device)
        states = inputs_data["states"].to(device)
        act_seq_len = inputs_data["act_seq_len"].to(device)
        mask = inputs_data["mask"].to(device)
        tril_mask = inputs_data["tril_mask"].to(device)
        eye = inputs_data["eye"].to(device)

        outputs = {
            "core_attn_out": torch.zeros(T, Nv, D, dtype=torch.float32, device=device),
            "final_state": torch.zeros(B, Nv, D, D, dtype=torch.float32, device=device),
            "inverse_a_block_res": torch.zeros(B, Nv, max_chunks, L, L, dtype=torch.float32, device=device),
            "all_states_res": torch.zeros(B, Nv, max_chunks + 1, D, D, dtype=torch.float32, device=device),
            "a_block": torch.zeros(B, Nv, max_chunks, L, L, dtype=torch.float32, device=device),
        }
        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "beta": beta,
            "gate": gate,
            "states": states,
            "mask": mask,
            "tril_mask": tril_mask,
            "eye": eye,
            "act_seq_len": act_seq_len,
        }
        debug_tensor_stats("query", query)
        debug_tensor_stats("key", key)
        debug_tensor_stats("value", value)
        debug_tensor_stats("beta", beta)
        debug_tensor_stats("gate", gate)
        debug_tensor_stats("states", states)
        debug_tensor_stats("mask", mask)
        debug_tensor_stats("tril_mask", tril_mask)
        debug_tensor_stats("eye", eye)

        pypto_chunk_gated_delta_rule_fwd_dyn_clear_wo_l2norm_test_ablock(inputs, outputs)
        torch_npu.npu.synchronize()

        print(f"\n--- Round {r + 1} / {num_rounds} ---")
        debug_tensor_stats("a_block", outputs["a_block"])
        debug_tensor_stats("inverse_a_block_res", outputs["inverse_a_block_res"])
        debug_tensor_stats("core_attn_out", outputs["core_attn_out"])
        debug_tensor_stats("final_state", outputs["final_state"])
        # 分 chunk 统计 a_block / inverse_a_block_res
        for c in range(max_chunks):
            ab = outputs["a_block"][0, :, c, :, :]   # [Nv, L, L]
            inv = outputs["inverse_a_block_res"][0, :, c, :, :]
            ab_amax = ab.abs().max().item()
            ab_mean = ab.mean().item()
            inv_amax = inv.abs().max().item()
            inv_mean = inv.mean().item()
            inv_std = inv.std().item()
            bad = inv_amax > 1.1 or inv_mean < 0.0
            mark = " <-- bad" if bad else ""
            print(f"[GDN_BWD_DEBUG][rank0] chunk {c:2d}: a_block amax={ab_amax:.6f} mean={ab_mean:.2e} | inv amax={inv_amax:.4f} mean={inv_mean:.4f} std={inv_std:.4f}{mark}")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_pypto_autograd_perf_default_wo_l2norm()
