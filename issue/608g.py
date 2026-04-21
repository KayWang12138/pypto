import torch
import pypto
import math
import os

os.environ["PTO_TILE_LIB_CODE_PATH"] = "/mnt/workspace/gitCode/cann/pto-isa"

def rms_norm_torch(
    x: torch.Tensor, gamma: torch.Tensor = None, epsilon: float = 1e-6, beta: torch.Tensor = None) -> torch.Tensor:
    # 计算每个样本的 RMS
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))  + epsilon

    # 归一化
    x_norm = x / rms

    # 如果有 gamma 和 beta，则进行缩放和平移
    if gamma is not None:
        x_norm = x_norm * gamma

    if beta is not None:
        x_norm = x_norm + beta

    return x_norm


def rms_norm_backward_golden(dy: torch.tensor, x: torch.tensor, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x_hat = x / rms

    d_gamma = (dy * x_hat).sum(dim=(0,1))

    dx = (gamma / rms) * (
        dy - x_hat * (dy * x_hat).mean(-1, keepdim=True)
    )

    return dx, d_gamma


def linear_backward_golden(dy, x, weight):
    """
    dy: [B, L, D_out] - 输出梯度
    x: [B, L, D_in]   - 输入
    weight: [D_in, D_out]

    返回：
        dx: [B, L, D_in] - 输入梯度
        dW: [D_in, D_out] - 权重梯度
        db: [D_out] - bias 梯度
    """
    # 输入梯度
    dx = dy @ weight.T  # [B,L,D_out] @ [D_out,D_in] -> [B,L,D_in]

    # 权重梯度
    B, L, D_in = x.shape
    D_out = dy.shape[2]
    x_flat = x.reshape(B*L, D_in)
    dy_flat = dy.reshape(B*L, D_out)
    dW = x_flat.T @ dy_flat   # [D_in,D_out]

    # bias 梯度
    db = dy.sum(dim=(0,1))    # [D_out]

    return dx, dW, db


def torch_engram_backward(
    grad_out, # [B, L, H, D]
    hidden_states, # [B, L, H, D]
    embeddings, # [B, L, D]
    key_proj_weights, # [H, D, D]
    value_proj_weights, # [D, D]
    key_gamma, # [H, D]
    query_gamma, # [H, D]
    key_lineared, # [B, L, H, D]
    value_lineared, # [B, L, D]
    gate_back, # [B, L, H, 1]
    score_output, # [B, L, H, 1]
):
    # 获取输入张量的形状
    B, L, H, D = hidden_states.shape  # B=batch, L=序列长度, H=头数, D=特征维度
    sqrt_D = math.sqrt(D)
    sqrt_eps = 1e-4

    # 初始化梯度张量
    d_hidden = torch.zeros_like(hidden_states)  # [B, L, H, D]
    d_embeddings = torch.zeros_like(embeddings)  # [B, L, D]

    d_key_w = torch.zeros_like(key_proj_weights)  # [H, D, D]
    d_key_b = torch.zeros(key_proj_weights.shape[0], D, device=embeddings.device)  # [H, D]

    d_val_w = torch.zeros_like(value_proj_weights)  # [D, D]
    d_val_b = torch.zeros(D, device=embeddings.device)  # [D]

    d_key_gamma = torch.zeros_like(key_gamma)  # [H, D]
    d_query_gamma = torch.zeros_like(query_gamma)  # [H, D]

    value = value_lineared  # [B, L, D]
    gates = gate_back  # [B, L, H, 1]
    scores = score_output.squeeze(-1)  # [B, L, H]

    # -----------------------------
    # Step1: value / gate 分支
    # -----------------------------
    # grad_out: [B, L, H, D], gates: [B, L, H, 1]
    d_value = (grad_out * gates).sum(dim=2)  # [B, L, D] - value的梯度

    # value: [B, L, D] -> unsqueeze: [B, L, 1, D], grad_out: [B, L, H, D]
    value_expanded = value.unsqueeze(2)  # [B, L, 1, D]
    d_gates = grad_out * value_expanded  # [B, L, H, D] - gates的梯度

    # -----------------------------
    # value linear backward
    # -----------------------------
    # d_value: [B, L, D], embeddings: [B, L, D], value_proj_weights: [D, D]
    dx_val, dW_val, db_val = linear_backward_golden(
        d_value, embeddings, value_proj_weights
    )
    # dx_val: [B, L, D] - 对embeddings的梯度（value分支）
    # dW_val: [D, D] - 对value投影权重的梯度
    # db_val: [D] - 对value投影偏置的梯度

    d_embeddings += dx_val  # [B, L, D]
    d_val_w += dW_val  # [D, D]
    d_val_b += db_val  # [D]

    # -----------------------------
    # Step2: 每个 head 反传
    # -----------------------------
    for hc_idx in range(H):
        key = key_lineared[:, :, hc_idx, :]  # [B, L, D]
        query = hidden_states[:,:,hc_idx,:]  # [B, L, D]

        gate = gates[:,:,hc_idx,:]  # [B, L, 1]
        score = scores[:,:,hc_idx]  # [B, L]

        # -------------------------
        # gate -> score
        # -------------------------
        d_gate = d_gates[:,:,hc_idx,:]  # [B, L, D]

        # 计算sigmoid导数: dσ/dz = σ(1-σ)
        # gate: [B, L, 1], dz: [B, L, D]
        dz = d_gate * gate * (1 - gate)  # [B, L, D]
        dz_sum = dz.sum(dim=-1)  # [B,L]
        # 计算f(s) = √(|s| + eps)的导数: df/ds = 0.5 / √(|s| + eps)
        abs_score = score.abs() + sqrt_eps  # [B, L]
        df_ds = 0.5 / torch.sqrt(abs_score)  # [B, L]

        # dz.sum(-1): [B, L], df_ds: [B, L], d_score: [B, L]
        d_score = dz_sum * df_ds  # [B, L]

        # -------------------------
        # recompute normed Q/K
        # -------------------------
        normed_key = rms_norm_torch(key, gamma=key_gamma[hc_idx])  # [B, L, D]
        normed_query = rms_norm_torch(query, gamma=query_gamma[hc_idx])  # [B, L, D]

        # -------------------------
        # score -> Qn / Kn
        # -------------------------
        # d_score: [B, L] -> unsqueeze: [B, L, 1]
        # normed_key: [B, L, D], d_query_normed: [B, L, D]
        d_query_normed = d_score.unsqueeze(-1) * normed_key / sqrt_D  # [B, L, D]

        # d_score: [B, L] -> unsqueeze: [B, L, 1]
        # normed_query: [B, L, D], d_key_normed: [B, L, D]
        d_key_normed = d_score.unsqueeze(-1) * normed_query / sqrt_D  # [B, L, D]

        # -------------------------
        # RMSNorm backward
        # -------------------------
        # d_key_normed: [B, L, D], key: [B, L, D], key_norm_weights[hc_idx]: [D]
        d_key, dk_gamma = rms_norm_backward_golden(
            d_key_normed, key, key_gamma[hc_idx]
        )
        # d_key: [B, L, D] - key的梯度
        # dk_gamma: [D] - key norm gamma的梯度

        d_hidden[:,:,hc_idx,:] += d_query_normed  # [B, L, D]
        d_key_gamma[hc_idx] += dk_gamma  # [D]

        # -------------------------
        # key linear backward
        # -------------------------
        # d_key: [B, L, D], embeddings: [B, L, D], key_proj_weights[hc_idx]: [D, D]
        dx_key, dW_key, db_key = linear_backward_golden(
            d_key, embeddings, key_proj_weights[hc_idx]
        )
        # dx_key: [B, L, D] - 对embeddings的梯度（key分支）
        # dW_key: [D, D] - 对key投影权重的梯度（第hc_idx个头）
        # db_key: [D] - 对key投影偏置的梯度（第hc_idx个头）

        d_embeddings += dx_key  # [B, L, D]
        d_key_w[hc_idx] += dW_key  # [D, D]
        d_key_b[hc_idx] += db_key  # [D]

    return (
        d_hidden,        # [B, L, H, D]
        d_embeddings,    # [B, L, D]
        d_key_w,         # [H, D, D]
        d_key_b,         # [H, D]
        d_val_w,         # [D, D]
        d_val_b,         # [D]
        d_key_gamma,     # [H, D]
    )


def engram_backward_golden(
    grad_out, # [B, L, H, D]
    hidden_states, # [B, L, H, D]
    embeddings, # [B, L, D]
    key_proj_weights, # [H, D, D]
    value_proj_weights, # [D, D]
    key_gamma, # [H, D]
    query_gamma, # [H, D]
    key_lineared, # [B, L, H, D]
    value_lineared, # [B, L, D]
    gate_back, # [B, L, H, 1]
    score_output, # [B, L, H, 1]
):
    return torch_engram_backward(
        grad_out,
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        key_lineared,
        value_lineared,
        gate_back,
        score_output
    )


def sign_sqrt(tensor, eps):
    """
    return tensor.abs().sqrt() * tensor.sign()
    """
    pos = pypto.ge(tensor, 0.0)
    if eps is None:
        tensor_sqrt = pypto.sqrt(pypto.abs(tensor))
    else:
        tensor_sqrt = pypto.sqrt(pypto.abs(tensor) + eps)
    neg_sqrt = pypto.neg(tensor_sqrt)
    return pypto.where(pos, tensor_sqrt, neg_sqrt)


def rms_norm_backward_module(dy, x, gamma, eps=1e-6):
    """
    dy: [tile, D]
    x: [tile, D] ->
    gamma: [D] ->
    """
    # 计算前向中间变量

    D = x.shape[-1]
    pypto.set_vec_tile_shapes(16, 1024)
    sum_x = pypto.sum(x * x * 1.0, -1, keepdim=True)
    pypto.set_vec_tile_shapes(64, 256)
    mean_x2 = sum_x / D
    rms = pypto.sqrt(mean_x2 + eps) # [tile, 1]
    x_hat = x / rms # [tile, 1]

    # 计算 d_gamma
    pypto.set_vec_tile_shapes(16, 1024)
    d_gamma = pypto.sum(dy * x_hat, 0) # [D]

    # 计算 dx
    pypto.set_vec_tile_shapes(16, 1024)
    dy_x_hat = pypto.sum(dy * x_hat, -1, keepdim=True) # [tile, 1]
    pypto.set_vec_tile_shapes(64, 256)
    dx = (dy - x_hat * dy_x_hat / D) / rms * gamma # [tile, D]
    return dx, d_gamma


def linear_backward_module(dy, x, weight):
    """
    dy: [tile, D] - 输出梯度
    x: [tile, D]   - 输入
    weight: [D, D]

    returns:
    dx: [L, D] - 输入梯度
    dW: [D, D] - 权重梯度
    db: [D] - bias 梯度
    """
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dx = pypto.matmul(dy, weight, pypto.DT_FP32, b_trans=True)
    dW = pypto.matmul(x, dy, pypto.DT_FP32, a_trans=True)
    pypto.set_vec_tile_shapes(16, 1024)
    db = pypto.sum(dy, 0)
    return dx, dW, db


@pypto.frontend.jit
def engram_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    hidden_states: pypto.Tensor([], pypto.DT_FP32),
    embeddings: pypto.Tensor([], pypto.DT_FP32),
    key_w: pypto.Tensor([], pypto.DT_FP32),
    value_w: pypto.Tensor([], pypto.DT_FP32),
    key_gamma: pypto.Tensor([], pypto.DT_FP32),
    query_gamma: pypto.Tensor([], pypto.DT_FP32),
    key_lineared: pypto.Tensor([], pypto.DT_FP32),
    value_lineared: pypto.Tensor([], pypto.DT_FP32),
    gate: pypto.Tensor([], pypto.DT_FP32),
    score: pypto.Tensor([], pypto.DT_FP32),
    d_hidden: pypto.Tensor([], pypto.DT_FP32),
    d_embeddings: pypto.Tensor([], pypto.DT_FP32),
    d_key_w: pypto.Tensor([], pypto.DT_FP32),
    d_key_b: pypto.Tensor([], pypto.DT_FP32),
    d_value_w: pypto.Tensor([], pypto.DT_FP32),
    d_value_b: pypto.Tensor([], pypto.DT_FP32),
    d_key_gamma: pypto.Tensor([], pypto.DT_FP32),
):
    # pypto.set_print_options(edgeitems=1024 * 1024, linewidth=200)
    B, L, H, D = hidden_states.shape  # B=batch, L=序列长度, H=头数, D=特征维度
    sqrt_D = math.sqrt(D)
    sqrt_eps = 1e-4
    eps = 1e-6
    unroll_list = [64]
    d_embeddings_tmp = pypto.tensor([B, L, H, D], d_embeddings.dtype, "d_embeddings_tmp")
    pypto.set_vec_tile_shapes(128, 128)
    d_value_w_tmp = pypto.full(size=[D, D], fill_value=0.0, dtype=d_value_w.dtype)
    pypto.set_vec_tile_shapes(1024)
    d_value_b_tmp = pypto.full(size=[D], fill_value=0.0, dtype=d_value_b.dtype)
    for h_idx in range(H):
        pypto.set_vec_tile_shapes(1024)
        d_key_gamma_tmp = pypto.full(size=[D], fill_value=0.0, dtype=d_key_gamma.dtype)
        d_key_b_tmp = pypto.full(size=[D], fill_value=0.0, dtype=d_key_b.dtype)
        pypto.set_vec_tile_shapes(128, 128)
        d_key_w_tmp = pypto.full(size=[D, D], fill_value=0.0, dtype=d_key_w.dtype)
        for b_idx in pypto.loop(0, B, 1):
            for l_idx, tile in pypto.loop_unroll(0, L, 1, unroll_list=unroll_list):
                pypto.set_vec_tile_shapes(1, 64, 256)
                value_lineared_view_2d = value_lineared[b_idx, l_idx:l_idx+tile] # [tile, D]
                embeddings_view_2d = embeddings[b_idx, l_idx:l_idx+tile] # [tile, D]

                pypto.set_vec_tile_shapes(1, 64, 1, 256)
                grad_out_view = grad_out[b_idx, l_idx:l_idx+tile, h_idx] # [tile, D]
                gate_view = gate[b_idx, l_idx:l_idx+tile, h_idx] # [tile, 1]
                score_view = score[b_idx, l_idx:l_idx+tile, h_idx] # [tile, 1]
                key_lineared_view = key_lineared[b_idx, l_idx:l_idx+tile, h_idx] # [tile, D]
                query_view = hidden_states[b_idx, l_idx:l_idx+tile, h_idx] # [tile, D]

                pypto.set_vec_tile_shapes(64, 256)
                d_value = grad_out_view * gate_view # [tile, D]
                d_gate = grad_out_view * value_lineared_view_2d # [tile, D]
                pypto.set_vec_tile_shapes(16, 1024)
                d_gate_sum = pypto.sum(d_gate, -1, keepdim=True) # [tile, 1]
                pypto.set_vec_tile_shapes(64, 256)
                d_gate_input = d_gate_sum * gate_view * (pypto.neg(gate_view) + 1) # [tile, 1]
                abs_score = pypto.abs(score_view) + sqrt_eps # [tile]
                d_score = d_gate_input / (pypto.sqrt(abs_score) * 2.0) # [tile, 1]
                key_gamma_view = key_gamma[h_idx] # [D]
                query_gamma_view = query_gamma[h_idx] # [D]
                pypto.set_vec_tile_shapes(16, 1024)
                key_normed = pypto.rms_norm(key_lineared_view, key_gamma_view, eps) # [tile, D]
                query_normed = pypto.rms_norm(query_view, query_gamma_view, eps) # [tile, D]

                pypto.set_vec_tile_shapes(64, 256)
                d_query_normed = d_score * key_normed / sqrt_D # [tile, D]
                d_key_normed = d_score * query_normed / sqrt_D # [tile, D]
                d_key_lineared_view, d_key_gamma_view = rms_norm_backward_module(
                    d_key_normed, key_lineared_view, key_gamma_view) # [tile, D], [D]
                pypto.set_vec_tile_shapes(1, 64, 1, 256)
                d_hidden[b_idx, l_idx:l_idx+tile, h_idx] = d_query_normed
                pypto.set_vec_tile_shapes(1024)
                d_key_gamma_tmp[:] = d_key_gamma_tmp + d_key_gamma_view # 每个tile累加一次

                d_key_view, d_key_w_view, d_key_b_view = linear_backward_module(
                    d_key_lineared_view, embeddings_view_2d, key_w[h_idx]) # [tile, D], [D, D], [D]
                # ! 累加线性梯度
                pypto.set_vec_tile_shapes(128, 128)
                d_key_w_tmp[:] = d_key_w_tmp + d_key_w_view # [D, D]

                pypto.set_vec_tile_shapes(1024)
                d_key_b_tmp[:] = d_key_b_tmp + d_key_b_view
                d_value_view, d_value_w_view, d_value_b_view = linear_backward_module(
                    d_value, embeddings_view_2d, value_w) # [tile, D], [D, D], [D]
                pypto.set_vec_tile_shapes(64, 256)
                d_k_v_view = d_key_view + d_value_view # [tile, D]
                # pypto.pass_verify_print("d_k_v_view\n", d_k_v_view)
                d_value_w_tmp[:] = d_value_w_tmp + d_value_w_view
                pypto.set_vec_tile_shapes(1024)
                d_value_b_tmp[:] = d_value_b_tmp + d_value_b_view
                pypto.set_vec_tile_shapes(1, 64, 1, 512)
                d_embeddings_tmp[b_idx, l_idx:l_idx+tile, h_idx] = d_k_v_view

        pypto.set_vec_tile_shapes(1, 1024)
        d_key_gamma[h_idx] = d_key_gamma_tmp
        d_key_b[h_idx] = d_key_b_tmp
        pypto.set_vec_tile_shapes(1, 128, 128)
        d_key_w[h_idx] = d_key_w_tmp

    for b_idx in pypto.loop(0, B, 1, name="b_loop_1"):
        for l_idx, tile in pypto.loop_unroll(0, L, 1, name="l_loop_2", unroll_list=unroll_list):
            pypto.set_vec_tile_shapes(1, 64, 1, 256)
            d_embeddings_view = d_embeddings_tmp[b_idx, l_idx:l_idx+tile] # [tile, H, D]
            pypto.set_vec_tile_shapes(64, 1, 256)
            d_embeddings_view_summed = pypto.sum(d_embeddings_view, 1) # [tile, D]
            pypto.set_vec_tile_shapes(1, 64, 512)
            d_embeddings[b_idx, l_idx: l_idx+tile] = d_embeddings_view_summed

    pypto.set_vec_tile_shapes(128, 128)
    pypto.assemble(d_value_w_tmp, [0, 0], d_value_w)

    pypto.set_vec_tile_shapes(1024)
    pypto.assemble(d_value_b_tmp, [0], d_value_b)


def engram_backward_pto(
    grad_out, # [B, L, H, D]
    hidden_states, # [B, L, H, D]
    embeddings, # [B, L, D]
    key_w, # [H, D, D]
    value_w, # [D, D]
    key_gamma, # [H, D]
    query_gamma, # [H, D]
    key_lineared, # [B, L, H, D]
    value_lineared, # [B, L, D]
    gate, # [B, L, H, 1]
    score, # [B, L, H, 1]
):
    B, L, H, D = hidden_states.shape
    device = grad_out.device
    d_hidden = torch.zeros_like(hidden_states)
    d_embeddings = torch.zeros([B, L, D], dtype=torch.float32, device=device)
    d_key_w = torch.zeros_like(key_w)
    d_key_b = torch.zeros([H, D], dtype=torch.float32, device=device)
    d_value_w = torch.zeros_like(value_w)
    d_value_b = torch.zeros([D], dtype=torch.float32, device=device)
    d_key_gamma = torch.zeros_like(key_gamma)

    engram_backward_kernel(
        grad_out, hidden_states, embeddings, key_w, value_w,
        key_gamma, query_gamma, key_lineared, value_lineared,
        gate, score,
        d_hidden, d_embeddings, d_key_w, d_key_b,
        d_value_w, d_value_b, d_key_gamma
    )
    return d_hidden, d_embeddings, d_key_w, d_key_b, d_value_w, d_value_b, d_key_gamma


def detailed_tensor_compare(
    tensor1, tensor2, rtol=1e-3, atol=1e-3, verbose=True, max_outliers_display=20
):
    """
    Detailed tensor comparison.

    Parameters
    ---------
    tensor1: first tensor
    tensor2: second tensor
    rtol: relative tolerance
    atol: absolute tolerance
    verbose: print details
    max_outliers_display: Maximum number of elements exceeding tolerance displayed

    Return
    ---------
    dict: dictionary containing the comparison results
    """
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # calculate differences
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)  # avoid division by zero

    # tolerance check
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # collect information statistics
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # count differences
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    # count differences out of tolerance
    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()

        # obtain the index and value out of tolerance
        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        # sort by the difference (from largest to smallest)
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
        "total_elements": total_elements,
        "out_of_tolerance_count": out_of_tolerance_count,
        "out_of_tolerance_ratio": out_of_tolerance_ratio,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_out_of_tolerance_diff": max_out_diff,
        "mean_out_of_tolerance_diff": mean_out_diff,
        "all_close": out_of_tolerance_count == 0,
        "tolerance_mask": tolerance_mask,
        "diff_tensor": diff,
        "outlier_indices": sorted_outlier_indices,
        "outlier_values1": sorted_outlier_values1,
        "outlier_values2": sorted_outlier_values2,
        "outlier_diffs": sorted_outlier_diffs,
        "outlier_relative_diffs": sorted_outlier_relative_diffs,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("📊 Detailed tensor comparison report")
        print("=" * 60)
        print(f"Total elements: {total_elements:,}")
        print(f"Out of tolerance count: {out_of_tolerance_count:,}")
        print(
            f"Out of tolerance ratio: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio*100:.4f}%)"
        )
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Standard difference: {std_diff:.6f}")
        print(f"Relative tolerance: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"Maximum difference exceeding the tolerance: {max_out_diff:.6f}")
            print(f"Mean difference exceeding the tolerance: {mean_out_diff:.6f}")

            # display out of tolerance details
            print(
                f"\nDetails of elements out of tolerance (displaying the first {min(max_outliers_display, out_of_tolerance_count)} elements):"
            )
            print("-" * 80)
            print(f"{'Index':<20} {'Tensor1':<15} {'Tensor2':<15} {'Absolute tolerance':<12} {'Relative tolerance':<12}")
            print("-" * 80)

            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices))))
                print(f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} {sorted_outlier_values2[i].item():<15.6f} "
                      f"{sorted_outlier_diffs[i].item():<12.6f} {sorted_outlier_relative_diffs[i].item():<12.6f}")

            if out_of_tolerance_count > max_outliers_display:
                print(f"... There are {out_of_tolerance_count - max_outliers_display} out of tolerance elements not shown.")

        print(f"\n✅ Tensor Matching: {result['all_close']}")
        print("=" * 60)

    return result


def test_engram_backward(B, L, D, H=1):
    device_id = 0
    torch.npu.set_device(device_id)
    device_str = f"npu:{device_id}"
    torch.manual_seed(42)
    grad_out = torch.randn(B, L, H, D, dtype=torch.float32, device=device_str)
    hidden_states = torch.randn(B, L, H, D, dtype=torch.float32, device=device_str)
    embeddings = torch.randn(B, L, D, dtype=torch.float32, device=device_str)
    key_w = torch.rand([H, D, D], dtype=torch.float32, device=device_str)
    value_w = torch.rand([D, D], dtype=torch.float32, device=device_str)
    key_gamma = torch.rand([H, D], dtype=torch.float32, device=device_str)
    query_gamma = torch.rand([H, D], dtype=torch.float32, device=device_str)
    key_lineared = torch.randn(B, L, H, D, dtype=torch.float32, device=device_str)
    value_lineared = torch.randn(B, L, D, dtype=torch.float32, device=device_str)
    gate_back = torch.rand(B, L, H, 1, dtype=torch.float32, device=device_str)
    score_output = torch.randn(B, L, H, 1, dtype=torch.float32, device=device_str)
    inputs = [
        grad_out,
        hidden_states,
        embeddings,
        key_w,
        value_w,
        key_gamma,
        query_gamma,
        key_lineared,
        value_lineared,
        gate_back,
        score_output
    ]
    inputs_cloned = [t.clone() for t in inputs]
    golden_outputs = [
        d_hidden_golden,        # [B, L, H, D]
        d_embeddings_golden,    # [B, L, D]
        d_key_w_golden,         # [H, D, D]
        d_key_b_golden,         # [H, D]
        d_val_w_golden,         # [D, D]
        d_val_b_golden,         # [D]
        d_key_gamma_golden,     # [H, D]
    ] = engram_backward_golden(*inputs_cloned)
    print("finish golden")
    goldens = [None] * len(inputs) + list(golden_outputs)
    goldens_cpu = [g.cpu() if g is not None else None for g in goldens]
    pypto.set_verify_golden_data(goldens=goldens_cpu)
    pto_outputs = [
        d_hidden_pto,        # [B, L, H, D]
        d_embeddings_pto,    # [B, L, D]
        d_key_w_pto,         # [H, D, D]
        d_key_b_pto,         # [H, D]
        d_val_w_pto,         # [D, D]
        d_val_b_pto,         # [D]
        d_key_gamma_pto,     # [H, D]
    ] = engram_backward_pto(*inputs)
    for i, (golden, pto) in enumerate(zip(golden_outputs, pto_outputs)):
        print(f"==== check {i}")
        detailed_tensor_compare(golden, pto)


def run_engram_backward_pto_test_cases():
    for D in [512, 768, 1024]:
        for B in [1, 8, 16]:
            for l in [1, 2, 4, 8]:
                L = l * 1024
                print(f"==== test {B}, {L}, {D}")
                test_engram_backward(B, L, D)


if __name__ == '__main__':
    #run_engram_backward_pto_test_cases()
    test_engram_backward(1, 1024 * 8, 12)