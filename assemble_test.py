import torch
import pypto
from numpy.testing import assert_allclose
import torch.nn.functional as F
import math

verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_pass_filter": ["RemoveRedundantReshape"]
}


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
    pypto.set_vec_tile_shapes(128, 128)
    D = x.shape[-1]
    mean_x2 = pypto.sum(x * x * 1.0, -1, keepdim=True) / D
    rms = pypto.sqrt(mean_x2 + eps) # [tile, 1]
    x_hat = x / rms # [tile, 1]

    # 计算 d_gamma
    pypto.set_vec_tile_shapes(128, 128)
    d_gamma = pypto.sum(dy * x_hat, 0) # [D]

    # 计算 dx
    pypto.set_vec_tile_shapes(128, 128)
    dy_x_hat = pypto.sum(dy * x_hat, -1, keepdim=True) # [tile, 1]
    pypto.set_vec_tile_shapes(128, 128)
    dx = (dy - x_hat * dy_x_hat / D) / rms * gamma # [tile, D]
    return dx, d_gamma


@pypto.jit(
    # debug_options={"runtime_debug_mode": 1},
    # verify_options=verify_options
)
def rms_norm_backward_kernel(dy, x, gamma, dx, dgamma):
    B = dy.shape[0]
    pypto.set_vec_tile_shapes(128)
    dgamma_tmp = pypto.full(dgamma.shape, 0.0, dtype=dgamma.dtype)
    for b_idx in pypto.loop(0, B, 1):
        dy_view = dy[b_idx]
        x_view= x[b_idx]
        dx_inner, dgamma_inner = rms_norm_backward_module(dy_view, x_view, gamma)
        pypto.set_vec_tile_shapes(128)
        dgamma_tmp = dgamma_tmp + dgamma_inner
        pypto.set_vec_tile_shapes(1, 128, 128)
        dx[b_idx:b_idx+1] = dx_inner.unsqueeze(0)
    pypto.set_vec_tile_shapes(128)
    pypto.assemble(dgamma_tmp, [0], dgamma)


def rms_norm_backward_pto(dy, x, gamma):
    dx = torch.zeros_like(x)
    dgamma = torch.zeros_like(gamma)
    inputs = [dy, x, gamma, dx, dgamma]
    inputs = [pypto.from_torch(t) for t in inputs]
    rms_norm_backward_kernel(*inputs)
    return dx, dgamma


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
    pypto.set_vec_tile_shapes(128, 128)
    db = pypto.sum(dy, 0)
    return dx, dW, db


@pypto.jit(
    # debug_options={"runtime_debug_mode": 1},
    # verify_options=verify_options
)
def linear_backward_kernel(dy, x, weight, dx, dW, db):
    B = dy.shape[0]
    for b_idx in pypto.loop(0, B, 1):
        pypto.set_vec_tile_shapes(1, 128, 128)
        dy_view = dy[b_idx]
        x_view= x[b_idx]
        dx_inner, dW_inner, db_inner = linear_backward_module(dy_view, x_view, weight)
        pypto.set_vec_tile_shapes(1, 128, 128)
        dx[b_idx:b_idx+1] = dx_inner.unsqueeze(0)
        dW[:] = dW + dW_inner
        db[:] = db + db_inner


def linear_backward_pto(dy, x, weight):
    D = x.shape[-1]
    dx = torch.zeros_like(x)
    dW = torch.zeros_like(weight)
    db = torch.zeros(D, dtype=torch.float32, device=x.device)
    inputs = [dy, x, weight, dx, dW, db]
    inputs = [pypto.from_torch(t) for t in inputs]
    linear_backward_kernel(*inputs)
    return dx, dW, db



def gated_attention_backward_module(
    grad_output, # [tile, D]
    gate, # [tile, 1]
    query_normed, # [tile, D]
    key_normed, # [tile, D]
    value_lineared, # [tile, D]
    score, # [tile]
    sqrt_eps,
    sqrt_D):
    """
    TODO value能否和qk一起求梯度
    """
    pypto.set_vec_tile_shapes(64, 128)
    d_value = grad_output * gate # [tile, D]
    d_gate_sum = pypto.sum(grad_output * value_lineared, -1, keepdim=True) # [tile, 1]
    d_gate_input = d_gate_sum * gate * (pypto.neg(gate) + 1) # [tile, 1]
    abs_score = pypto.abs(score) + sqrt_eps # [tile]
    abs_score_3d = abs_score.unsqueeze(-1) # [tile, 1]
    d_score = d_gate_input / (pypto.sqrt(abs_score_3d) * 2) # [tile, 1]
    d_query = d_score * key_normed / sqrt_D # [tile, D]
    d_key = d_score * query_normed / sqrt_D # [tile, D]
    return d_query, d_key, d_value


@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 32,
        # "stitch_function_num_initial": 512,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 32,
    },
    # debug_options={"runtime_debug_mode": 1},
    # verify_options=verify_options
)
def gated_attention_backward_kernel(
    key_normed,
    query_normed,
    value_lineared,
    grad_output,
    gate,
    score,
    d_query,
    d_key,
    d_value,
    ):
    """
    # inputs:
    - key_normed: [B, L, D]
    - query_normed: [B, L, D]
    - value_lineared: [B, L, D]
    - grad_output: [B, L, D]
    - gate: [B, L, 1]
    - score: [B, L]
    ------
    # outputs:
    - d_query: [B, L, D]
    - d_key: [B, L, D]
    - d_value: [B, L, D]
    """
    B, L, D = key_normed.shape
    sqrt_D = math.sqrt(D)
    sqrt_eps = 1e-4
    unroll_list = [16]
    for b_idx in pypto.loop(0, B, 1):
        for l_idx, unroll_length in pypto.loop_unroll(0, L, 1, unroll_list=unroll_list):
            tile = unroll_length
            # view
            pypto.set_vec_tile_shapes(1, 64, 128)
            query_normed_view = query_normed[b_idx, l_idx:l_idx+tile] # [tile, D]
            key_normed_view = key_normed[b_idx, l_idx:l_idx+tile] # [tile, D]
            value_lineared_view = value_lineared[b_idx, l_idx:l_idx+tile] # [tile, D]
            grad_output_view = grad_output[b_idx, l_idx:l_idx+tile] # [tile, D]
            gate_view = gate[b_idx, l_idx:l_idx+tile] # [tile, 1]
            score_view = score[b_idx, l_idx:l_idx+tile]
            # compute
            d_query_tmp, d_key_tmp, d_value_tmp = gated_attention_backward_module(
                grad_output_view,
                gate_view,
                query_normed_view,
                key_normed_view,
                value_lineared_view,
                score_view,
                sqrt_eps,
                sqrt_D
            )
            # copyout
            pypto.set_vec_tile_shapes(1, 64, 128)
            d_value[b_idx, l_idx:l_idx+tile, :] = d_value_tmp
            d_query[b_idx, l_idx:l_idx+tile, :] = d_query_tmp
            d_key[b_idx, l_idx:l_idx+tile, :] = d_key_tmp


def gated_attention_backward_pto(
    key,
    query,
    value,
    grad_output,
    gate,
    score):
    """
    # params:
    - key: [B, L, D]
    - query: [B, L, D]
    - value: [B, L, D]
    - grad_output: [B, L, D]
    - gate: [B, L, 1]
    - score: [B, L]
    ------
    # returns:
    - grad_query: [B, L, D]
    - grad_key: [B, L, D]
    - grad_value: [B, L, D]
    """

    device = key.device
    B, L, D = key.shape
    # ! create output tensors
    grad_query = torch.zeros([B, L, D], device=device).float()
    grad_key = torch.zeros([B, L, D], device=device).float()
    grad_value = torch.zeros([B, L, D], device=device).float()
    input_tensors = {
        key: [0, 1],
        query: [0, 1],
        value: [0, 1],
        grad_output: [0, 1],
        gate: [0, 1],
        score: [0, 1],
    }
    outputs_tensors = {
        grad_query: [0, 1],
        grad_key: [0, 1],
        grad_value: [0, 1],
    }
    input_tensors = [pypto.from_torch(t, dynamic_axis=axis) for t, axis in input_tensors.items()]
    output_tensors = [pypto.from_torch(t, dynamic_axis=axis) for t, axis in outputs_tensors.items()]
    gated_attention_backward_kernel(*input_tensors, *output_tensors)
    return grad_query, grad_key, grad_value


def gated_attention_backward_golden(
    key,
    query,
    value,
    grad_output,
    gate,
    score
    ):
    """
    # inputs:
    - key: [B, L, D]
    - query: [B, L, D]
    - value: [B, L, D]
    - grad_output: [B, L, D]
    - gate: [B, L, 1]
    - score: [B, L]
    ------
    # outputs:
    - grad_query: [B, L, D]
    - grad_key: [B, L, D]
    - grad_value: [B, L, D]
    """
    B, L, D = key.shape
    sqrt_D = math.sqrt(D)
    eps = 1e-4

    grad_value = grad_output * gate # [B, L, D]
    grad_gate = torch.sum(grad_output * value, dim=-1, keepdim=True) # [B, L, 1]
    grad_gate_input = grad_gate * gate * (1 - gate) # [B, L, 1]
    # scores_sqrt = score.abs().clamp_min(eps).sqrt().unsqueeze(-1) # [B, L]
    scores_sqrt = (score.abs() + eps).sqrt().unsqueeze(-1) # [B, L]
    # sign = score.sign().unsqueeze(-1)
    # grad_score = grad_gate_input * sign / (2 * scores_sqrt) # [B, L, 1]
    grad_score = grad_gate_input / (2 * scores_sqrt) # [B, L, 1]
    grad_key = grad_score * query / sqrt_D
    grad_query = grad_score * key / sqrt_D
    return grad_query, grad_key, grad_value


def test_gated_attention_backward(B, L, D):
    device_id = 1
    torch.npu.set_device(device_id)
    print(f"set {device_id=}")
    key = torch.rand([B, L, D], device=f"npu:{device_id}", dtype=torch.float32)
    query = torch.rand([B, L, D], device=f"npu:{device_id}", dtype=torch.float32)
    value = torch.rand([B, L, D], device=f"npu:{device_id}", dtype=torch.float32)
    grad_output = torch.rand([B, L, D], device=f"npu:{device_id}", dtype=torch.float32)
    gate = torch.rand([B, L, 1], device=f"npu:{device_id}", dtype=torch.float32)
    score = torch.rand([B, L], device=f"npu:{device_id}", dtype=torch.float32)
    grad_query_golden, grad_key_golden, grad_value_golden = gated_attention_backward_golden(
        key.clone(),
        query.clone(),
        value.clone(),
        grad_output.clone(),
        gate.clone(),
        score.clone(),
    )
    print("finish golden")
    # pypto.set_verify_golden_data(
    #     goldens=[None, None, None, None, None, None, grad_query_golden, grad_key_golden, grad_value_golden])
    grad_query, grad_key, grad_value = gated_attention_backward_pto(
        key,
        query,
        value,
        grad_output,
        gate,
        score,
    )

    detailed_tensor_compare(grad_query_golden, grad_query)
    detailed_tensor_compare(grad_key_golden, grad_key)
    detailed_tensor_compare(grad_value_golden, grad_value)


def run_gated_attention_test_cases():
    for D in [512, 768, 1024]:
        for B in [1, 8, 16]:
            for l in [1, 2, 4, 8]:
                L = l * 1024
                for i in range(5):
                    test_gated_attention_backward(B, L, D)


@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128*8,
        # "stitch_function_num_initial": 512,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128*8,
    },
    # debug_options={"runtime_debug_mode": 1},
    verify_options=verify_options
)
def engram_backward_kernel(
    # ! inputs
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
    # ! outputs
    d_hidden,        # [B, L, H, D]
    d_embeddings,    # [B, L, 1, D]
    d_key_w,         # [H, D, D]
    d_key_b,         # [H, D]
    d_value_w,         # [D, D]
    d_value_b,         # [D]
    d_key_gamma,     # [H, D]
):
    pypto.set_print_options(edgeitems=1024 * 1024, linewidth=200)
    B, L, H, D = hidden_states.shape  # B=batch, L=序列长度, H=头数, D=特征维度
    sqrt_D = math.sqrt(D)
    sqrt_eps = 1e-4
    eps = 1e-6
    unroll_list = [16]
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
        for b_idx in pypto.loop(0, B, 1, name="b_loop"):
            for l_idx, tile in pypto.loop_unroll(0, L, 1, name="l_loop", unroll_list=unroll_list):
                pypto.set_vec_tile_shapes(1, 64, 512)
                value_lineared_view_2d = value_lineared[b_idx, l_idx:l_idx+tile] # [tile, D]
                embeddings_view_2d = embeddings[b_idx, l_idx:l_idx+tile] # [1, tile, D]

                pypto.set_vec_tile_shapes(1, 64, 1, 512)
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
                pypto.set_vec_tile_shapes(1, 64, 1, 512)
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
                pypto.pass_verify_print("d_k_v_view\n", d_k_v_view)
                d_value_w_tmp[:] = d_value_w_tmp + d_value_w_view
                pypto.set_vec_tile_shapes(1024)
                d_value_b_tmp[:] = d_value_b_tmp + d_value_b_view
                pypto.set_vec_tile_shapes(1, 64, 1, 512)
                d_embeddings_tmp[b_idx, l_idx:l_idx+tile, h_idx] = d_k_v_view
                pypto.pass_verify_print("d_embeddings_tmp\n", d_embeddings_tmp)

        pypto.set_vec_tile_shapes(1, 1024)
        d_key_gamma[h_idx] = d_key_gamma_tmp
        d_key_b[h_idx] = d_key_b_tmp
        pypto.set_vec_tile_shapes(1, 128, 128)
        d_key_w[h_idx] = d_key_w_tmp

    for b_idx in pypto.loop(0, B, 1, name="b_loop_1"):
        for l_idx, tile in pypto.loop_unroll(0, L, 1, name="l_loop_2", unroll_list=unroll_list):
            pypto.set_vec_tile_shapes(1, 64, 1, 256)
            d_embeddings_view = d_embeddings_tmp[b_idx, l_idx:l_idx+tile] # [tile, H, D]
            # pypto.pass_verify_print("d_embeddings_tmp\n", d_embeddings_tmp)
            pypto.set_vec_tile_shapes(64, 1, 256)
            # pypto.pass_verify_print("d_embeddings_view\n", d_embeddings_view)
            d_embeddings_view_summed = pypto.sum(d_embeddings_view, 1) # [tile, D]

            pypto.set_vec_tile_shapes(64, 256)
            # pypto.pass_verify_print("d_embeddings_view_summed\n", d_embeddings_view_summed)

            pypto.set_vec_tile_shapes(1, 64, 512)
            d_embeddings[b_idx, l_idx: l_idx+tile] = d_embeddings_view_summed
            # pypto.pass_verify_print(d_embeddings)

    # pypto.set_vec_tile_shapes(1, 64, 512)
    # pypto.pass_verify_print(d_embeddings)

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

    # inputs = {
    #     grad_out: [0, 1],
    #     hidden_states: [0, 1],
    #     embeddings: [0, 1],
    #     key_w: [],
    #     value_w: [],
    #     key_gamma: [],
    #     query_gamma: [],
    #     key_lineared: [0, 1],
    #     value_lineared: [0, 1],
    #     gate: [0, 1],
    #     score: [0, 1],
    # }
    # outputs = {
    #     d_hidden: [0, 1],
    #     d_embeddings: [0, 1],
    #     d_key_w: [],
    #     d_key_b: [],
    #     d_value_w: [],
    #     d_value_b: [],
    #     d_key_gamma: [],
    # }

    inputs = {
        grad_out: [],
        hidden_states: [],
        embeddings: [],
        key_w: [],
        value_w: [],
        key_gamma: [],
        query_gamma: [],
        key_lineared: [],
        value_lineared: [],
        gate: [],
        score: [],
    }
    outputs = {
        d_hidden: [],
        d_embeddings: [],
        d_key_w: [],
        d_key_b: [],
        d_value_w: [],
        d_value_b: [],
        d_key_gamma: [],
    }

    pto_inputs = [pypto.from_torch(t, dynamic_axis=axis) for t, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(t, dynamic_axis=axis) for t, axis in outputs.items()]

    engram_backward_kernel(*pto_inputs, *pto_outputs)
    return d_hidden, d_embeddings, d_key_w, d_key_b, d_value_w, d_value_b, d_key_gamma


def test_engram_backward(B, L, D, H=1):
    device_id = 2
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
    # golden_outputs = [
    #     d_hidden_golden,        # [B, L, H, D]
    #     d_embeddings_golden,    # [B, L, D]
    #     d_key_w_golden,         # [H, D, D]
    #     d_key_b_golden,         # [H, D]
    #     d_val_w_golden,         # [D, D]
    #     d_val_b_golden,         # [D]
    #     d_key_gamma_golden,     # [H, D]
    # ] = engram_backward_golden(*inputs_cloned)
    print("finish golden")
    # goldens = [None] * len(inputs) + list(golden_outputs)
    # goldens_cpu = [g.cpu() if g is not None else None for g in goldens]
    # pypto.set_verify_golden_data(goldens=goldens_cpu)
    pto_outputs = [
        d_hidden_pto,        # [B, L, H, D]
        d_embeddings_pto,    # [B, L, D]
        d_key_w_pto,         # [H, D, D]
        d_key_b_pto,         # [H, D]
        d_val_w_pto,         # [D, D]
        d_val_b_pto,         # [D]
        d_key_gamma_pto,     # [H, D]
    ] = engram_backward_pto(*inputs)
    # for i, (golden, pto) in enumerate(zip(golden_outputs, pto_outputs)):
    #     print(f"==== check {i}")
    #     detailed_tensor_compare(golden, pto)


def run_engram_backward_pto_test_cases():
    for D in [512, 768, 1024]:
        for B in [1, 8, 16]:
            for l in [1, 2, 4, 8]:
                L = l * 1024
                for i in range(5):
                    test_engram_backward(B, L, D)


if __name__ == '__main__':
    # run_engram_backward_pto_test_cases()
    test_engram_backward(1, 64, 16)