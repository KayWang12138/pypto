"""
DenseLightningIndexerSoftmaxLse PyPTO 实现 - 使用掩码实现完整因果注意力

核心思路：
1. 预先创建一个因果掩码矩阵 [T1, T2]
2. 掩码中，无效位置（需要被掩盖的位置）设为 True
3. 使用 pypto.where 将无效位置的 score 设为 -inf
4. 这样可以在一次 reduction 操作中正确计算 softmax max 和 sum
"""
import torch
import pypto
from typing import Optional


# 配置参数
B = 1
T1 = 32
T2 = 32
N1 = 8
N2 = 1
D = 128
G = N1 // N2
S1G = T1 * G


def create_causal_mask_kernel(
    row_indices: pypto.Tensor([T1], pypto.DT_FP32),
    col_indices: pypto.Tensor([T2], pypto.DT_FP32),
    mask: pypto.Tensor([T1, T2], pypto.DT_BOOL),
    next_tokens: int,
):
    """
    创建因果掩码的 kernel
    
    掩码逻辑：
    - 对于位置 (i, j)，如果 j > i + next_tokens，则需要掩盖（设为 True）
    - 否则保留（设为 False）
    
    这意味着：
    - next_tokens=0: 标准因果掩码，j > i 的位置被掩盖
    - next_tokens=N: 允许看到未来 N 个 token，j > i+N 的位置被掩盖
    """
    pypto.set_vec_tile_shapes(4, 16)
    
    # row_indices: [T1] -> [T1, 1]
    row_idx_2d = pypto.unsqueeze(row_indices, 1)
    
    # col_indices: [T2] -> [1, T2]
    col_idx_2d = pypto.unsqueeze(col_indices, 0)
    
    # 计算掩码条件：col_idx > row_idx + next_tokens
    # 由于广播，row_idx_2d [T1, 1] 和 col_idx_2d [1, T2] 会广播为 [T1, T2]
    threshold = row_idx_2d + next_tokens
    mask_cond = pypto.gt(col_idx_2d, threshold)
    
    # 将结果写入 mask
    mask[:] = mask_cond


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})
def dense_lightning_indexer_softmax_lse_kernel_with_mask(
    query: pypto.Tensor((T1, N1, D), pypto.DT_FP16),
    key: pypto.Tensor((T2, N2, D), pypto.DT_FP16),
    weights: pypto.Tensor((T1, N1), pypto.DT_FP16),
    s1_starts: pypto.Tensor((B,), pypto.DT_INT32),
    s1_ends: pypto.Tensor((B,), pypto.DT_INT32),
    s2_starts: pypto.Tensor((B,), pypto.DT_INT32),
    s2_ends: pypto.Tensor((B,), pypto.DT_INT32),
    softmax_max: pypto.Tensor((N2, T1), pypto.DT_FP32),
    softmax_sum: pypto.Tensor((N2, T1), pypto.DT_FP32),
    causal_mask: pypto.Tensor((T1, T2), pypto.DT_FP32),  # 改为 FP32 类型，无效位置为 -inf
    next_tokens: int,
):
    """
    使用掩码实现完整的因果注意力
    
    关键改进：
    - 使用预先计算的因果掩码（浮点类型，无效位置为负无穷）
    - 避免 PyPTO 不支持的动态长度 reduction 操作
    - 正确实现因果注意力逻辑
    """
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(1, S1G, T2)

    query_float = pypto.cast(query, pypto.DT_FP32)
    key_float = pypto.cast(key, pypto.DT_FP32)
    weights_float = pypto.cast(weights, pypto.DT_FP32)

    # ========== B 循环：遍历每个 batch ==========
    for b_idx in pypto.loop(0, B, 1, name="b_loop", idx_name="b"):
        s1_start = s1_starts[b_idx]
        s1_end = s1_ends[b_idx]
        s2_start = s2_starts[b_idx]
        s2_end = s2_ends[b_idx]
        
        s1_len = s1_end - s1_start
        s2_len = s2_end - s2_start
        
        # ========== N2 循环：遍历每个 key head ==========
        for n2_idx in pypto.loop(0, N2, 1, name="n2_loop", idx_name="n2"):
            g_start = n2_idx * G
            
            # 使用view提取当前batch和n2的数据
            q_view_shape = [T1, G, D]
            q_offsets = [s1_start, g_start, 0]
            q_valid_shape = [s1_len, G, D]
            q_view = pypto.view(query_float, q_view_shape, q_offsets, valid_shape=q_valid_shape)
            
            k_view_shape = [T2, 1, D]
            k_offsets = [s2_start, n2_idx, 0]
            k_valid_shape = [s2_len, 1, D]
            k_view = pypto.view(key_float, k_view_shape, k_offsets, valid_shape=k_valid_shape)
            
            w_view_shape = [T1, G]
            w_offsets = [s1_start, g_start]
            w_valid_shape = [s1_len, G]
            w_view = pypto.view(weights_float, w_view_shape, w_offsets, valid_shape=w_valid_shape)
            
            # 计算 score = Q @ K^T
            s1_len_g = s1_len * G
            q_2d = pypto.reshape(q_view, [T1 * G, D], valid_shape=[s1_len_g, D])
            k_2d = pypto.reshape(k_view, [T2, D], valid_shape=[s2_len, D])
            score = pypto.matmul(q_2d, k_2d, out_dtype=pypto.DT_FP32, b_trans=True)
            score = pypto.relu(score)
            
            # reshape: [s1_len * G, s2_len] -> [s1_len, G, s2_len]
            score_3d = pypto.reshape(score, [T1, G, T2], valid_shape=[s1_len, G, s2_len])
            
            # 权重乘法
            w_expanded = pypto.reshape(w_view, [T1, G, 1], valid_shape=[s1_len, G, 1])
            weighted = pypto.mul(score_3d, w_expanded)
            
            # 聚合 G 组 -> [s1_len, s2_len]
            res = pypto.sum(weighted, dim=1, keepdim=False)
            
            # ========== 应用因果掩码 ==========
            # 提取当前 batch 对应的掩码部分
            # 掩码是 FP32 类型，无效位置为 -1e9，有效位置为 0
            mask_view = pypto.view(causal_mask, [T1, T2], [s1_start, s2_start], 
                                   valid_shape=[s1_len, s2_len])
            
            # 将掩码加到 score 上
            masked_res = pypto.add(res, mask_view)
            
            # ========== 计算 softmax max 和 sum ==========
            # 现在可以安全地计算整个矩阵的 max 和 sum
            row_max = pypto.amax(masked_res, dim=-1, keepdim=True)  # [s1_len, 1]
            
            # 计算 exp(score - max)
            exp_scores = pypto.exp(pypto.sub(masked_res, row_max))
            
            # 计算 sum
            row_sum = pypto.sum(exp_scores, dim=-1, keepdim=False)  # [s1_len]
            row_max = pypto.reshape(row_max, [T1], valid_shape=[s1_len])  # [s1_len]
            
            # 使用 assemble 写入结果
            row_max_2d = pypto.reshape(row_max, [1, T1], valid_shape=[1, s1_len])
            row_sum_2d = pypto.reshape(row_sum, [1, T1], valid_shape=[1, s1_len])
            
            pypto.assemble(row_max_2d, [n2_idx, s1_start], softmax_max)
            pypto.assemble(row_sum_2d, [n2_idx, s1_start], softmax_sum)


def dense_lightning_indexer_softmax_lse_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    next_tokens: int = 2147483647,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden 参考实现"""
    T1, N1, D = query.shape
    T2, N2, _ = key.shape
    G = N1 // N2

    device = query.device
    query_float = query.float()
    key_float = key.float()
    weights_float = weights.float()

    softmax_max = torch.zeros(N2, T1, dtype=torch.float32, device=device)
    softmax_sum = torch.zeros(N2, T1, dtype=torch.float32, device=device)

    B = actual_seq_lengths_query.shape[0]
    actual_seq_q = torch.cat([torch.tensor([0], device=device), actual_seq_lengths_query])
    actual_seq_k = torch.cat([torch.tensor([0], device=device), actual_seq_lengths_key])

    for b in range(B):
        s1_start = actual_seq_q[b].item()
        s1_end = actual_seq_q[b + 1].item()
        s2_start = actual_seq_k[b].item()
        s2_end = actual_seq_k[b + 1].item()
        s1_len = s1_end - s1_start
        s2_len = s2_end - s2_start

        if s1_len == 0 or s2_len == 0:
            continue

        q_b = query_float[s1_start:s1_end]
        k_b = key_float[s2_start:s2_end]
        w_b = weights_float[s1_start:s1_end]

        for n2 in range(N2):
            q_heads = q_b[:, n2 * G:(n2 + 1) * G, :]
            k_head = k_b[:, n2, :]

            score = torch.einsum('qgd,kd->qgk', q_heads, k_head)
            score = torch.relu(score)

            w_heads = w_b[:, n2 * G:(n2 + 1) * G]
            weighted_score = score * w_heads.unsqueeze(-1)
            res = weighted_score.sum(dim=1)

            # 应用因果掩码
            for s1_idx in range(s1_len):
                valid_s2_len = min(s2_len, s1_idx + 1 + next_tokens)
                
                row = res[s1_idx, :valid_s2_len].clone()
                # 对无效位置填充 -inf
                if valid_s2_len < s2_len:
                    full_row = torch.full((s2_len,), float('-inf'), device=device)
                    full_row[:valid_s2_len] = row
                    row = full_row
                
                max_val = row.max().item()
                softmax_max[n2, s1_start + s1_idx] = max_val

                exp_vals = torch.exp(row - max_val)
                sum_val = exp_vals.sum().item()
                softmax_sum[n2, s1_start + s1_idx] = sum_val

    return softmax_max, softmax_sum


def test_with_mask():
    """测试使用掩码的版本"""
    import torch_npu
    import os

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 14))
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    print("=" * 60)
    print("Test: 使用掩码实现完整因果注意力")
    print("=" * 60)

    torch.manual_seed(42)
    t1, t2 = 16, 16
    next_tokens = 2  # 允许看到未来 2 个 token
    
    query = torch.randn(t1, N1, D, dtype=torch.float16, device=device) * 0.1
    key = torch.randn(t2, N2, D, dtype=torch.float16, device=device) * 0.1
    weights = torch.abs(torch.randn(t1, N1, dtype=torch.float16, device=device)) * 0.01 + 0.005

    s1_starts = torch.tensor([0], dtype=torch.int32, device=device)
    s1_ends = torch.tensor([t1], dtype=torch.int32, device=device)
    s2_starts = torch.tensor([0], dtype=torch.int32, device=device)
    s2_ends = torch.tensor([t2], dtype=torch.int32, device=device)
    
    softmax_max = torch.zeros(N2, T1, dtype=torch.float32, device=device)
    softmax_sum = torch.zeros(N2, T1, dtype=torch.float32, device=device)

    # 创建因果掩码（浮点类型）
    # 无效位置（需要掩盖）设为 -1e9，有效位置设为 0
    causal_mask = torch.zeros(T1, T2, dtype=torch.float32, device=device)
    
    # 在 CPU 上创建掩码
    for i in range(T1):
        for j in range(T2):
            if j > i + next_tokens:
                causal_mask[i, j] = -1e9  # 需要掩盖的位置
    
    print(f"掩码示例（前5行前5列），无效位置值为 -1e9：")
    print(causal_mask[:5, :5].cpu())
    print()

    # 调用 PyPTO kernel
    dense_lightning_indexer_softmax_lse_kernel_with_mask(
        query, key, weights, s1_starts, s1_ends, s2_starts, s2_ends, 
        softmax_max, softmax_sum, causal_mask, next_tokens
    )

    # Golden
    actual_seq_lengths_query = torch.tensor([t1], dtype=torch.int64, device=device)
    actual_seq_lengths_key = torch.tensor([t2], dtype=torch.int64, device=device)
    golden_max, golden_sum = dense_lightning_indexer_softmax_lse_golden(
        query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, next_tokens
    )

    max_diff_max = (softmax_max[:, :t1].cpu() - golden_max.cpu()).abs().max().item()
    max_diff_sum = (softmax_sum[:, :t1].cpu() - golden_sum.cpu()).abs().max().item()

    print(f"Max diff (softmax_max): {max_diff_max:.6f}")
    print(f"Max diff (softmax_sum): {max_diff_sum:.6f}")

    print(f"\nsoftmax_max (前10个):")
    print(softmax_max[0, :10].cpu())
    print(f"\ngolden_max (前10个):")
    print(golden_max[0, :10].cpu())

    assert max_diff_max < 1e-3 and max_diff_sum < 1e-3, "Test failed"
    print("\n✓ Test passed with causal mask!")


if __name__ == "__main__":
    test_with_mask()