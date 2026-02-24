#!/usr/bin/env python3
"""
分析 special_view_opcomposite_torch 与 special_view_opcomposite_torch2 结果不一致的原因。

结论（简要）：
- v1：对整个 (batch*seq, hidden) 做一次划分，按「全局」前一半行、后一半行乘 3.0 / 5.0。
- v2：按 tile 循环，每个 tile 内按「该 tile 的」前一半行、后一半行乘 3.0 / 5.0。
因此「前半行 / 后半行」的划分范围不同，导致 3.0/5.0 作用在不同位置，结果不一致。
本脚本在相同输入下对比两版输出，并标出差异位置与统计信息。
"""

import torch
from dataclasses import dataclass


@dataclass
class CompositeParams:
    batch_size: int
    seq_len: int
    hidden_size: int
    tile_b: int
    tile_s: int
    input_datarange: list
    dtype: str
    cube_tile_shapes: list
    vector_tile_shapes_2d: list
    vector_tile_shapes_3d: list


def special_view_opcomposite_torch(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: CompositeParams):
    input_a_view_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_a_1 = input_a_view_2d[:input_a_view_2d.shape[0] // 2, :input_a_view_2d.shape[-1] // 2]
    input_tensor_a_2 = input_a_view_2d[:input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] // 2 : input_a_view_2d.shape[-1]]
    input_tensor_a_3 = input_a_view_2d[input_a_view_2d.shape[0] // 2 : input_a_view_2d.shape[0], :input_a_view_2d.shape[-1] // 2]
    input_tensor_a_4 = input_a_view_2d[input_a_view_2d.shape[0] // 2 : input_a_view_2d.shape[0], input_a_view_2d.shape[-1] // 2 : input_a_view_2d.shape[-1]]

    tensor_a_tmp_dim1_1 = torch.concat([input_tensor_a_1, input_tensor_a_2], dim=-1)
    tensor_a_tmp_dim1_2 = torch.concat([input_tensor_a_3, input_tensor_a_4], dim=-1)
    tmp_dim1_a_mul_1 = torch.mul(tensor_a_tmp_dim1_1, 3.0)
    tmp_dim1_a_mul_2 = torch.mul(tensor_a_tmp_dim1_2, 5.0)
    concat_tmp_a_1 = torch.concat([tmp_dim1_a_mul_1, tmp_dim1_a_mul_2], dim=0)

    input_b_view_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])
    input_tensor_b_1 = input_b_view_2d[:input_b_view_2d.shape[0] // 2, : input_b_view_2d.shape[-1] // 2]
    input_tensor_b_2 = input_b_view_2d[:input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] // 2 : input_b_view_2d.shape[-1]]
    input_tensor_b_3 = input_b_view_2d[input_b_view_2d.shape[0] // 2 : input_b_view_2d.shape[0], : input_b_view_2d.shape[-1] // 2]
    input_tensor_b_4 = input_b_view_2d[input_b_view_2d.shape[0] // 2 : input_b_view_2d.shape[0], input_b_view_2d.shape[-1] // 2 : input_b_view_2d.shape[-1]]

    tensor_b_tmp_dim1_1 = torch.concat([input_tensor_b_1, input_tensor_b_2], dim=-1)
    tensor_b_tmp_dim1_2 = torch.concat([input_tensor_b_3, input_tensor_b_4], dim=-1)
    tmp_dim1_b_mul_1 = torch.mul(tensor_b_tmp_dim1_1, 3.0)
    tmp_dim1_b_mul_2 = torch.mul(tensor_b_tmp_dim1_2, 5.0)
    concat_tmp_b_1 = torch.concat([tmp_dim1_b_mul_1, tmp_dim1_b_mul_2], dim=0)

    concat_final = torch.concat([concat_tmp_a_1, concat_tmp_b_1], dim=-1)
    output_tensor = concat_final.reshape(output_tensor.shape)
    return output_tensor


def special_view_opcomposite_torch2(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: CompositeParams):
    tile_b = params.tile_b
    tile_s = params.tile_s
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b - 1) // tile_b
    s_loop = (seq_len + tile_s - 1) // tile_s

    for b_idx in range(b_loop):
        for s_idx in range(s_loop):
            input_a_view = input_tensor_a[b_idx * tile_b:(b_idx + 1) * tile_b, s_idx * tile_s:(s_idx + 1) * tile_s, :]
            input_a_view_2d = input_a_view.reshape(-1, input_a_view.shape[-1])
            input_tensor_a_1 = input_a_view_2d[:input_a_view_2d.shape[0] // 2, :input_a_view_2d.shape[-1] // 2]
            input_tensor_a_2 = input_a_view_2d[:input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] // 2:input_a_view_2d.shape[-1]]
            input_tensor_a_3 = input_a_view_2d[input_a_view_2d.shape[0] // 2:input_a_view_2d.shape[0], :input_a_view_2d.shape[-1] // 2]
            input_tensor_a_4 = input_a_view_2d[input_a_view_2d.shape[0] // 2:input_a_view_2d.shape[0], input_a_view_2d.shape[-1] // 2:input_a_view_2d.shape[-1]]

            tensor_a_tmp_dim1_1 = torch.concat([input_tensor_a_1, input_tensor_a_2], dim=-1)
            tensor_a_tmp_dim1_2 = torch.concat([input_tensor_a_3, input_tensor_a_4], dim=-1)
            tmp_dim1_a_mul_1 = torch.mul(tensor_a_tmp_dim1_1, 3.0)
            tmp_dim1_a_mul_2 = torch.mul(tensor_a_tmp_dim1_2, 5.0)
            concat_tmp_a_1 = torch.concat([tmp_dim1_a_mul_1, tmp_dim1_a_mul_2], dim=0)

            input_b_view = input_tensor_b[b_idx * tile_b:(b_idx + 1) * tile_b, s_idx * tile_s:(s_idx + 1) * tile_s, :]
            input_b_view_2d = input_b_view.reshape(-1, input_b_view.shape[-1])
            input_tensor_b_1 = input_b_view_2d[:input_b_view_2d.shape[0] // 2, :input_b_view_2d.shape[-1] // 2]
            input_tensor_b_2 = input_b_view_2d[:input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] // 2:input_b_view_2d.shape[-1]]
            input_tensor_b_3 = input_b_view_2d[input_b_view_2d.shape[0] // 2:input_b_view_2d.shape[0], :input_b_view_2d.shape[-1] // 2]
            input_tensor_b_4 = input_b_view_2d[input_b_view_2d.shape[0] // 2:input_b_view_2d.shape[0], input_b_view_2d.shape[-1] // 2:input_b_view_2d.shape[-1]]

            tensor_b_tmp_dim1_1 = torch.concat([input_tensor_b_1, input_tensor_b_2], dim=-1)
            tensor_b_tmp_dim1_2 = torch.concat([input_tensor_b_3, input_tensor_b_4], dim=-1)
            tmp_dim1_b_mul_1 = torch.mul(tensor_b_tmp_dim1_1, 3.0)
            tmp_dim1_b_mul_2 = torch.mul(tensor_b_tmp_dim1_2, 5.0)
            concat_tmp_b_1 = torch.concat([tmp_dim1_b_mul_1, tmp_dim1_b_mul_2], dim=0)

            concat_final = torch.concat([concat_tmp_a_1, concat_tmp_b_1], dim=-1)
            concat_final_3d = concat_final.reshape(tile_b, tile_s, concat_final.shape[-1])
            output_tensor[b_idx * tile_b:(b_idx + 1) * tile_b, s_idx * tile_s:(s_idx + 1) * tile_s, :] = concat_final_3d

    return output_tensor


def run_analysis(
    batch_size=32,
    seq_len=32,
    hidden_size=64,
    tile_b=4,
    tile_s=4,
    seed=42,
    dtype=torch.float32,
):
    torch.manual_seed(seed)
    params = CompositeParams(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        tile_b=tile_b,
        tile_s=tile_s,
        input_datarange=["0_0.01_normal_normal", "-1_1_normal_normal"],
        dtype="float32",
        cube_tile_shapes=[[128, 128], [128, 128], [128, 128]],
        vector_tile_shapes_2d=[4, 64],
        vector_tile_shapes_3d=[4, 4, 128],
    )

    # CPU 上生成相同输入，便于复现
    tensor_a = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    tensor_b = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    tensor_c = torch.randn(hidden_size, hidden_size, dtype=dtype)
    out_shape = (batch_size, seq_len, hidden_size * 2)
    tensor_out_1 = torch.zeros(out_shape, dtype=dtype)
    tensor_out_2 = torch.zeros(out_shape, dtype=dtype)

    out1 = special_view_opcomposite_torch(
        tensor_a, tensor_b, tensor_c, tensor_out_1, params
    )
    out2 = special_view_opcomposite_torch2(
        tensor_a, tensor_b, tensor_c, tensor_out_2, params
    )

    diff = (out1 - out2).float()
    abs_diff = diff.abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    ne_mask = abs_diff > 1e-5
    num_diff_elements = ne_mask.sum().item()
    total_elements = out1.numel()

    # 按 (b, s) 统计：该位置是否有任意特征不同
    diff_per_position = ne_mask.reshape(batch_size, seq_len, -1).any(dim=-1)
    num_diff_positions = diff_per_position.sum().item()

    print("=" * 60)
    print("special_view_opcomposite_torch vs special_view_opcomposite_torch2 差异分析")
    print("=" * 60)
    print(f"shape: batch={batch_size}, seq={seq_len}, hidden={hidden_size}, tile_b={tile_b}, tile_s={tile_s}")
    print(f"output shape: {tuple(out1.shape)}")
    print()
    print("差异统计:")
    print(f"  最大绝对差: {max_abs_diff}")
    print(f"  平均绝对差: {mean_abs_diff}")
    print(f"  不同元素数: {num_diff_elements} / {total_elements}")
    print(f"  不同 (b,s) 位置数: {num_diff_positions} / {batch_size * seq_len}")
    print()

    # 说明划分逻辑差异
    total_rows = batch_size * seq_len
    half_global = total_rows // 2
    tile_rows = tile_b * tile_s
    half_tile = tile_rows // 2
    print("划分逻辑差异（为何 3.0/5.0 作用位置不同）:")
    print(f"  v1: 整张 (batch*seq, hidden) 按「行」一分为二：")
    print(f"       前 {half_global} 行 -> 与 3.0 相关的分支，后 {total_rows - half_global} 行 -> 与 5.0 相关的分支。")
    print(f"  v2: 每个 tile 形状 (tile_b*tile_s, hidden) = ({tile_rows}, hidden)，")
    print(f"       每个 tile 内前 {half_tile} 行 -> 3.0，后 {tile_rows - half_tile} 行 -> 5.0。")
    print()

    # 哪些 (b,s) 在 v1 里是「前半」（乘 3 的那一半）vs「后半」
    # v1: 行号 = b * seq_len + s，行号 < total_rows//2 为前半
    v1_is_first_half = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for b in range(batch_size):
        for s in range(seq_len):
            row = b * seq_len + s
            v1_is_first_half[b, s] = row < half_global

    # v2: 每个 tile 内，前 half_tile 行为前半（即 tile 内行号 0..half_tile-1）
    # tile 内行号 = (b - b_idx*tile_b) * tile_s + (s - s_idx*tile_s)
    v2_is_first_half = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for b_idx in range((batch_size + tile_b - 1) // tile_b):
        for s_idx in range((seq_len + tile_s - 1) // tile_s):
            for bi in range(tile_b):
                for si in range(tile_s):
                    b = b_idx * tile_b + bi
                    s = s_idx * tile_s + si
                    if b >= batch_size or s >= seq_len:
                        continue
                    tile_row = bi * tile_s + si
                    v2_is_first_half[b, s] = tile_row < half_tile

    disagree_which_half = (v1_is_first_half != v2_is_first_half)
    num_disagree = disagree_which_half.sum().item()
    print("「前半行 / 后半行」归属不一致的 (b,s) 数量（这些位置会乘的系数在 v1/v2 中不同）:")
    print(f"  {num_disagree} / {batch_size * seq_len}")
    if num_disagree > 0 and num_disagree <= 50:
        bs_disagree = torch.nonzero(disagree_which_half, as_tuple=False)
        print("  示例 (b, s):")
        for i in range(min(10, len(bs_disagree))):
            b, s = bs_disagree[i, 0].item(), bs_disagree[i, 1].item()
            v1_h = "前" if v1_is_first_half[b, s].item() else "后"
            v2_h = "前" if v2_is_first_half[b, s].item() else "后"
            print(f"    ({b}, {s}): v1={v1_h}半, v2={v2_h}半")
    print()

    # 抽样打印几个位置的数值
    print("抽样对比（前几个有差异的 (b,s) 位置，取该位置前 4 个特征）:")
    count = 0
    for b in range(batch_size):
        for s in range(seq_len):
            if not diff_per_position[b, s].item():
                continue
            count += 1
            if count > 5:
                break
            print(f"  (b={b}, s={s}): out1[:4]={out1[b, s, :4].tolist()}")
            print(f"             out2[:4]={out2[b, s, :4].tolist()}")
        if count > 5:
            break
    print("=" * 60)
    return out1, out2, diff_per_position, params


if __name__ == "__main__":
    run_analysis()
