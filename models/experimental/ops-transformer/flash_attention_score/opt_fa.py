from typing import Optional
import argparse
import torch
import pypto
import time
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'deepseek_v32_exp'))
from utils.compare import compare

S1_TILE = 64
# [optimizations] 4. KV block expansion + increased vector tiling
S2_TILE = 64

def attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

def attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    run_mode: str = "npu",
):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError("run_mode must be 'npu' or 'sim'")

    D_TILE = head_dim
    CUBE_TILE = 128

    # tiling size 参考 glm_attention
    c1_tile = [[S1_TILE, S1_TILE], [CUBE_TILE, CUBE_TILE], [2*CUBE_TILE, 2*CUBE_TILE]]
    v1_tile = [32, S2_TILE]
    c2_tile = [[S1_TILE, S1_TILE], [CUBE_TILE, CUBE_TILE], [CUBE_TILE, CUBE_TILE]]
    v2_tile = [32, CUBE_TILE]

    @pypto.frontend.jit(
        runtime_options={
            "run_mode": mode,
            "stitch_function_max_num": 129,
            "stitch_cfgcache_size": 3000000,
        },
        debug_options={"runtime_debug_mode": 1},
        pass_options={
            "cube_l1_reuse_setting": {-1: 4},
            "cube_nbuffer_setting": {1: 2},
            "vec_nbuffer_setting": {-1: 4}
        },
    )
    def attention_kernel(
        q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
        k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
        v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
        out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    ):
        scale = 1.0 / math.sqrt(head_dim)

        pypto.experimental.set_operation_options(combine_axis=True)

        num_q_blocks = seq_len // S1_TILE
        num_k_blocks = seq_len // S2_TILE
        kb_unroll = [1,2,4,8]

        # [optimizations] 1. tensor flattening (4D -> 2D layout)
        # 1. qkv在for外面reshape成二维
        q_2d = pypto.reshape(q, (batch_size * num_heads * seq_len, head_dim), inplace=True)
        k_2d = pypto.reshape(k, (batch_size * num_heads * seq_len, head_dim), inplace=True)
        v_2d = pypto.reshape(v, (batch_size * num_heads * seq_len, head_dim), inplace=True)

        for b in pypto.loop(batch_size, name="LOOP_b", idx_name="b"):
            for h in pypto.loop(num_heads, name="LOOP_h", idx_name="h"):
                for qb in pypto.loop(num_q_blocks, name="LOOP_qb", idx_name="qb"):
                    # 2. mi li oi在for b for h for qb的下面
                    oi = pypto.tensor([S1_TILE, head_dim], pypto.DT_FP32, "oi")
                    mi = pypto.tensor([S1_TILE, 1], pypto.DT_FP32, "mi")
                    li = pypto.tensor([S1_TILE, 1], pypto.DT_FP32, "li")

                    q_row_ofs = (b * num_heads + h) * seq_len + qb * S1_TILE
                    actual_q_tile = (seq_len - qb * S1_TILE).min(S1_TILE)

                    for kb in pypto.loop(
                        num_k_blocks,
                        name="LOOP_kb",
                        idx_name="kb",
                        unroll_list=kb_unroll,
                    ):
                        kv_row_ofs = (b * num_heads + h) * seq_len + kb * S2_TILE
                        actual_kv_tile = (seq_len - kb * S2_TILE).min(S2_TILE)
                        pypto.set_semantic_label("LOOP")

                        # 1. 循环内用二维view和matmul
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        # [optimizations] 2. inner-loop Q placement
                        qi = pypto.view(q_2d, [S1_TILE, head_dim], [q_row_ofs, 0])
                        ki = pypto.view(k_2d, [S2_TILE, head_dim], [kv_row_ofs, 0])

                        # c1: QK matmul
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        scores = pypto.matmul(qi, ki, pypto.DT_FP32, a_trans=False, b_trans=True)
                        scores = pypto.view(scores, [S1_TILE, S2_TILE], [0, 0]) # should change for valid_shape

                        # 3. softmax处tilesize尾轴等于S2_TILE，不切尾轴
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(kb):
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(scores, scale)
                            m_blk = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            tsub = pypto.sub(sij_scale, m_blk)
                            p_fp32 = pypto.exp(tsub)
                            p_bf16 = pypto.cast(p_fp32, pypto.DT_BF16)
                            li[:] = pypto.sum(p_fp32, dim=-1, keepdim=True)
                            mi[:] = m_blk
                            pypto.set_pass_options(sg_set_scope=-1)

                            # c2: PV matmul
                            vi = pypto.view(v_2d, [S2_TILE, head_dim], [kv_row_ofs, 0])
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(p_bf16, vi, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi[:] = oi_tmp
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(scores, scale)
                            m_blk = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            tsub = pypto.sub(sij_scale, m_blk)
                            p_fp32 = pypto.exp(tsub)
                            p_bf16 = pypto.cast(p_fp32, pypto.DT_BF16)
                            sum_local = pypto.sum(p_fp32, dim=-1, keepdim=True)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_pass_options(sg_set_scope=2)
                            m_new = pypto.maximum(mi, m_blk)
                            tsub2 = pypto.sub(mi, m_new)
                            tsub3 = pypto.sub(m_blk, m_new)
                            mi[:] = m_new
                            alpha = pypto.exp(tsub2)
                            beta = pypto.exp(tsub3)
                            li[:] = li * alpha + sum_local * beta
                            pypto.set_pass_options(sg_set_scope=-1)

                            # c2: PV matmul
                            vi = pypto.view(v_2d, [S2_TILE, head_dim], [kv_row_ofs, 0])
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(p_bf16, vi, pypto.DT_FP32)

                            # v2: update oi
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            pypto.set_pass_options(sg_set_scope=2)
                            oi[:] = oi * alpha + oi_tmp * beta
                            pypto.set_pass_options(sg_set_scope=-1)

                        # [optimizations] 3. in-loop output finalization
                        if pypto.is_loop_end(kb):
                            # assemble结果到out
                            pypto.set_pass_options(sg_set_scope=2)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_final = pypto.div(oi, li)
                            pypto.set_vec_tile_shapes(1, 1, v2_tile[0], v2_tile[1])
                            oi_final_4d = pypto.cast(
                                pypto.reshape(oi_final, [1, 1, S1_TILE, head_dim]),
                                pypto.DT_FP32)
                            pypto.assemble(oi_final_4d, [b, h, qb * S1_TILE, 0], out)
                            pypto.set_pass_options(sg_set_scope=-1)

    return attention_kernel

def test_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device_id: int = 5,
    run_mode: str = "npu",
    dynamic: bool = False,
) -> None:
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    import torch_npu

    if run_mode == "npu":
        torch.npu.set_device(device_id)

    q_torch = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k_torch = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_torch = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    attn = attention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        run_mode=run_mode,
    )

    out = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
        device=device,
    )

    print(f"Running shape: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    attn(q_torch, k_torch, v_torch, out)

    scale = 1.0 / (head_dim ** 0.5)
    golden = attention_golden(q_torch.float(), k_torch.float(), v_torch.float(), scale)

    if run_mode == "npu":
        max_diff = (out - golden).abs().max().item()
        compare(out, golden, "atten_out", atol=0.001, rtol=0.005, max_error_count=10)
        print(f"Batch={batch_size}, SeqQ={seq_len}, SeqKV={seq_len}, Max diff: {max_diff:.10f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", type=str, default="npu", choices=["npu", "sim"])
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_attention(
        batch_size=args.batch,
        num_heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device_id=args.device_id,
        run_mode=args.run_mode,
    )


