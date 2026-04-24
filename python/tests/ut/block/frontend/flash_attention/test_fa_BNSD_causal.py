"""FlashAttention kernel with Causal Mask - BN axis version.

Causal (lower triangular) attention: each position can only attend to itself and previous positions.

Features:
  1. Causal mask: Q[i] can only attend to K/V[0:i+1]
  2. Optimized KV loop: only compute necessary KV tiles (j <= qi)
  3. External mask: mask tensor is passed from outside the kernel
  4. Mask format: [Sq, Skv] UINT8 tensor (0=mask out, 1=attend)
     - Cast to FP16, then converted to bit-packed predicate using cmps (TCMPS instruction)
     - cmps with NE mode: mask != 0 -> bit=1 (attend)

Tensor shapes:
  - Q/K/V/O: [B, N, Sq, D], qk_buf: [B, N, Sq, Skv], p_buf: [B, N, Sq, Skv]
  - attn_mask: [Sq, Skv] UINT8 - attention mask (0=mask out, 1=attend)

Mask processing flow (following AscendC pattern):
  1. Load UINT8 mask (each element is 0 or 1)
  2. Cast UINT8 -> FP16
  3. cmps(mask_bit, mask_fp16, 0, NE) -> bit-packed mask
  4. sel(dst, mask_bit, src0, src1, tmp)

Usage:
    python3 tests/ut/frontend/flash_attention/test_fa_BNSD_causal.py
"""

import math
import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm

TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

Q_F16   = TS * TD * 2
KT_F16  = TD * TKV * 2
QK_F16  = TS * TKV * 2
V_F16   = TKV * TD * 2
P_F16   = TS * TKV * 2
QK_F32  = TS * TKV * 4
PV_F32  = TS * TD * 4

MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16  * 2
MA3_PONG = MA3 + V_F16

LA0 = 0
LA1 = Q_F16

RA0 = 0
RA1 = KT_F16

CA0 = 0
CA1 = QK_F32

VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB1_KV = TS_HALF * TKV * 1  # Bit-packed mask: same shape as data tile, INT8
VB4    = TS_HALF * TD * 4
VB2    = TS_HALF * TD * 2
VB_RED = TS_HALF * 1 * 4

VA0 = 0
VA1 = VA0 + VB4_KV
VA2 = VA1 + VB4_KV
VA3 = VA2 + VB2_KV
VA4 = VA3 + VB_RED
VA5 = VA4 + VB_RED
VA6 = VA5 + VB_RED
VA7 = VA6 + VB_RED  # FIX: was VA6 + VB4, causing 32KB gap and overlaps
VA8 = VA7 + VB4
VA9 = VA8 + VB4
VA10 = VA9 + VB2
VA11 = VA10 + VB1_KV  # New: separate address for p_f16

QK_READY = 0
P_READY = 1
PV_READY = 2

B = pl.DynVar('B')
N = pl.DynVar('N')
Sq2 = pl.DynVar('Sq')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')
NumRanges = pl.DynVar('NumRanges')

NEG_INF = -1e9


@fe.kernel
def fa_causal_kernel_bn(
    q: pl.Tensor[[B, N, Sq2, D2], pl.FP16],
    k: pl.Tensor[[B, N, Skv2, D2], pl.FP16],
    v: pl.Tensor[[B, N, Skv2, D2], pl.FP16],
    o: pl.Tensor[[B, N, Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[B, N, Sq2, Skv2], pl.FP32],
    p_buf: pl.Tensor[[B, N, Sq2, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[48 * TS, D2], pl.FP32],
    work_ranges: pl.Tensor[[NumRanges, 2], pl.INT32],
    attn_mask: pl.Tensor[[Sq2, Skv2], pl.UINT8],  # Causal mask in UINT8 format (0 or 1)
) -> pl.Tensor[[B, N, Sq2, D2], pl.FP16]:
    with pl.section_cube():
        sq_dim = Sq2
        skv_dim = Skv2
        sq_tiles = (sq_dim + (TS - 1)) // TS
        skv_tiles = (skv_dim + (TKV - 1)) // TKV
        num_cores = pl.block.index_cast(pl.block.get_block_num())
        core_id = pl.block.index_cast(pl.block.get_block_idx())
        n_dim = N

        q_mat_type = plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        q_mat_0 = plm.make_tile(q_mat_type, addr=MA0, size=Q_F16)
        q_mat_1 = plm.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)
        q_mat_buf = (q_mat_0, q_mat_1)

        k_mat_type = plm.TileType(shape=[TD, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
        k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
        k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)
        k_mat_buf = (k_mat_0, k_mat_1)

        p_mat_0 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2, size=P_F16)
        p_mat_1 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2_PONG, size=P_F16)
        p_mat_buf = (p_mat_0, p_mat_1)

        v_mat_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3, size=V_F16)
        v_mat_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3_PONG, size=V_F16)
        v_mat_buf = (v_mat_0, v_mat_1)

        left_0 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA0, size=Q_F16)
        left_1 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA1, size=Q_F16)
        left_buf = (left_0, left_1)

        right_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=KT_F16)
        right_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=KT_F16)
        right_buf = (right_0, right_1)

        acc_0 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_F32)
        acc_1 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_F32)
        acc_buf = (acc_0, acc_1)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)

        q_count = 0
        left_index = 0
        right_index = 0
        work_start = pl.block.index_cast(pl.read(work_ranges, [core_id, 0]))
        work_end = pl.block.index_cast(pl.read(work_ranges, [core_id, 1]))
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
                sq_off = qi * TS
                q_mat_idx = q_count % 2
                plm.load_tile(q_mat_buf[q_mat_idx], q, [b_idx, n_idx, qi, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

                causal_kv_tiles = pl.min(qi + 1, skv_tiles)
                for j in pl.range(0, causal_kv_tiles):
                    buf_idx = (q_count * skv_tiles + j) % 2
                    skv_off = j * TKV

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    plm.move(left_buf[left_index], q_mat_buf[q_mat_idx])
                    if j == causal_kv_tiles - 1:
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)

                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    plm.load_tile(k_mat_buf[buf_idx], k, [b_idx, n_idx, j, 0], layout="dn")
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    plm.move(right_buf[right_index], k_mat_buf[buf_idx])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)

                    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
                    plm.matmul(acc_buf[buf_idx], left_buf[left_index], right_buf[right_index])
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    right_index = 1 - right_index
                    left_index = 1 - left_index
                    plm.store_tile(qk_buf, acc_buf[buf_idx], [b_idx, n_idx, qi, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)

                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=P_READY)

                    buf_idx_pv = (q_count * skv_tiles + j) % 2
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
                    plm.load_tile(p_mat_buf[buf_idx], p_buf, [b_idx, n_idx, qi, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
                    plm.load_tile(v_mat_buf[buf_idx], v, [b_idx, n_idx, j, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    plm.move(left_buf[left_index], p_mat_buf[buf_idx])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    plm.move(right_buf[right_index], v_mat_buf[buf_idx])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)

                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
                    plm.matmul(acc_buf[buf_idx_pv], left_buf[left_index], right_buf[right_index])
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    right_index = 1 - right_index
                    left_index = 1 - left_index
                    plm.store_tile(pv_buf, acc_buf[buf_idx_pv], [core_id * 2 + q_mat_idx, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)
                q_count = q_count + 1
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)

    with pl.section_vector():
        sq_dim = Sq2
        skv_dim = Skv2
        sq_tiles = (sq_dim + (TS - 1)) // TS
        skv_tiles = (skv_dim + (TKV - 1)) // TKV
        num_cores = pl.block.index_cast(pl.block.get_block_num())
        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * TS_HALF

        qk_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_KV)
        tmp_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)

        # UINT8 mask tile loaded from GM
        mask_u8 = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB1_KV)
        # FP16 mask tile (input to cmps)
        mask_fp16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)
        # Bit-packed mask tile (output from cmps, input to sel)
        mask_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec), addr=VA10, size=VB1_KV)

        p_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB2_KV)

        reduce_dst = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA3, size=VB_RED)
        global_max = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA4, size=VB_RED)
        global_sum = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA5, size=VB_RED)
        exp_corr = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA6, size=VB_RED)

        reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)
        global_max_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA4, size=VB_RED)
        global_sum_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA5, size=VB_RED)
        exp_corr_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA6, size=VB_RED)

        running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
        pv_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
        o_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)

        q_count = 0
        n_dim = N
        work_start = pl.block.index_cast(pl.read(work_ranges, [core_id, 0]))
        work_end = pl.block.index_cast(pl.read(work_ranges, [core_id, 1]))
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                sq_off = qi * TS
                q_mat_idx = q_count % 2
                causal_kv_tiles = pl.min(qi + 1, skv_tiles)

                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=QK_READY)
                plm.load_tile(qk_vec, qk_buf, [b_idx, n_idx, qi * 2 + sub_id, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                # Only the diagonal KV tile needs causal masking.
                # For qi > 0, j = 0 is fully valid and should bypass masking.
                if qi == 0:
                    plm.load_tile(mask_u8, attn_mask, [qi * 2 + sub_id, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    plm.cast(mask_fp16, mask_u8, target_type=pl.FP16, mode="round")
                    pl.system.bar_v()

                    plm.cmps(mask_vec, mask_fp16, 0.0, cmp_type=1)
                    pl.system.bar_v()

                    # Before running_o is loaded, we can safely reuse it as the NEG_INF tile.
                    plm.expands(running_o, NEG_INF)
                    pl.system.bar_v()
                    plm.sel(out=pv_vec, mask=mask_vec, lhs=qk_vec, rhs=running_o, tmp=tmp_vec)
                    pl.system.bar_v()

                if qi == 0:
                    plm.row_max(reduce_dst, pv_vec, tmp_vec)
                else:
                    plm.row_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()

                if qi == 0:
                    plm.row_expand_sub(tmp_vec, pv_vec, reduce_dst)
                else:
                    plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
                pl.system.bar_v()
                plm.muls(global_max, reduce_dst, 1.0)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                pl.system.bar_v()
                plm.exp(qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.row_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()

                plm.muls(global_sum, reduce_dst, 1.0)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store_tile(p_buf, p_f16, [b_idx, n_idx, qi * 2 + sub_id, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=PV_READY)
                plm.load_tile(running_o, pv_buf, [core_id * 4 + q_mat_idx * 2 + sub_id, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                for j in pl.range(1, causal_kv_tiles):
                    skv_off = j * TKV
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=QK_READY)
                    plm.load_tile(qk_vec, qk_buf, [b_idx, n_idx, qi * 2 + sub_id, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                    if j == qi:
                        plm.load_tile(mask_u8, attn_mask, [qi * 2 + sub_id, j])
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        plm.cast(mask_fp16, mask_u8, target_type=pl.FP16, mode="round")
                        pl.system.bar_v()

                        plm.cmps(mask_vec, mask_fp16, 0.0, cmp_type=1)
                        pl.system.bar_v()

                        plm.expands(pv_vec, NEG_INF)
                        pl.system.bar_v()
                        plm.sel(out=qk_vec, mask=mask_vec, lhs=qk_vec, rhs=pv_vec, tmp=tmp_vec)
                        pl.system.bar_v()

                    plm.row_max(reduce_dst, qk_vec, tmp_vec)
                    pl.system.bar_v()
                    plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
                    pl.system.bar_v()
                    plm.sub(exp_corr_rm, global_max_rm, reduce_dst_rm)
                    pl.system.bar_v()
                    plm.muls(global_max_rm, reduce_dst_rm, 1.0)
                    pl.system.bar_v()
                    plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
                    plm.muls(exp_corr_rm, exp_corr_rm, SCALE)
                    plm.muls(tmp_vec, tmp_vec, SCALE)
                    plm.exp(exp_corr_rm, exp_corr_rm)
                    plm.exp(qk_vec, tmp_vec)
                    plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                    pl.system.bar_v()

                    plm.mul(global_sum_rm, global_sum_rm, exp_corr_rm)
                    plm.row_sum(reduce_dst, qk_vec, tmp_vec)
                    pl.system.bar_v()

                    plm.add(global_sum_rm, global_sum_rm, reduce_dst_rm)

                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                    plm.store_tile(p_buf, p_f16, [b_idx, n_idx, qi * 2 + sub_id, j])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=PV_READY)
                    plm.load_tile(pv_vec, pv_buf, [core_id * 4 + q_mat_idx * 2 + sub_id, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    plm.row_expand_mul(running_o, running_o, exp_corr)
                    plm.add(running_o, running_o, pv_vec)
                q_count = q_count + 1

                plm.row_expand_div(running_o, running_o, global_sum)
                plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store_tile(o, o_f16, [b_idx, n_idx, qi * 2 + sub_id, 0])
    return o


def flash_attention_causal_ref_bn(q, k, v, d, debug=False):
    scale_val = 1.0 / math.sqrt(d)
    b, n, sq, d_ = q.shape
    _, _, skv, _ = k.shape
    o_ref = torch.zeros_like(q)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, ni].float(), k[bi, ni].float().T) * scale_val
            causal_mask = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=q.device), diagonal=1)
            qk = qk.masked_fill(causal_mask, float('-inf'))

            if debug:
                print(f"\n=== Golden Debug for batch={bi}, head={ni} ===")
                print(f"QK (first 8x8): \n{qk[:8, :8]}")
                print(f"QK min={qk.min().item():.4f}, max={qk.max().item():.4f}")

            # Compute softmax step by step for debugging
            qk_max = qk.max(dim=-1, keepdim=True)[0]  # row max
            if debug:
                print(f"QK row_max (first 8): {qk_max[:8, 0]}")

            qk_exp = torch.exp(qk - qk_max)  # exp(qk - max)
            if debug:
                print(f"exp(QK - max) (first 8x8): \n{qk_exp[:8, :8]}")

            qk_sum = qk_exp.sum(dim=-1, keepdim=True)  # row sum
            if debug:
                print(f"row_sum (first 8): {qk_sum[:8, 0]}")

            attn = qk_exp / qk_sum  # softmax
            if debug:
                print(f"Attention (first 8x8): \n{attn[:8, :8]}")
                print(f"Attention min={attn.min().item():.6f}, max={attn.max().item():.6f}, sum={attn.sum(dim=-1)[:8]}")

            o_ref[bi, ni] = torch.matmul(attn, v[bi, ni].float()).half()
            if debug:
                print(f"Output (first 8 dims): {o_ref[bi, ni, 0, :8]}")
    return o_ref


def _window_len(dim, max_len=8):
    return min(dim, max_len)


def _safe_row_indices(num_rows):
    if num_rows <= 0:
        return []
    candidates = [0, 1, 64, 65, 127, 128, num_rows - 1]
    result = []
    for idx in candidates:
        if 0 <= idx < num_rows and idx not in result:
            result.append(idx)
    return result


def _print_mask_samples(attn_mask):
    sq, skv = attn_mask.shape
    col_len = _window_len(skv)
    print(f"\n=== attn_mask verification ({attn_mask.dtype}) ===")
    for row_idx in _safe_row_indices(sq):
        print(f"Row {row_idx} (first {col_len}): {attn_mask[row_idx, :col_len].cpu().numpy()}")


def test_fa_causal_bn():
    compiled = fe.compile(fa_causal_kernel_bn, arch="a3")
    print("compiled:", compiled.lib_path)
    device = "npu:3"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    debug_max_tokens = 256 * 256
    golden_max_tokens = 512 * 512

    for b, n, sq, skv, d, num_cores in [
        (1, 1, 128, 128, TD, 1),
        (2, 4, 256, 256, TD, 4),
        (1, 3, 512, 512, TD, 24),
        (4, 2, 256, 256, TD, 16),
        (1, 1, 1024, 1024, TD, 24),
        (2, 3, 8192, 8192, TD, 24),
    ]:
        print(f"\nFA-Causal-BN (b={b}, n={n}, sq={sq}, skv={skv}, d={d}) cores={num_cores}")
        q = torch.rand((b, n, sq, d), device=device, dtype=torch.float16)
        k = torch.rand((b, n, skv, d), device=device, dtype=torch.float16)
        v = torch.rand((b, n, skv, d), device=device, dtype=torch.float16)
        o = torch.zeros((b, n, sq, d), device=device, dtype=torch.float16)
        qk_buf = torch.zeros((b, n, sq, skv), device=device, dtype=torch.float32)
        p_buf = torch.zeros((b, n, sq, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((48 * TS, d), device=device, dtype=torch.float32)

        # Vectorized lower-triangular causal mask in UINT8 format (0 or 1).
        attn_mask = torch.tril(torch.ones((sq, skv), device=device, dtype=torch.uint8))
        if sq * skv <= debug_max_tokens:
            _print_mask_samples(attn_mask)

        total_work = b * n
        work_ranges = torch.zeros((num_cores, 2), device=device, dtype=torch.int32)

        work_per_core = (total_work + num_cores - 1) // num_cores
        for core in range(num_cores):
            work_start = core * work_per_core
            work_end = min((core + 1) * work_per_core, total_work)
            work_ranges[core, 0] = work_start
            work_ranges[core, 1] = work_end

        actual_num_cores = min(num_cores, total_work)
        fe.launch(None, actual_num_cores, compiled, q, k, v, o, qk_buf, p_buf, pv_buf, work_ranges, attn_mask)
        torch.npu.synchronize()

        print(f"  Q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
        print(f"  K stats: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
        print(f"  V stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        print(f"  QK_buf stats: min={qk_buf.min().item():.4f}, max={qk_buf.max().item():.4f}, mean={qk_buf.mean().item():.4f}")
        print(f"  P_buf stats: min={p_buf.min().item():.4f}, max={p_buf.max().item():.4f}, mean={p_buf.mean().item():.4f}")
        print(f"  O stats: min={o.min().item():.4f}, max={o.max().item():.4f}, mean={o.mean().item():.4f}")

        if torch.isnan(qk_buf).any():
            print(f"  WARNING: QK_buf contains NaN!")
        if torch.isinf(qk_buf).any():
            print(f"  WARNING: QK_buf contains Inf!")
        if torch.isnan(p_buf).any():
            print(f"  WARNING: P_buf contains NaN!")
        if torch.isinf(p_buf).any():
            print(f"  WARNING: P_buf contains Inf!")
        if torch.isnan(o).any():
            print(f"  WARNING: O contains NaN!")
        if torch.isinf(o).any():
            print(f"  WARNING: O contains Inf!")

        run_golden = (sq * skv <= golden_max_tokens)
        run_debug = (sq * skv <= debug_max_tokens and b == 1 and n == 1)
        if not run_golden:
            print(f"  Skip golden compare for large case sq={sq}, skv={skv}")
            continue

        o_ref = flash_attention_causal_ref_bn(q, k, v, d, debug=run_debug)

        if run_debug:
            print(f"\n{'='*60}")
            print(f"Step-by-step Comparison: NPU vs Golden")
            print(f"{'='*60}")

            row_len = _window_len(sq)
            col_len = _window_len(skv)
            out_row_len = _window_len(sq, 2)
            out_col_len = _window_len(d)

            q_g = q[0, 0].float()
            k_g = k[0, 0].float()
            v_g = v[0, 0].float()
            scale_g = 1.0 / math.sqrt(d)

            qk_raw_g = q_g @ k_g.T
            mask_g = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=device), diagonal=1)

            print(f"\n--- Step 1: QK_raw = Q @ K^T (no scale, no mask) ---")
            print(f"[NPU] QK_buf[0,0,:{row_len},:{col_len}]:\n{qk_buf[0, 0, :row_len, :col_len]}")
            print(f"[Golden] QK_raw[:{row_len},:{col_len}]:\n{qk_raw_g[:row_len, :col_len]}")
            qk_diff = (qk_buf[0, 0, :row_len, :col_len] - qk_raw_g[:row_len, :col_len]).abs().max().item()
            print(f"[Diff] max|QK_raw diff| = {qk_diff:.6f}")
            if qk_diff < 0.1:
                print("✓ QK_raw matches!")

            print(f"\n--- Step 2: Apply causal mask ---")
            qk_masked_g = qk_raw_g.masked_fill(mask_g, float('-inf'))
            print(f"[NPU] dump_tile(qk_vec) - check console output above")
            print(f"[Golden] QK_masked[:{row_len},:{col_len}]:\n{qk_masked_g[:row_len, :col_len]}")
            print(f"Compare: NPU dump should match Golden QK_masked (upper triangle = -inf)")

            print(f"\n--- Step 3: Apply scale ---")
            qk_scaled_g = qk_masked_g * scale_g
            print(f"[Golden] QK_scaled[:{row_len},:{col_len}]:\n{qk_scaled_g[:row_len, :col_len]}")
            print(f"Note: NPU applies scale in vector section")

            print(f"\n--- Step 4: Softmax ---")
            qk_max_g = qk_scaled_g.max(dim=-1, keepdim=True)[0]
            qk_exp_g = (qk_scaled_g - qk_max_g).exp()
            qk_sum_g = qk_exp_g.sum(dim=-1, keepdim=True)
            attn_g = qk_exp_g / qk_sum_g

            print(f"[Golden] row_max[:{row_len}]: {qk_max_g[:row_len, 0]}")
            print(f"[Golden] row_sum[:{row_len}]: {qk_sum_g[:row_len, 0]}")
            print(f"[Golden] Attention[:{row_len},:{col_len}]:\n{attn_g[:row_len, :col_len]}")

            print(f"\n[NPU] P_buf[0,0,:{row_len},:{col_len}]:\n{p_buf[0, 0, :row_len, :col_len]}")
            p_diff = (p_buf[0, 0, :row_len, :col_len].float() - attn_g[:row_len, :col_len]).abs().max().item()
            print(f"[Diff] max|P diff| = {p_diff:.6f}")
            if p_diff < 0.01:
                print("✓ P matches!")
            else:
                print("✗ P mismatch! Check dump_tile output above for details")

            print(f"\n--- Step 3: O = P @ V ---")
            print(f"[NPU] O[0,0,:{out_row_len},:{out_col_len}]:\n{o[0, 0, :out_row_len, :out_col_len]}")
            print(f"[Golden] O_ref[0,0,:{out_row_len},:{out_col_len}]:\n{o_ref[0, 0, :out_row_len, :out_col_len]}")
            print(f"[NPU] O stats: min={o.min().item():.4f}, max={o.max().item():.4f}, mean={o.mean().item():.4f}")
            print(f"[Golden] O_ref stats: min={o_ref.min().item():.4f}, max={o_ref.max().item():.4f}, mean={o_ref.mean().item():.4f}")

        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.6f}")

        for ni in range(n):
            diff_head = (o[:, ni, :, :] - o_ref[:, ni, :, :]).abs().max().item()
            print(f"  Head {ni} max|diff|={diff_head:.4f}")

        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        print("  PASS")


if __name__ == "__main__":
    print("FA Causal Attention - DN layout for K, multi-core + Q tiling + double buffer + BN axis")
    print("=" * 80)
    test_fa_causal_bn()
    print("\nAll FlashAttention Causal tests passed!")
