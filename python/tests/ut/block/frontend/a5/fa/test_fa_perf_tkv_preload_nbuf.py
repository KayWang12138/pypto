"""FlashAttention performance kernel using NBuffer + auto_mutex.

Refactored from test_fa_perf_tkv_preload.py to use:
  - NBuffer with current() auto-rotate cursor (no manual buf_idx)
  - auto_mutex=True for automatic Mutex synchronization
  - Cross-core event_id synchronization preserved (QK_READY, P_READY, PV_READY)

Features:
  1. Multi-core: each Cube core processes multiple Q tiles via strided loop
  2. Double buffer: NBuffer with current() auto-rotate for Q/K/V/P/L0A/L0B/L0C
  3. FIFO cross-core GM buffers with configurable depth
  4. Cross-core event ID ping/pong
  5. QK pre-compute: Cube runs QK_PRELOAD tiles ahead
  6. Vector: FIFO exp_corr, double-buffered global_max/global_sum

Usage:
    python3 python/tests/ut/block/frontend/a5/test_fa_perf_tkv_preload_nbuf.py
"""

import math
import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 1
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128; TKV = 128; TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

# Buffer sizes (bytes)
Q_F16 = TS * TD * 2; KT_F16 = TD * TKV * 2; V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2; QK_HALF_F32 = TS * TKV * 4; PV_HALF_F32 = TS * TD * 4

# VEC buffer sizes
VB4_KV = TS_HALF * TKV * 4; VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4; VB2 = TS_HALF * TD * 2
VB6 = (TS_HALF + 1) * TD * 2
VB_RED = TS_HALF * 1 * 4

# ================================================================
#  Buffer addresses
# ================================================================
# MAT (512KB) - L1 buffers
MA0_Q = 0
MA1_K = Q_F16 * 2
MA2_P = MA1_K + KT_F16 * 2
MA3_V = MA2_P + P_F16 * 2

# L0A/L0B/L0C addresses
LA0 = 0; LA1 = P_F16
RA0 = 0; RA1 = KT_F16
CA0 = 0; CA1 = QK_HALF_F32

# VEC (192KB) addresses
VA0 = 0
VA1 = VA0 + VB4_KV * 2
VA2 = VA1 + VB4_KV
VA3 = VA2 + VB2_KV
VA_GMAX0 = VA3 + VB_RED
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GSUM0 = VA_GMAX1 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_EXP_BASE = VA_GSUM1 + VB_RED
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * VB_RED
VA7 = VA_AFTER_EXP
VA8 = VA7 + VB4
VA9 = VA8 + VB4 * 2
VA10 = VA9 + VB2
VA11 = VA10 + VB6
assert VA11 <= 248 * 1024

# ================================================================
#  Event IDs (cross-core)
# ================================================================
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16
QK_MAX_EID = FIFO_SIZE
P_MAX_EID = 2 * FIFO_SIZE
PV_MAX_EID = 3 * FIFO_SIZE

# ================================================================
#  Mutex IDs - Cube and Vector use independent buf_id spaces
# ================================================================
# Cube-only (inside section_cube): 0-11
#   Q L1: (0, 1), K L1: (2, 3), V L1: (4, 5)
#   L0A: (6, 7), L0B: (8, 9), L0C: (10, 11)
#
# Vector-only (inside section_vector): 0-11
#   tmp_vec: 0, p_f16: 1, reduce_dst: 2
#   gmax_rm: (3, 4), gsum: (5, 6)
#   exp_corr: (7, 8), running_o: 9, o_f16: 10, tile_nz: 11
#
# Cross-core shared (outside sections): 12-17
#   P MAT: (12, 13)
#   qk_vec UB: (14, 15)
#   pv_vec UB: (16, 17)

# Cross-core shared buffer IDs
P_MUTEX_IDS = (12, 13)
QK_VEC_BUF_IDS = (14, 15)
PV_VEC_BUF_IDS = (16, 17)

PV_CORE_STRIDE = 2 * FIFO_SIZE * TS

Sq2 = pl.DynVar('Sq')
Sq_fifo = pl.DynVar('SqFifo')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')


# ================================================================
#  Kernel with NBuffer + auto_mutex
# ================================================================
@fe.kernel(auto_mutex=True)
def fa_perf_tkv_preload_nbuf_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[Sq_fifo, Skv2], pl.FP32],
    p_buf: pl.Tensor[[Sq_fifo, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, D2], pl.FP32],
) -> pl.Tensor[[Sq2, D2], pl.FP16]:

    sq_dim = Sq2
    skv_dim = Skv2
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.block.index_cast(pl.block.get_block_num())
    core_id = pl.block.index_cast(pl.block.get_block_idx())

    # ========== Cross-core shared buffers (UBNBuffer for double-buffer) ==========
    # P MAT - Vector insert, Cube PV read
    p_mat_db = pl.L1NBuffer(MA2_P, pl.FP16, shape=(TS, TKV), buf_ids=P_MUTEX_IDS)
    
    # qk_vec UB - Cube store from ACC, Vector softmax (double-buffer for FIFO)
    qk_vec_db = pl.UBNBuffer(VA0, pl.FP32, shape=(TS_HALF, TKV), buf_ids=QK_VEC_BUF_IDS)
    
    # pv_vec UB - Cube store from ACC, Vector GU (double-buffer for FIFO)
    pv_vec_db = pl.UBNBuffer(VA8, pl.FP32, shape=(TS_HALF, TD), buf_ids=PV_VEC_BUF_IDS)

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        # Cube-only buffers (independent buf_id space: 0-11)
        q_l1_db = pl.L1NBuffer(MA0_Q, pl.FP16, shape=(TS, TD), buf_ids=(0, 1))
        k_l1_db = pl.L1NBuffer(MA1_K, pl.FP16, shape=(TD, TKV), buf_ids=(2, 3), blayout=1, slayout=2)
        v_l1_db = pl.L1NBuffer(MA3_V, pl.FP16, shape=(TKV, TD), buf_ids=(4, 5))

        left_db = pl.L0ANBuffer(LA0, pl.FP16, shape=(TS, TD), buf_ids=(6, 7))
        right_db = pl.L0BNBuffer(RA0, pl.FP16, shape=(TKV, TD), buf_ids=(8, 9))
        acc_db = pl.L0CNBuffer(CA0, pl.FP32, shape=(TS, TKV), buf_ids=(10, 11))

        task_id = 0
        ctx_arr = pl.StructArray(3, task_id=0, ki=0)

        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS

            cur_q_slot = q_l1_db.current()
            for ki in pl.range(0, skv_tiles):
                # Save current context
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.task_id = task_id
                ctx_curr.ki = ki

                # ========== compute_qk (current step) ==========
                skv_off = ki * TKV
                cur_k_slot = k_l1_db.current()
                qk_left = left_db.current()
                qk_right = right_db.current()
                qk_acc = acc_db.current()

                plm.load(cur_k_slot.tile, k, [skv_off, 0], layout="dn")
                if ki == 0:
                    plm.load(cur_q_slot.tile, q, [sq_off, 0])

                plm.move(qk_left.tile, cur_q_slot.tile)
                plm.move(qk_right.tile, cur_k_slot.tile)
                plm.matmul(qk_acc.tile, qk_left.tile, qk_right.tile)

                qk_slot = qk_vec_db.current()
                plm.move(qk_slot.tile, qk_acc.tile, acc_to_vec_mode="dual_split_m")

                qk_eid = task_id % FIFO_SIZE
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_eid], max_event_id=QK_MAX_EID)

                # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
                if task_id > 0:
                    ctx_pre = ctx_arr[(task_id + 2) % 3]
                    pv_eid = ctx_pre.task_id % FIFO_SIZE
                    sv_off = ctx_pre.ki * TKV

                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_eid], max_event_id=P_MAX_EID)

                    cur_v_slot = v_l1_db.current()
                    # # current() advances p_mat_db cursor to stay in sync with Vec,
                    cur_p_slot = p_mat_db.current()
                    pv_left = left_db.current()
                    pv_right = right_db.current()
                    pv_acc = acc_db.current()

                    plm.load(cur_v_slot.tile, v, [sv_off, 0])
                    plm.move(pv_left.tile, cur_p_slot.tile)
                    plm.move(pv_right.tile, cur_v_slot.tile)
                    plm.matmul(pv_acc.tile, pv_left.tile, pv_right.tile)

                    pv_slot = pv_vec_db.current()
                    plm.move(pv_slot.tile, pv_acc.tile, acc_to_vec_mode="dual_split_m")
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_eid], max_event_id=PV_MAX_EID)

                task_id = task_id + 1

        # Cube epilogue: drain last PV
        ctx_pre = ctx_arr[(task_id + 2) % 3]
        pv_eid = ctx_pre.task_id % FIFO_SIZE
        sv_off = ctx_pre.ki * TKV

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_eid], max_event_id=P_MAX_EID)

        cur_v_slot = v_l1_db.current()
        cur_p_slot = p_mat_db.current()
        pv_left = left_db.current()
        pv_right = right_db.current()
        pv_acc = acc_db.current()

        plm.load(cur_v_slot.tile, v, [sv_off, 0])
        plm.move(pv_left.tile, cur_p_slot.tile)
        plm.move(pv_right.tile, cur_v_slot.tile)
        plm.matmul(pv_acc.tile, pv_left.tile, pv_right.tile)

        pv_slot = pv_vec_db.current()
        plm.move(pv_slot.tile, pv_acc.tile, acc_to_vec_mode="dual_split_m")
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_eid], max_event_id=PV_MAX_EID)

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        # Vector-only buffers (independent buf_id space: 0-11)
        tmp_vec_buf = pl.UBBuffer(VA1, pl.FP32, shape=(TS_HALF, TKV))
        p_f16_buf = pl.UBBuffer(VA2, pl.FP16, shape=(TS_HALF, TKV), buf_id=1)
        reduce_dst_buf = pl.UBBuffer(VA3, pl.FP32, shape=(TS_HALF, 1), blayout=2)
        reduce_dst_rm_buf = pl.UBBuffer(VA3, pl.FP32, shape=(1, TS_HALF))

        running_o_buf = pl.UBBuffer(VA7, pl.FP32, shape=(TS_HALF, TD))
        o_f16_buf = pl.UBBuffer(VA9, pl.FP16, shape=(TS_HALF, TD), buf_id=10)
        tile_nz_buf = pl.UBBuffer(VA10, pl.FP16, shape=(64, 128), buf_id=11,
                                  blayout=2, slayout=1)

        # Double-buffered global state (per Q tile) — use tile tuples for dynamic
        # indexing by q_count % 2, since StructArray ctx references need runtime index.
        gmax_rm_type = plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32,
                                    target_memory=pl.MemorySpace.Vec)
        gmax_rm_0 = plm.make_tile(gmax_rm_type, addr=VA_GMAX0, size=VB_RED)
        gmax_rm_1 = plm.make_tile(gmax_rm_type, addr=VA_GMAX1, size=VB_RED)
        global_max_rm_buf = (gmax_rm_0, gmax_rm_1)

        gsum_type = plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32,
                                 target_memory=pl.MemorySpace.Vec, blayout=2)
        gsum_rm_type = plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32,
                                    target_memory=pl.MemorySpace.Vec)
        gsum_0 = plm.make_tile(gsum_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = plm.make_tile(gsum_type, addr=VA_GSUM1, size=VB_RED)
        gsum_rm_0 = plm.make_tile(gsum_rm_type, addr=VA_GSUM0, size=VB_RED)
        gsum_rm_1 = plm.make_tile(gsum_rm_type, addr=VA_GSUM1, size=VB_RED)
        global_sum_buf = (gsum_0, gsum_1)
        global_sum_rm_buf = (gsum_rm_0, gsum_rm_1)

        # FIFO exp_corr — use NBuffer with current() auto-rotate
        exp_corr_db = pl.UBNBuffer(VA_EXP_BASE, pl.FP32, shape=(TS_HALF, 1), num_slots=2, blayout=2)
        exp_corr_rm_db = pl.UBNBuffer(VA_EXP_BASE, pl.FP32, shape=(1, TS_HALF), num_slots=2)

        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * TS_HALF
        task_id = 0
        q_count = 0

        # StructArray(3) for pipeline context tracking (same as original)
        ctx_arr = pl.StructArray(3, sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0)

        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            q_idx = q_count % 2

            for ki in pl.range(0, skv_tiles):
                # Save current context
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.qi = qi
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count

                # --- compute_p (delayed 1 step: uses ctx from task_id-1) ---
                if task_id > 0:
                    ctx_p = ctx_arr[(task_id + 2) % 3]
                    p_eid = ctx_p.task_id % FIFO_SIZE
                    pl.system.wait_cross_core(pipe=pl.PipeType.V,
                        event_id=QK_READY_IDS[p_eid], max_event_id=QK_MAX_EID)

                    qk_slot = qk_vec_db.current()
                    q_idx_p = ctx_p.q_count % 2
                    gmax_p = global_max_rm_buf[q_idx_p]
                    gsum_rm_p = global_sum_rm_buf[q_idx_p]

                    if ctx_p.ki == 0:
                        plm.row_max(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
                        plm.row_expand_sub(tmp_vec_buf.tile, qk_slot.tile, reduce_dst_buf.tile)
                        plm.muls(gmax_p, reduce_dst_rm_buf.tile, 1.0)
                        plm.muls(tmp_vec_buf.tile, tmp_vec_buf.tile, SCALE)
                        plm.exp(qk_slot.tile, tmp_vec_buf.tile)
                        plm.row_sum(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
                        plm.muls(gsum_rm_p, reduce_dst_rm_buf.tile, 1.0)
                        plm.cast(p_f16_buf.tile, qk_slot.tile, target_type=pl.FP16, mode="round")
                    if ctx_p.ki > 0:
                        exp_corr_slot = exp_corr_rm_db.current()
                        plm.row_max(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
                        plm.maximum(reduce_dst_rm_buf.tile, reduce_dst_rm_buf.tile, gmax_p)
                        plm.sub(exp_corr_slot.tile, gmax_p, reduce_dst_rm_buf.tile)
                        plm.muls(gmax_p, reduce_dst_rm_buf.tile, 1.0)
                        plm.row_expand_sub(tmp_vec_buf.tile, qk_slot.tile, reduce_dst_buf.tile)
                        plm.muls(exp_corr_slot.tile, exp_corr_slot.tile, SCALE)
                        plm.muls(tmp_vec_buf.tile, tmp_vec_buf.tile, SCALE)
                        plm.exp(exp_corr_slot.tile, exp_corr_slot.tile)
                        plm.exp(qk_slot.tile, tmp_vec_buf.tile)
                        plm.cast(p_f16_buf.tile, qk_slot.tile, target_type=pl.FP16, mode="round")
                        plm.mul(gsum_rm_p, gsum_rm_p, exp_corr_slot.tile)
                        plm.row_sum(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
                        plm.add(gsum_rm_p, gsum_rm_p, reduce_dst_rm_buf.tile)

                    plm.move(tile_nz_buf.tile, p_f16_buf.tile)
                    cur_p_slot = p_mat_db.current()
                    plm.insert(cur_p_slot.tile, tile_nz_buf.tile, offset=(64 * sub_id * 32))
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
                        event_id=P_READY_IDS[p_eid], max_event_id=P_MAX_EID)

                # --- compute_gu (delayed 2 steps: uses ctx from task_id-2) ---
                if task_id > 1:
                    ctx_gu = ctx_arr[(task_id + 1) % 3]
                    gu_eid = ctx_gu.task_id % FIFO_SIZE
                    pl.system.wait_cross_core(pipe=pl.PipeType.V,
                        event_id=PV_READY_IDS[gu_eid], max_event_id=PV_MAX_EID)

                    pv_slot = pv_vec_db.current()
                    if ctx_gu.ki == 0:
                        plm.move(running_o_buf.tile, pv_slot.tile)
                    if ctx_gu.ki > 0:
                        exp_corr_gu = exp_corr_db.current()
                        plm.row_expand_mul(running_o_buf.tile, running_o_buf.tile, exp_corr_gu.tile)
                        plm.add(running_o_buf.tile, running_o_buf.tile, pv_slot.tile)
                    if ctx_gu.ki == ctx_gu.skv_tiles - 1:
                        gsum_gu = global_sum_buf[ctx_gu.q_count % 2]
                        plm.row_expand_div(running_o_buf.tile, running_o_buf.tile, gsum_gu)
                        plm.cast(o_f16_buf.tile, running_o_buf.tile, target_type=pl.FP16, mode="round")
                        plm.store(o, o_f16_buf.tile, [ctx_gu.sq_off + row_off, 0])

                task_id = task_id + 1
            q_count = q_count + 1

        # --- Vector epilogue: drain pipeline ---
        # compute_p for last task
        ctx_p = ctx_arr[(task_id + 2) % 3]
        p_eid = ctx_p.task_id % FIFO_SIZE
        pl.system.wait_cross_core(pipe=pl.PipeType.V,
            event_id=QK_READY_IDS[p_eid], max_event_id=QK_MAX_EID)

        qk_slot = qk_vec_db.current()
        q_idx_p = ctx_p.q_count % 2
        gmax_p = global_max_rm_buf[q_idx_p]
        gsum_rm_p = global_sum_rm_buf[q_idx_p]

        if ctx_p.ki == 0:
            plm.row_max(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
            plm.row_expand_sub(tmp_vec_buf.tile, qk_slot.tile, reduce_dst_buf.tile)
            plm.muls(gmax_p, reduce_dst_rm_buf.tile, 1.0)
            plm.muls(tmp_vec_buf.tile, tmp_vec_buf.tile, SCALE)
            plm.exp(qk_slot.tile, tmp_vec_buf.tile)
            plm.row_sum(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
            plm.muls(gsum_rm_p, reduce_dst_rm_buf.tile, 1.0)
            plm.cast(p_f16_buf.tile, qk_slot.tile, target_type=pl.FP16, mode="round")
        if ctx_p.ki > 0:
            exp_corr_slot = exp_corr_rm_db.current()
            plm.row_max(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
            plm.maximum(reduce_dst_rm_buf.tile, reduce_dst_rm_buf.tile, gmax_p)
            plm.sub(exp_corr_slot.tile, gmax_p, reduce_dst_rm_buf.tile)
            plm.muls(gmax_p, reduce_dst_rm_buf.tile, 1.0)
            plm.row_expand_sub(tmp_vec_buf.tile, qk_slot.tile, reduce_dst_buf.tile)
            plm.muls(exp_corr_slot.tile, exp_corr_slot.tile, SCALE)
            plm.muls(tmp_vec_buf.tile, tmp_vec_buf.tile, SCALE)
            plm.exp(exp_corr_slot.tile, exp_corr_slot.tile)
            plm.exp(qk_slot.tile, tmp_vec_buf.tile)
            plm.cast(p_f16_buf.tile, qk_slot.tile, target_type=pl.FP16, mode="round")
            plm.mul(gsum_rm_p, gsum_rm_p, exp_corr_slot.tile)
            plm.row_sum(reduce_dst_buf.tile, qk_slot.tile, tmp_vec_buf.tile)
            plm.add(gsum_rm_p, gsum_rm_p, reduce_dst_rm_buf.tile)

        plm.move(tile_nz_buf.tile, p_f16_buf.tile)
        cur_p_slot = p_mat_db.current()
        plm.insert(cur_p_slot.tile, tile_nz_buf.tile, offset=(64 * sub_id * 32))
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3,
            event_id=P_READY_IDS[p_eid], max_event_id=P_MAX_EID)

        # compute_gu for task_id-2
        if task_id > 1:
            ctx_gu = ctx_arr[(task_id + 1) % 3]
            gu_eid = ctx_gu.task_id % FIFO_SIZE
            pl.system.wait_cross_core(pipe=pl.PipeType.V,
                event_id=PV_READY_IDS[gu_eid], max_event_id=PV_MAX_EID)
            pv_slot = pv_vec_db.current()
            if ctx_gu.ki == 0:
                plm.move(running_o_buf.tile, pv_slot.tile)
            if ctx_gu.ki > 0:
                exp_corr_gu = exp_corr_db.current()
                plm.row_expand_mul(running_o_buf.tile, running_o_buf.tile, exp_corr_gu.tile)
                plm.add(running_o_buf.tile, running_o_buf.tile, pv_slot.tile)
            if ctx_gu.ki == ctx_gu.skv_tiles - 1:
                gsum_gu = global_sum_buf[ctx_gu.q_count % 2]
                plm.row_expand_div(running_o_buf.tile, running_o_buf.tile, gsum_gu)
                plm.cast(o_f16_buf.tile, running_o_buf.tile, target_type=pl.FP16, mode="round")
                plm.store(o, o_f16_buf.tile, [ctx_gu.sq_off + row_off, 0])

        task_id = task_id + 1

        # Final compute_gu for task_id-2 (last task)
        ctx_gu = ctx_arr[(task_id + 1) % 3]
        gu_eid = ctx_gu.task_id % FIFO_SIZE
        pl.system.wait_cross_core(pipe=pl.PipeType.V,
            event_id=PV_READY_IDS[gu_eid], max_event_id=PV_MAX_EID)
        pv_slot = pv_vec_db.current()
        if ctx_gu.ki == 0:
            plm.move(running_o_buf.tile, pv_slot.tile)
        if ctx_gu.ki > 0:
            exp_corr_gu = exp_corr_db.current()
            plm.row_expand_mul(running_o_buf.tile, running_o_buf.tile, exp_corr_gu.tile)
            plm.add(running_o_buf.tile, running_o_buf.tile, pv_slot.tile)
        if ctx_gu.ki == ctx_gu.skv_tiles - 1:
            gsum_gu = global_sum_buf[ctx_gu.q_count % 2]
            plm.row_expand_div(running_o_buf.tile, running_o_buf.tile, gsum_gu)
            plm.cast(o_f16_buf.tile, running_o_buf.tile, target_type=pl.FP16, mode="round")
            plm.store(o, o_f16_buf.tile, [ctx_gu.sq_off + row_off, 0])

    return o


# ================================================================
#  Reference + Tests
# ================================================================
def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T)
    scale = qk * scale_val
    max_val = torch.max(scale, dim=-1, keepdim=True).values
    x_sub = scale - max_val
    x_exp = torch.exp(x_sub)
    attn = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
    return qk, x_exp, torch.matmul(attn, v.float()).half()


def test_fa_perf_nbuf():
    compiled = fe.compile(fa_perf_tkv_preload_nbuf_kernel, arch="a5", codegen_mode="cce")
    print("compiled:", compiled.lib_path)
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    for sq, skv, d, num_cores in [
        (8192, 8192, TD, 28),
    ]:
        print(f"\nFA-Perf-NBuffer ({sq},{skv},{d}) cores={num_cores} QK_PRELOAD={QK_PRELOAD}")
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)

        fe.launch(None, num_cores, compiled, q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
        torch.npu.synchronize()

        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        diff = (o_t - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        print("  PASS")


if __name__ == "__main__":
    print(f"FA perf with NBuffer + auto_mutex (QK_PRELOAD={QK_PRELOAD}, FIFO={FIFO_SIZE})")
    print("=" * 60)
    test_fa_perf_nbuf()  # uncomment on A5 NPU
    print("\nParse test passed!")
