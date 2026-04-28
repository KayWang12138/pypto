import math
import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)
FIFO_SIZE = 2

# Cube tile byte sizes
Q_F16 = TS * TD * 2        # [TS,  TD]  FP16 = 32KB (DN: stored as [TD, TS])
KT_F16 = TKV * TD * 2      # [TKV, TD]  FP16 = 32KB
V_F16 = TKV * TD * 2       # [TKV, TD]  FP16 = 32KB
P_F16 = TS * TKV * 2       # [TS,  TKV] FP16 = 32KB
QK_F32 = TKV * TS * 4      # [TKV, TS]  FP32 = 64KB (DN acc shape)
PV_F32 = TS * TD * 4       # [TS,  TD]  FP32 = 64KB
PV_CORE_STRIDE = 2 * FIFO_SIZE * TS

# ---- MAT (512KB) ----
MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16 * 2
MA3_PONG = MA3 + V_F16

# DN: Left holds K [TKV, TD], Right holds Q^T [TD, TS].
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_F32

# ---- VEC addresses (248KB on a5) ----
VB4_KV = TKV * TS_HALF * 4          # [TKV, TS_HALF] FP32 = 32KB
VB2_KV = TKV * TS_HALF * 2          # [TKV, TS_HALF] FP16 = 16KB
VB6_DN = (TKV + 1) * TS_HALF * 2    # [TKV+1, TS_HALF] FP16 = 16512B
VB1_KV = TKV * TS_HALF              # [TKV, TS_HALF] UINT8 = 8KB
VB4 = TS_HALF * TD * 4              # [TS_HALF, TD] FP32 = 32KB
VB2 = TS_HALF * TD * 2              # [TS_HALF, TD] FP16 = 16KB
VB_RED = TS_HALF * 4                # [TS_HALF, 1] FP32 = 256B


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


VA0 = 0                             # qk_vec  [TKV, TS_HALF] FP32
VA1 = _align_up(VA0 + VB4_KV)       # tmp_vec [TKV, TS_HALF] FP32
VA2 = _align_up(VA1 + VB4_KV)       # p_f16   [TKV, TS_HALF] FP16
VA3 = _align_up(VA2 + VB2_KV)       # reduce_dst / reduce_dst_rm
# Keep the whole VEC layout on 1KB boundaries; A5 CCE is stricter about
# VEC address alignment than the older A3 path.
VA_GMAX0 = _align_up(VA3 + VB_RED)          # global_max slot 0
VA_GMAX1 = _align_up(VA_GMAX0 + VB_RED)     # global_max slot 1
VA_GSUM0 = _align_up(VA_GMAX1 + VB_RED)     # global_sum slot 0
VA_GSUM1 = _align_up(VA_GSUM0 + VB_RED)     # global_sum slot 1
VA_EXP0 = _align_up(VA_GSUM1 + VB_RED)      # exp_corr slot 0
VA_EXP1 = _align_up(VA_EXP0 + VB_RED)       # exp_corr slot 1
VA7 = _align_up(VA_EXP1 + VB_RED)           # running_o [TS_HALF, TD] FP32
VA8 = _align_up(VA7 + VB4)                  # pv_vec    [TS_HALF, TD] FP32
VA9 = _align_up(VA8 + VB4)                  # o_f16     [TS_HALF, TD] FP16
VA10 = _align_up(VA9 + VB2)                 # tile_nz   [TKV+1, TS_HALF] FP16
VA11 = _align_up(VA10 + VB6_DN)             # qk_vec1   [TKV, TS_HALF] FP32
VA12 = _align_up(VA11 + VB4_KV)             # pv_vec1   [TS_HALF, TD] FP32
assert VA12 + VB4 <= 248 * 1024, f"VEC overflow: {VA12 + VB4} > {248 * 1024}"

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0, 1)
P_READY_IDS = (2, 3)
PV_READY_IDS = (4, 5)
QK_MAX_EID = 2
P_MAX_EID = 4
PV_MAX_EID = 6

B = pl.DynVar("B")
N = pl.DynVar("N")
Sq2 = pl.DynVar("Sq")
Skv2 = pl.DynVar("Skv")
D2 = pl.DynVar("D")
NumRanges = pl.DynVar("NumRanges")


def alloc_cube_buffer():
    q_mat_type = plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    q_mat_0 = plm.make_tile(q_mat_type, addr=MA0, size=Q_F16)
    q_mat_1 = plm.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)
    k_mat_type = plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
    k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)
    v_mat_type = plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
    v_mat_0 = plm.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = plm.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)
    left_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1), addr=LA0, size=KT_F16)
    left_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1), addr=LA1, size=KT_F16)
    right_0 = plm.make_tile(plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=Q_F16)
    right_1 = plm.make_tile(plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=Q_F16)
    acc_0 = plm.make_tile(plm.TileType(shape=[TKV, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_F32)
    acc_1 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_F32)
    return (q_mat_0, q_mat_1), (k_mat_0, k_mat_1), (v_mat_0, v_mat_1), (left_0, left_1), (right_0, right_1), acc_0, acc_1


def compute_qk(ctx, state, q, k, q_mat_buf, k_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2, qk_vec, qk_vec1):
    qk_slot = ctx.task_id % FIFO_SIZE
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_01[buf_idx])
    if ctx.ki == 0:
        plm.load_tile(q_mat_buf[ctx.q_count % 2], q, [ctx.b_idx, ctx.qi, ctx.n_idx, 0], layout="dn", tile_dims=[1, 3])
    plm.load_tile(k_mat_buf[buf_idx], k, [ctx.b_idx, ctx.ki, ctx.n_idx, 0], tile_dims=[1, 3])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[state.l0ab_idx])
    plm.move(left_buf[state.l0ab_idx], k_mat_buf[buf_idx])
    plm.move(right_buf[state.l0ab_idx], q_mat_buf[ctx.q_count % 2])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_01[buf_idx])

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[state.l0c_idx])
    if state.l0c_idx == 0:
        plm.matmul(acc_buf1, left_buf[state.l0ab_idx], right_buf[state.l0ab_idx])
    else:
        plm.matmul(acc_buf2, left_buf[state.l0ab_idx], right_buf[state.l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[state.l0ab_idx])

    if qk_slot == 0:
        if state.l0c_idx == 0:
            plm.move(qk_vec, acc_buf1, acc_to_vec_mode="dual_split_n")
        else:
            plm.move(qk_vec, acc_buf2, acc_to_vec_mode="dual_split_n")
    else:
        if state.l0c_idx == 0:
            plm.move(qk_vec1, acc_buf1, acc_to_vec_mode="dual_split_n")
        else:
            plm.move(qk_vec1, acc_buf2, acc_to_vec_mode="dual_split_n")
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[state.l0c_idx])
    state.l0ab_idx = 1 - state.l0ab_idx
    state.l0c_idx = 1 - state.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_slot], max_event_id=QK_MAX_EID)


def compute_pv(ctx, state, v, pv_vec, pv_vec1, p_mat_buf1, p_mat_buf2, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2):
    pv_slot = ctx.task_id % FIFO_SIZE
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_slot], max_event_id=P_MAX_EID)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_23[buf_idx])
    plm.load_tile(v_mat_buf[buf_idx], v, [ctx.b_idx, ctx.ki, ctx.n_idx, 0], tile_dims=[1, 3])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[state.l0ab_idx])
    if buf_idx == 0:
        plm.move(left_buf[state.l0ab_idx], p_mat_buf1)
    else:
        plm.move(left_buf[state.l0ab_idx], p_mat_buf2)
    plm.move(right_buf[state.l0ab_idx], v_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_23[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[state.l0c_idx])
    if state.l0c_idx == 0:
        plm.matmul(acc_buf1, left_buf[state.l0ab_idx], right_buf[state.l0ab_idx])
    else:
        plm.matmul(acc_buf2, left_buf[state.l0ab_idx], right_buf[state.l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=EVENT_IDS_01[state.l0ab_idx])

    if pv_slot == 0:
        if state.l0c_idx == 0:
            plm.move(pv_vec, acc_buf1, acc_to_vec_mode="dual_split_m")
        else:
            plm.move(pv_vec, acc_buf2, acc_to_vec_mode="dual_split_m")
    else:
        if state.l0c_idx == 0:
            plm.move(pv_vec1, acc_buf1, acc_to_vec_mode="dual_split_m")
        else:
            plm.move(pv_vec1, acc_buf2, acc_to_vec_mode="dual_split_m")
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[state.l0c_idx])
    state.l0ab_idx = 1 - state.l0ab_idx
    state.l0c_idx = 1 - state.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_slot], max_event_id=PV_MAX_EID)

@pl.inline
def softmax_body(ctx, sq_dim, skv_dim, qk_vec, qk_vec1, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf, global_sum_rm_buf,
                 exp_corr_rm, exp_corr_rm1,
                 tile_nz, p_mat_buf1, p_mat_buf2, p_buf, p_f16_store):
    p_slot = ctx.task_id % FIFO_SIZE
    q_idx = ctx.q_count % 2
    global_max_rm = global_max_rm_buf[q_idx]
    global_sum_rm = global_sum_rm_buf[q_idx]
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    if p_slot == 0:
        if ctx.ki == 0:
            plm.col_max(reduce_dst_rm, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.col_expand_sub(tmp_vec, qk_vec, reduce_dst_rm)
            plm.muls(global_max_rm, reduce_dst_rm, 1.0)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.col_sum(reduce_dst_rm, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.muls(global_sum_rm, reduce_dst_rm, 1.0)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
        else:
            plm.col_max(reduce_dst_rm, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
            pl.system.bar_v()
            plm.sub(exp_corr_rm, global_max_rm, reduce_dst_rm)
            pl.system.bar_v()
            plm.muls(global_max_rm, reduce_dst_rm, 1.0)
            pl.system.bar_v()
            plm.col_expand_sub(tmp_vec, qk_vec, reduce_dst_rm)
            plm.muls(exp_corr_rm, exp_corr_rm, SCALE)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(exp_corr_rm, exp_corr_rm)
            plm.exp(qk_vec, tmp_vec)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            pl.system.bar_v()
            plm.mul(global_sum_rm, global_sum_rm, exp_corr_rm)
            plm.col_sum(reduce_dst_rm, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.add(global_sum_rm, global_sum_rm, reduce_dst_rm)
    else:
        if ctx.ki == 0:
            plm.col_max(reduce_dst_rm, qk_vec1, tmp_vec)
            pl.system.bar_v()
            plm.col_expand_sub(tmp_vec, qk_vec1, reduce_dst_rm)
            plm.muls(global_max_rm, reduce_dst_rm, 1.0)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec1, tmp_vec)
            pl.system.bar_v()
            plm.col_sum(reduce_dst_rm, qk_vec1, tmp_vec)
            pl.system.bar_v()
            plm.muls(global_sum_rm, reduce_dst_rm, 1.0)
            plm.cast(p_f16, qk_vec1, target_type=pl.FP16, mode="round")
        else:
            plm.col_max(reduce_dst_rm, qk_vec1, tmp_vec)
            pl.system.bar_v()
            plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
            pl.system.bar_v()
            plm.sub(exp_corr_rm1, global_max_rm, reduce_dst_rm)
            pl.system.bar_v()
            plm.muls(global_max_rm, reduce_dst_rm, 1.0)
            pl.system.bar_v()
            plm.col_expand_sub(tmp_vec, qk_vec1, reduce_dst_rm)
            plm.muls(exp_corr_rm1, exp_corr_rm1, SCALE)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(exp_corr_rm1, exp_corr_rm1)
            plm.exp(qk_vec1, tmp_vec)
            plm.cast(p_f16, qk_vec1, target_type=pl.FP16, mode="round")
            pl.system.bar_v()
            plm.mul(global_sum_rm, global_sum_rm, exp_corr_rm1)
            plm.col_sum(reduce_dst_rm, qk_vec1, tmp_vec)
            pl.system.bar_v()
            plm.add(global_sum_rm, global_sum_rm, reduce_dst_rm)

    plm.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        plm.insert(p_mat_buf1, tile_nz, index_col=TS_HALF * ctx.sub_id)
    else:
        plm.insert(p_mat_buf2, tile_nz, index_col=TS_HALF * ctx.sub_id)
    plm.store_tile(p_buf, p_f16_store, [ctx.b_idx, ctx.qi * 2 + ctx.sub_id, ctx.n_idx, ctx.ki], tile_dims=[1, 3])


@pl.inline
def compute_p(ctx, sq_dim, skv_dim, qk_vec, qk_vec1, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf, global_sum_rm_buf,
              exp_corr_rm, exp_corr_rm1,
              tile_nz, p_mat_buf1, p_mat_buf2, p_buf, p_f16_store):
    p_slot = ctx.task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_slot], max_event_id=QK_MAX_EID)
    softmax_body(ctx, sq_dim, skv_dim, qk_vec, qk_vec1, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf, global_sum_rm_buf,
                 exp_corr_rm, exp_corr_rm1,
                 tile_nz, p_mat_buf1, p_mat_buf2, p_buf, p_f16_store)
    # Both vector subblocks contribute one half of the shared P MAT tile. Make
    # sure both halves have finished TINSERT before cube starts PV.
    pl.system.bar_all()
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_slot], max_event_id=P_MAX_EID)


def compute_gu(ctx, o, pv_vec, pv_vec1, running_o, exp_corr, exp_corr1, global_sum_buf, o_f16):
    pv_slot = ctx.task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot], max_event_id=PV_MAX_EID)
    if pv_slot == 0:
        if ctx.ki == 0:
            plm.move(running_o, pv_vec)
        else:
            plm.row_expand_mul(running_o, running_o, exp_corr)
            plm.add(running_o, running_o, pv_vec)
    else:
        if ctx.ki == 0:
            plm.move(running_o, pv_vec1)
        else:
            plm.row_expand_mul(running_o, running_o, exp_corr1)
            plm.add(running_o, running_o, pv_vec1)
    if ctx.ki == ctx.skv_tiles - 1:
        plm.row_expand_div(running_o, running_o, global_sum_buf[ctx.q_count % 2])
        plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store_tile(o, o_f16, [ctx.b_idx, ctx.qi * 2 + ctx.sub_id, ctx.n_idx, 0], tile_dims=[1, 3])


@fe.kernel
def fa_bsnd_dn_kernel(
    q: pl.Tensor[[B, Sq2, N, D2], pl.FP16],
    k: pl.Tensor[[B, Skv2, N, D2], pl.FP16],
    v: pl.Tensor[[B, Skv2, N, D2], pl.FP16],
    o: pl.Tensor[[B, Sq2, N, D2], pl.FP16],
    p_buf: pl.Tensor[[B, Sq2, N, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, D2], pl.FP32],
    work_ranges: pl.Tensor[[NumRanges, 2], pl.INT32],
) -> pl.Tensor[[B, Sq2, N, D2], pl.FP16]:
    sq_dim = Sq2
    skv_dim = Skv2
    sq_tiles = (sq_dim + TS - 1) // TS
    skv_tiles = (skv_dim + TKV - 1) // TKV
    core_id = pl.block.index_cast(pl.block.get_block_idx())
    n_dim = N

    qk_vec = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_KV)
    tmp_vec = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)
    # Keep the source descriptor identical to the proven DN perf kernel.
    # TMOV(tile_nz, p_f16) is sensitive to the Vec tile descriptor here.
    p_f16 = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)
    p_f16_store = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)
    reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)
    red_rm_type = plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    red_type = plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2)
    global_max_rm_buf = (
        plm.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED),
        plm.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED),
    )
    global_sum_buf = (
        plm.make_tile(red_type, addr=VA_GSUM0, size=VB_RED),
        plm.make_tile(red_type, addr=VA_GSUM1, size=VB_RED),
    )
    global_sum_rm_buf = (
        plm.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED),
        plm.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED),
    )
    exp_corr = plm.make_tile(red_type, addr=VA_EXP0, size=VB_RED)
    exp_corr_rm = plm.make_tile(red_rm_type, addr=VA_EXP0, size=VB_RED)
    exp_corr1 = plm.make_tile(red_type, addr=VA_EXP1, size=VB_RED)
    exp_corr_rm1 = plm.make_tile(red_rm_type, addr=VA_EXP1, size=VB_RED)
    running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
    pv_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
    o_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)
    tile_nz = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec,
                                         blayout=2, slayout=1), addr=VA10, size=VB6_DN)
    qk_vec1 = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB4_KV)
    pv_vec1 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA12, size=VB4)
    p_mat_type = plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    p_mat_buf1 = plm.make_tile(p_mat_type, addr=MA2, size=P_F16)
    p_mat_buf2 = plm.make_tile(p_mat_type, addr=MA2_PONG, size=P_F16)

    with pl.section_cube():
        q_mat_buf, k_mat_buf, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2 = alloc_cube_buffer()
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

        work_start = pl.block.index_cast(pl.read(work_ranges, [core_id, 0]))
        work_end = pl.block.index_cast(pl.read(work_ranges, [core_id, 1]))
        for work_id in pl.range(work_start, work_end):
            task_id = 0
            q_count = 0
            state = pl.struct(l0ab_idx=0, l0c_idx=0)
            ctx_arr = pl.StructArray(3, b_idx=0, n_idx=0, qi=0, ki=0, skv_tiles=0, q_count=0, sub_id=0, task_id=0)
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                for ki in pl.range(0, skv_tiles):
                    ctx_curr = ctx_arr[task_id % 3]
                    ctx_curr.b_idx = b_idx
                    ctx_curr.n_idx = n_idx
                    ctx_curr.qi = qi
                    ctx_curr.ki = ki
                    ctx_curr.skv_tiles = skv_tiles
                    ctx_curr.q_count = q_count
                    ctx_curr.sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
                    ctx_curr.task_id = task_id
                    compute_qk(ctx_curr, state, q, k, q_mat_buf, k_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2, qk_vec, qk_vec1)
                    if task_id > 0:
                        compute_pv(ctx_arr[(task_id + 2) % 3], state, v, pv_vec, pv_vec1, p_mat_buf1, p_mat_buf2, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2)
                    task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                compute_pv(ctx_arr[(task_id + 2) % 3], state, v, pv_vec, pv_vec1, p_mat_buf1, p_mat_buf2, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2)

    with pl.section_vector():
        work_start = pl.block.index_cast(pl.read(work_ranges, [core_id, 0]))
        work_end = pl.block.index_cast(pl.read(work_ranges, [core_id, 1]))
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        for work_id in pl.range(work_start, work_end):
            task_id = 0
            q_count = 0
            ctx_arr = pl.StructArray(3, b_idx=0, n_idx=0, qi=0, ki=0, skv_tiles=0, q_count=0, sub_id=0, task_id=0)
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                for ki in pl.range(0, skv_tiles):
                    ctx_curr = ctx_arr[task_id % 3]
                    ctx_curr.b_idx = b_idx
                    ctx_curr.n_idx = n_idx
                    ctx_curr.qi = qi
                    ctx_curr.ki = ki
                    ctx_curr.skv_tiles = skv_tiles
                    ctx_curr.q_count = q_count
                    ctx_curr.sub_id = sub_id
                    ctx_curr.task_id = task_id
                    if task_id > 0:
                        compute_p(ctx_arr[(task_id + 2) % 3], sq_dim, skv_dim, qk_vec, qk_vec1, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf,
                                      global_sum_rm_buf, exp_corr_rm, exp_corr_rm1, tile_nz, p_mat_buf1, p_mat_buf2, p_buf,
                                      p_f16_store)
                    if task_id > 1:
                        compute_gu(ctx_arr[(task_id + 1) % 3], o, pv_vec, pv_vec1, running_o, exp_corr, exp_corr1, global_sum_buf, o_f16)
                    task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                compute_p(ctx_arr[(task_id + 2) % 3], sq_dim, skv_dim, qk_vec, qk_vec1, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf,
                                      global_sum_rm_buf, exp_corr_rm, exp_corr_rm1, tile_nz, p_mat_buf1, p_mat_buf2, p_buf,
                                      p_f16_store)
                if task_id > 1:
                    compute_gu(ctx_arr[(task_id + 1) % 3], o, pv_vec, pv_vec1, running_o, exp_corr, exp_corr1, global_sum_buf, o_f16)
                compute_gu(ctx_arr[(task_id + 2) % 3], o, pv_vec, pv_vec1, running_o, exp_corr, exp_corr1, global_sum_buf, o_f16)
    return o


def flash_attention_ref_bs(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, sq, n, _ = q.shape
    _, skv, _, _ = k.shape
    o_ref = torch.zeros_like(q)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, :, ni, :].float(), k[bi, :, ni, :].float().T) * scale_val
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, :, ni, :] = torch.matmul(attn, v[bi, :, ni, :].float()).half()
    return o_ref


def test_fa_bs_a5():
    compiled = fe.compile(fa_bsnd_dn_kernel, arch="a5", codegen_mode="cce", timeout=120)
    if compiled is None:
        raise RuntimeError("compile failed; check error.txt for fa_bsnd_dn_kernel")
    print("compiled:", compiled.lib_path)
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for b, sq, n, skv, d, num_cores in [
        (1, 128, 1, 128, TD, 1),
        (1, 128, 2, 128, TD, 2),
        (2, 1024, 3, 1024, TD, 24),
        (1, 8192, 2, 8192, TD, 24),
        (8, 256, 2, 256, TD, 6),
        (2, 512, 1, 256, TD, 4),
    ]:
        print(f"\nFA-BSND-DN-A5 (b={b}, sq={sq}, n={n}, skv={skv}, d={d}) cores={num_cores}")
        q = torch.rand((b, sq, n, d), device=device, dtype=torch.float16)
        k = torch.rand((b, skv, n, d), device=device, dtype=torch.float16)
        v = torch.rand((b, skv, n, d), device=device, dtype=torch.float16)
        o = torch.zeros((b, sq, n, d), device=device, dtype=torch.float16)
        p_buf = torch.zeros((b, sq, n, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        total_work = b * n
        work_ranges = torch.zeros((num_cores, 2), device=device, dtype=torch.int32)
        work_per_core = (total_work + num_cores - 1) // num_cores
        for core in range(num_cores):
            work_ranges[core, 0] = core * work_per_core
            work_ranges[core, 1] = min((core + 1) * work_per_core, total_work)
        actual_num_cores = min(num_cores, total_work)
        fe.launch(None, actual_num_cores, compiled, q, k, v, o, p_buf, pv_buf, work_ranges)
        torch.npu.synchronize()
        o_ref = flash_attention_ref_bs(q, k, v, d)
        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o, o_ref, rtol=5e-3, atol=5e-3)
        print("  PASS")


if __name__ == "__main__":
    print("FA BSND DN on A5 CCE")
    print("=" * 60)
    test_fa_bs_a5()
    print("\nAll FlashAttention DN tests passed!")

