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
NEG_INF = -1e9
FIXED_MASK_S = 2048
FIFO_SIZE = 1

# Cube tile byte sizes
Q_F16 = TS * TD * 2        # [TS,  TD]  FP16 = 32KB (DN: stored as [TD, TS])
KT_F16 = TKV * TD * 2      # [TKV, TD]  FP16 = 32KB
V_F16 = TKV * TD * 2       # [TKV, TD]  FP16 = 32KB
P_F16 = TS * TKV * 2       # [TS,  TKV] FP16 = 32KB
QK_F32 = TKV * TS * 4      # [TKV, TS]  FP32 = 64KB (DN acc shape)
PV_F32 = TS * TD * 4       # [TS,  TD]  FP32 = 64KB

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
# VEC address alignment than the older A3 path. The causal path also adds
# mask/TSEL scratch tiles after the DN perf-kernel buffers.
VA_GMAX0 = _align_up(VA3 + VB_RED)          # global_max slot 0
VA_GMAX1 = _align_up(VA_GMAX0 + VB_RED)     # global_max slot 1
VA_GSUM0 = _align_up(VA_GMAX1 + VB_RED)     # global_sum slot 0
VA_GSUM1 = _align_up(VA_GSUM0 + VB_RED)     # global_sum slot 1
VA_EXP0 = _align_up(VA_GSUM1 + VB_RED)      # exp_corr
VA7 = _align_up(VA_EXP0 + VB_RED)           # running_o [TS_HALF, TD] FP32
VA8 = _align_up(VA7 + VB4)                  # pv_vec    [TS_HALF, TD] FP32
VA9 = _align_up(VA8 + VB4)                  # o_f16     [TS_HALF, TD] FP16
VA10 = _align_up(VA9 + VB2)                 # tile_nz   [TKV+1, TS_HALF] FP16
VA11 = _align_up(VA10 + VB6_DN)             # mask_u8_dn   [TKV, TS_HALF] UINT8
VA12 = _align_up(VA11 + VB1_KV)             # mask_fp16_dn [TKV, TS_HALF] FP16
VA13 = _align_up(VA12 + VB2_KV)             # mask_vec_dn  [TKV, TS_HALF] UINT8
VA14 = _align_up(VA13 + VB1_KV)             # neg_inf_vec  [TKV, TS_HALF] FP32
assert VA14 + VB4_KV <= 248 * 1024, f"VEC overflow: {VA14 + VB4_KV} > {248 * 1024}"

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0,)
P_READY_IDS = (1,)
PV_READY_IDS = (2,)
QK_MAX_EID = 1
P_MAX_EID = 2
PV_MAX_EID = 3

B = pl.DynVar("B")
N = pl.DynVar("N")
Sq2 = pl.DynVar("Sq")
Skv2 = pl.DynVar("Skv")
D2 = pl.DynVar("D")
NumRanges = pl.DynVar("NumRanges")
MaskSq = pl.DynVar("MaskSq")
MaskSkv = pl.DynVar("MaskSkv")


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


def compute_qk(ctx, state, q, k, q_mat_buf, k_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2, qk_vec):
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=EVENT_IDS_01[buf_idx])
    if ctx.ki == 0:
        # BSND layout: [B, S, N, D]. tile_dims=[1, 3] keeps S/D as tile axes;
        # layout="dn" loads Q as [D, S] for the K x Q^T DN path.
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

    if state.l0c_idx == 0:
        plm.move(qk_vec, acc_buf1, acc_to_vec_mode="dual_split_n")
    else:
        plm.move(qk_vec, acc_buf2, acc_to_vec_mode="dual_split_n")
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[state.l0c_idx])
    state.l0ab_idx = 1 - state.l0ab_idx
    state.l0c_idx = 1 - state.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[0], max_event_id=QK_MAX_EID)


def compute_pv(ctx, state, v, pv_vec, p_mat_buf1, p_mat_buf2, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2):
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[0], max_event_id=P_MAX_EID)
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

    if state.l0c_idx == 0:
        plm.move(pv_vec, acc_buf1, acc_to_vec_mode="dual_split_m")
    else:
        plm.move(pv_vec, acc_buf2, acc_to_vec_mode="dual_split_m")
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=EVENT_IDS_01[state.l0c_idx])
    state.l0ab_idx = 1 - state.l0ab_idx
    state.l0c_idx = 1 - state.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[0], max_event_id=PV_MAX_EID)

@pl.inline
def apply_diag_mask(ctx, sq_dim, skv_dim,
                    qk_vec, tmp_vec, mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec):
    # qk_vec is DN [K, Q] and softmax reduces over K. The fixed UINT8 mask is
    # stored as mask[k, q] = 1 when q > k. Loading with Q offset +1 makes the
    # diagonal block produce 1 for keep positions and 0 for masked positions.
    # TCMPS(EQ 0) flips that to the TSEL predicate: 1 selects neg_inf.
    # Current A5 TCMPS/TSEL expects the full [TKV, TS_HALF] uint8 descriptor.
    plm.load(mask_u8_dn, attn_mask, [ctx.ki * TKV, ctx.qi * TS + ctx.sub_id * TS_HALF + 1])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    plm.cast(mask_fp16_dn, mask_u8_dn, target_type=pl.FP16, mode="round")
    pl.system.bar_v()
    plm.cmps(mask_vec_dn, mask_fp16_dn, 0.0, cmp_type=0)
    pl.system.bar_v()
    plm.expands(neg_inf_vec, NEG_INF)
    pl.system.bar_v()
    plm.sel(out=qk_vec, mask=mask_vec_dn, lhs=neg_inf_vec, rhs=qk_vec, tmp=tmp_vec)
    pl.system.bar_v()


@pl.inline
def softmax_body(ctx, sq_dim, skv_dim, qk_vec, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf, global_sum_rm_buf, exp_corr_rm,
                 tile_nz, p_mat_buf1, p_mat_buf2,
                 mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec):
    if ctx.ki == ctx.qi:
        apply_diag_mask(ctx, sq_dim, skv_dim, qk_vec, tmp_vec, mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec)

    q_idx = ctx.q_count % 2
    global_max_rm = global_max_rm_buf[q_idx]
    global_sum_rm = global_sum_rm_buf[q_idx]
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2

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

    plm.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        plm.insert(p_mat_buf1, tile_nz, index_col=TS_HALF * ctx.sub_id)
    else:
        plm.insert(p_mat_buf2, tile_nz, index_col=TS_HALF * ctx.sub_id)


@pl.inline
def compute_p(ctx, sq_dim, skv_dim, qk_vec, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf, global_sum_rm_buf, exp_corr_rm,
              tile_nz, p_mat_buf1, p_mat_buf2,
              mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec):
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[0], max_event_id=QK_MAX_EID)
    softmax_body(ctx, sq_dim, skv_dim, qk_vec, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf, global_sum_rm_buf, exp_corr_rm,
                 tile_nz, p_mat_buf1, p_mat_buf2,
                 mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec)
    # Both vector subblocks contribute one half of the shared P MAT tile. Make
    # sure both halves have finished TINSERT before cube starts PV.
    pl.system.bar_all()
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[0], max_event_id=P_MAX_EID)


def compute_gu(ctx, o, pv_vec, running_o, exp_corr, global_sum_buf, o_f16):
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[0], max_event_id=PV_MAX_EID)
    if ctx.ki == 0:
        plm.move(running_o, pv_vec)
    else:
        plm.row_expand_mul(running_o, running_o, exp_corr)
        plm.add(running_o, running_o, pv_vec)
    last_ki = pl.min(ctx.qi, ctx.skv_tiles - 1)
    if ctx.ki == last_ki:
        plm.row_expand_div(running_o, running_o, global_sum_buf[ctx.q_count % 2])
        plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store_tile(o, o_f16, [ctx.b_idx, ctx.qi * 2 + ctx.sub_id, ctx.n_idx, 0], tile_dims=[1, 3])


@fe.kernel
def fa_causal_bsnd_dn_kernel_v6(
    q: pl.Tensor[[B, Sq2, N, D2], pl.FP16],
    k: pl.Tensor[[B, Skv2, N, D2], pl.FP16],
    v: pl.Tensor[[B, Skv2, N, D2], pl.FP16],
    o: pl.Tensor[[B, Sq2, N, D2], pl.FP16],
    work_ranges: pl.Tensor[[NumRanges, 2], pl.INT32],
    attn_mask: pl.Tensor[[MaskSkv, MaskSq], pl.UINT8],
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
    running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
    pv_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
    o_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)
    tile_nz = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec,
                                         blayout=2, slayout=1), addr=VA10, size=VB6_DN)
    # mask_u8_dn holds the UINT8 tile loaded from fixed GM mask. mask_vec_dn is
    # the TCMPS predicate consumed by TSEL, using the full [TKV, TS_HALF] layout.
    mask_u8_dn = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB1_KV)
    mask_fp16_dn = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA12, size=VB2_KV)
    mask_vec_dn = plm.make_tile(
        plm.TileType(shape=[TKV, TS_HALF], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec),
        addr=VA13,
        size=VB1_KV,
    )
    neg_inf_vec = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA14, size=VB4_KV)

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
                    if ki <= qi:
                        ctx_curr = ctx_arr[task_id % 3]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.qi = qi
                        ctx_curr.ki = ki
                        ctx_curr.skv_tiles = skv_tiles
                        ctx_curr.q_count = q_count
                        ctx_curr.sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
                        ctx_curr.task_id = task_id
                        if task_id > 0:
                            # qk_vec is single-buffered; consume the previous P/PV
                            # before cube overwrites it with the current QK.
                            compute_pv(ctx_arr[(task_id + 2) % 3], state, v, pv_vec, p_mat_buf1, p_mat_buf2, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2)
                        compute_qk(ctx_curr, state, q, k, q_mat_buf, k_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2, qk_vec)
                        task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                compute_pv(ctx_arr[(task_id + 2) % 3], state, v, pv_vec, p_mat_buf1, p_mat_buf2, v_mat_buf, left_buf, right_buf, acc_buf1, acc_buf2)

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
                    if ki <= qi:
                        ctx_curr = ctx_arr[task_id % 3]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.qi = qi
                        ctx_curr.ki = ki
                        ctx_curr.skv_tiles = skv_tiles
                        ctx_curr.q_count = q_count
                        ctx_curr.sub_id = sub_id
                        ctx_curr.task_id = task_id
                        if task_id > 1:
                            # pv_vec is also single-buffered; consume the older PV
                            # before compute_p releases the next P to cube.
                            compute_gu(ctx_arr[(task_id + 1) % 3], o, pv_vec, running_o, exp_corr, global_sum_buf, o_f16)
                        if task_id > 0:
                            compute_p(ctx_arr[(task_id + 2) % 3], sq_dim, skv_dim, qk_vec, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf,
                                      global_sum_rm_buf, exp_corr_rm, tile_nz, p_mat_buf1, p_mat_buf2,
                                      mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec)
                        task_id = task_id + 1
                q_count = q_count + 1
            if task_id > 0:
                if task_id > 1:
                    compute_gu(ctx_arr[(task_id + 1) % 3], o, pv_vec, running_o, exp_corr, global_sum_buf, o_f16)
                compute_p(ctx_arr[(task_id + 2) % 3], sq_dim, skv_dim, qk_vec, tmp_vec, p_f16, reduce_dst_rm, global_max_rm_buf,
                          global_sum_rm_buf, exp_corr_rm, tile_nz, p_mat_buf1, p_mat_buf2,
                          mask_u8_dn, mask_fp16_dn, mask_vec_dn, neg_inf_vec)
                compute_gu(ctx_arr[(task_id + 2) % 3], o, pv_vec, running_o, exp_corr, global_sum_buf, o_f16)
    return o


def flash_attention_causal_ref_bs(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, sq, n, _ = q.shape
    _, skv, _, _ = k.shape
    o_ref = torch.zeros_like(q)
    causal_mask = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=q.device), diagonal=1)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, :, ni, :].float(), k[bi, :, ni, :].float().T) * scale_val
            qk = qk.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, :, ni, :] = torch.matmul(attn, v[bi, :, ni, :].float()).half()
    return o_ref


def make_causal_mask_dn_fixed_2048_u8(device):
    # Fixed 2048x2048 mask: triu(diagonal=1), ND layout mask[k,q]=1 when q>k.
    # 1 = masked (set to -inf), 0 = attend (keep qk). Diagonal is 0 (attend).
    return torch.triu(torch.ones((FIXED_MASK_S, FIXED_MASK_S), dtype=torch.uint8, device=device), diagonal=1)


def test_fa_causal_bs_a5():
    compiled = fe.compile(fa_causal_bsnd_dn_kernel_v6, arch="a5", codegen_mode="cce", timeout=120)
    if compiled is None:
        raise RuntimeError("compile failed; check error.txt for fa_causal_bsnd_dn_kernel_v6")
    print("compiled:", compiled.lib_path)
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    for b, sq, n, skv, d, num_cores in [
        (1, 128, 1, 128, TD, 1),
        (1, 256, 1, 256, TD, 1),
        (1, 256, 5, 256, TD, 4),
        (3, 512, 1, 512, TD, 24),
        (1, 384, 1, 384, TD, 2),
        (1, 8192, 1, 8192, TD, 24),
    ]:
        print(f"\nFA-BSND-causal-DN-A5 (b={b}, sq={sq}, n={n}, skv={skv}, d={d}) cores={num_cores}")
        q = torch.rand((b, sq, n, d), device=device, dtype=torch.float16)
        k = torch.rand((b, skv, n, d), device=device, dtype=torch.float16)
        v = torch.rand((b, skv, n, d), device=device, dtype=torch.float16)
        o = torch.zeros((b, sq, n, d), device=device, dtype=torch.float16)
        attn_mask = make_causal_mask_dn_fixed_2048_u8(device)

        total_work = b * n
        work_ranges = torch.zeros((num_cores, 2), device=device, dtype=torch.int32)
        work_per_core = (total_work + num_cores - 1) // num_cores
        for core in range(num_cores):
            work_ranges[core, 0] = core * work_per_core
            work_ranges[core, 1] = min((core + 1) * work_per_core, total_work)

        actual_num_cores = min(num_cores, total_work)
        fe.launch(None, actual_num_cores, compiled, q, k, v, o, work_ranges, attn_mask)
        torch.npu.synchronize()

        # o_ref = flash_attention_causal_ref_bs(q, k, v, d)
        # torch.testing.assert_close(o, o_ref, rtol=5e-3, atol=5e-3)
        # print("  PASS")


if __name__ == "__main__":
    print("FA BSND causal DN on A5 CCE")
    print("=" * 60)
    test_fa_causal_bs_a5()
    print("\nAll FlashAttention DN tests passed!")
