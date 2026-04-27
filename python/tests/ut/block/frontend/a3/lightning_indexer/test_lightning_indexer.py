"""Lightning Indexer kernel using PyPTO IR manual (non-SSA) mode.

Lightning Indexer computes Top-k indices for sparse attention:
    Indices = Top-k(W ⊙ ReLU(Q @ K^T))

This is a fused operator combining:
  1. MatMul: Q @ K^T (Cube operation)
  2. ReLU + weight scaling (Vector operation)
  3. Top-k via sort32 (per 32-element block) + mrgsort (per 128-element tile) +
     sequential cross-tile merge (Vector operation)
     NPU outputs sorted_workspace [B, N, Sq, topk_padded*2] — top-topk [val,idx] pairs.
     topk_padded = ceil(topk/64)*64; caller reads only the first topk pairs.

Supports BNG axes: Batch (B), Query-heads (N), KV-groups (G, for GQA N >= G, N % G == 0).
Supports sparseMode:
  0 — no mask (default)
  3 — rightDownCausal: query position q can attend to key [0, q + (Sk-Sq)].
      Mask is computed on-chip per row in the sort32 loop using sort_idx (key indices).

Usage:
    python3 tests/ut/frontend/lightning_indexer/test_lightning_indexer.py
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm

# Tile dimensions
TS        = 128          # Sq / Sk tile size
TD        = 128          # head dimension
TOPK      = 128

# Cube tile byte sizes
Q_F16      = TS * TD * 2     # 32768  — [128,128] FP16
K_F16      = TD * TS * 2     # 32768  — [128,128] FP16
SCORES_F32 = TS * TS * 4     # 65536  — [128,128] FP32

# MAT addresses (512KB budget) — double buffer for K
MA0      = 0                  # q_mat    [TS, TD]  FP16  32KB
MA1      = Q_F16              # k_mat_0  [TD, TS]  FP16  32KB  (ping)
MA1_PONG = MA1 + K_F16        # k_mat_1  [TD, TS]  FP16  32KB  (pong)

# LEFT addresses (64KB budget) — double buffer
LA0 = 0                       # left_0  [TS, TD]  FP16  32KB
LA1 = Q_F16                   # left_1  [TS, TD]  FP16  32KB

# RIGHT addresses (64KB budget) — double buffer
RA0 = 0                       # right_0  [TD, TS]  FP16  32KB
RA1 = K_F16                   # right_1  [TD, TS]  FP16  32KB

# ACC
CA0 = 0                       # acc  [TS, TS]  FP32  64KB

# Double-buffer event ID tuples (accessed as free vars in helpers, constant-index only)
event_ids_01 = (0, 1)

# VEC tile byte sizes (plain int constants — BinOp exprs are not allowed in size= kwargs)
TS_HALF          = TS // 2           # 64     — rows per AIV
VB4_HALF         = TS_HALF * TS * 4  # 32768  — [TS_HALF, TS]  FP32
WEIGHTS_HALF_SZ  = TS_HALF * 1 * 4  # 256    — [TS_HALF, 1]   FP32
SORT_SZ          = 32                # elements per sort32 call (sort32 sorts exactly 32 elements)
SORT_BLOCKS_PER_TILE = TS // SORT_SZ # 4  — sort32 blocks per 128-element tile
SORT_SZ_BYTES    = SORT_SZ * 4       # 128    — [1, 32]         FP32
SORT_IDX_BYTES   = SORT_SZ * 4       # 128    — [1, 32]         UINT32
SORT_DST_COLS    = SORT_SZ * 2       # 64     — 32 elements * 2 FP32 per element
SORT_DST_BYTES   = SORT_DST_COLS * 4 # 256    — [1, 64]         FP32
MRGSORT_COLS     = TS * 2            # 256   — 128 elements * 2 FP32 per element (one tile)
MRGSORT_BYTES    = MRGSORT_COLS * 4  # 1024  — [1, 256]        FP32

# VEC base address layout (192KB budget on a3; merge buffers scale with topk_padded at runtime)
VA0  = 0                       # scores_vec    [TS_HALF, TS]   FP32  —  32768 B
VA1  = VA0 + VB4_HALF          # tmp_vec       [TS_HALF, TS]   FP32  —  32768 B
VA2  = VA1 + VB4_HALF          # weights_fp32  [TS_HALF, 1]    FP32  —    256 B
VA3  = VA2 + WEIGHTS_HALF_SZ   # sort_src      [1, 32]         FP32  —    128 B
VA4  = VA3 + SORT_SZ_BYTES     # sort_idx      [1, 32]        UINT32 —    128 B
VA5  = VA4 + SORT_IDX_BYTES    # sort_dst      [1, 64]         FP32  —    256 B
VA6  = VA5 + SORT_DST_BYTES    # mrgsort_src   [1, 256]        FP32  —   1024 B
VA7  = VA6 + MRGSORT_BYTES     # mrgsort_dst   [1, 256]        FP32  —   1024 B
VA8  = VA7 + MRGSORT_BYTES     # key_fp_tmp    [1, 32]         FP32  —    128 B (sparseMode=3)
VA9  = VA8 + SORT_SZ_BYTES     # sort_mask     [1, 32]         FP32  —    128 B (sparseMode=3)
# merge_lo/hi/out start at VA9+128; sizes depend on topk_padded — computed in make_lightning_indexer_kernel

# Cross-core sync event IDs: Cube alternates between AIV0 and AIV1 (backend auto-adds +16 on wait side)
SCORES_READY      = 8   # fires to AIV0; AIV1 auto-waits on 8+16=24

B   = pl.DynVar('B')
N   = pl.DynVar('N')
G   = pl.DynVar('G')
Sq  = pl.DynVar('Sq')
Sk     = pl.DynVar('Sk')
Sk2    = pl.DynVar('Sk2')
SkReal = pl.DynVar('SkReal')   # real (pre-padding) Sk — encodes s2 via tensor shape
SqReal = pl.DynVar('SqReal')   # real (pre-padding) Sq — encodes s1 via tensor shape
D      = pl.DynVar('D')

def alloc_cube_buffer():
    q_mat = plm.make_tile(
        plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat),
        addr=MA0, size=Q_F16)
    k_mat_type = plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    k_mat_0 = plm.make_tile(k_mat_type, addr=MA1,      size=K_F16)
    k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=K_F16)
    k_mat_buf = (k_mat_0, k_mat_1)
    left_0 = plm.make_tile(
        plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left),
        addr=LA0, size=Q_F16)
    left_1 = plm.make_tile(
        plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left),
        addr=LA1, size=Q_F16)
    left_buf = (left_0, left_1)
    right_0 = plm.make_tile(
        plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right),
        addr=RA0, size=K_F16)
    right_1 = plm.make_tile(
        plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right),
        addr=RA1, size=K_F16)
    right_buf = (right_0, right_1)
    acc = plm.make_tile(
        plm.TileType(shape=[TS, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc),
        addr=CA0, size=SCORES_F32)
    return q_mat, k_mat_buf, left_buf, right_buf, acc


def compute_li_k(b_id, g_id, sk_off, buf_idx,
                 key_tensor, scores_workspace,
                 q_mat, k_mat_buf, left_buf, right_buf, acc,
                 b_n_sq_off):
    """Load K[ski], move Q/K to LEFT/RIGHT, matmul, store ACC.

    buf_idx selects ping(0)/pong(1) for k_mat/left/right.
    Sync order mirrors test_fa.py compute_qk:
      MTE2←MTE1[buf]: K slot free? → load K → K ready
      MTE2←MTE1[buf]: K ready      → move Q/K to L0 → L0 ready
      MTE1←M   [buf]: wait M done  → (L0A/L0B writable)
      FIX ←M   [0]:  wait FIX done → (L0C writable) → matmul
      M   ←FIX [0]:  M done        → store ACC → FIX done
      FIX ←M   [0]:  FIX done      → next iter can overwrite L0C
    """
    # ---- Load K into k_mat_buf[buf_idx] ----
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1,
                       wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    plm.load(k_mat_buf[buf_idx], key_tensor, [b_id, g_id, sk_off, 0], layout="dn")
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                       wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[buf_idx])
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                       wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[buf_idx])

    # ---- Move Q/K → L0A/L0B ----
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                       wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[buf_idx])
    plm.move(left_buf[buf_idx],  q_mat)
    plm.move(right_buf[buf_idx], k_mat_buf[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1,
                       wait_pipe=pl.PipeType.M, event_id=event_ids_01[buf_idx])
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1,
                       wait_pipe=pl.PipeType.M, event_id=event_ids_01[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1,
                       wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])

    # ---- Matmul: wait previous FIX store done (L0C free), then compute ----
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX,
                       wait_pipe=pl.PipeType.M, event_id=0)
    plm.matmul(acc, left_buf[buf_idx], right_buf[buf_idx])

    # ---- Store ACC → GM ----
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                       wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M,
                       wait_pipe=pl.PipeType.FIX, event_id=0)
    # L0AB is now consumed by matmul — release it so MTE1 can reuse
    pl.system.sync_src(set_pipe=pl.PipeType.M,
                       wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[buf_idx])
    plm.store(scores_workspace, acc, b_n_sq_off)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX,
                       wait_pipe=pl.PipeType.M, event_id=0)

def make_lightning_indexer_kernel(sparse_mode: int, topk: int = TOPK):
    """Return a compiled kernel for the given sparse_mode (0 or 3) and topk.

    topk may be any positive integer; it is rounded up to the next multiple of 64
    internally (topk_padded) to satisfy the mrgsort block_len alignment requirement.
    The kernel writes topk_padded pairs to sorted_workspace; the caller reads only
    the first topk pairs.
    """
    kernel_name = f"lightning_indexer_sm{sparse_mode}_topk{topk}"
    # Round topk up to the nearest multiple of 64 (mrgsort block_len constraint).
    topk_padded = ((topk + 63) // 64) * 64

    # merge buffers scale with topk_padded
    _topk_pairs    = topk_padded * 2        # FP32 count per side (lo or hi)
    _topk_pairs_sz = _topk_pairs * 4        # bytes
    _topk_quad     = topk_padded * 4        # FP32 count for merge_src / merge_out
    _topk_quad_sz  = _topk_quad * 4         # bytes
    _va_mrg_lo  = VA9 + SORT_SZ_BYTES
    _va_mrg_hi  = _va_mrg_lo + _topk_pairs_sz
    _va_mrg_out = _va_mrg_hi + _topk_pairs_sz
    # Alias tile specs for TMULS shape matching (Python-level constants, compile-time resolved).
    # All three alias tiles are ALWAYS created unconditionally via plm.make_tile — never via
    # Python if/else assignment — so the DSL never treats them as IfStmt return-vars and
    # ConvertToSSA never renames them with an SSA suffix (_0, _1, …).
    #
    # topk_padded > MRGSORT_COLS (>128 pairs = >256 FP32):
    #   merge_lo_w/merge_hi_w = [1, MRGSORT_COLS]@_va_mrg_lo/_va_mrg_hi — write window
    #   mrgsort_dst_w = stub [1,1]@VA7, size=4 (dead else-branch)
    # topk_padded <= MRGSORT_COLS (i.e. topk_padded==64 or 128):
    #   merge_lo_w/merge_hi_w = stub [1,1] (dead then-branch)
    #   mrgsort_dst_w = [1, topk_padded*2]@VA7 — read window into mrgsort_dst
    #
    # Stub tiles use size=4/[1,1]; dedup key always differs from real tiles (different type_key).
    if _topk_pairs > MRGSORT_COLS:
        _alias_lo_cols, _alias_lo_sz = MRGSORT_COLS, MRGSORT_BYTES
    else:
        _alias_lo_cols, _alias_lo_sz = 1, 4          # stub
    _alias_lo_addr = _va_mrg_lo
    _alias_hi_cols, _alias_hi_sz = _alias_lo_cols, _alias_lo_sz
    _alias_hi_addr = _va_mrg_hi

    if _topk_pairs < MRGSORT_COLS:
        # topk_padded=64: mrgsort_dst_w=[1,topk_padded*2] view into mrgsort_dst
        _mrgsort_dst_w_cols, _mrgsort_dst_w_sz = _topk_pairs, _topk_pairs_sz
    else:
        # topk_padded>=128: mrgsort_dst is already the right size; stub avoids dedup
        _mrgsort_dst_w_cols, _mrgsort_dst_w_sz = 1, 4   # stub
    _mrgsort_dst_w_addr = VA7


    def alloc_vector_buffer():
        scores_vec = plm.make_tile(
            plm.TileType(shape=[TS_HALF, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA0, size=VB4_HALF)
        tmp_vec = plm.make_tile(
            plm.TileType(shape=[TS_HALF, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA1, size=VB4_HALF)
        weights_fp32 = plm.make_tile(
            plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2),
            addr=VA2, size=WEIGHTS_HALF_SZ)
        sort_src = plm.make_tile(
            plm.TileType(shape=[1, SORT_SZ], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA3, size=SORT_SZ_BYTES)
        sort_idx = plm.make_tile(
            plm.TileType(shape=[1, SORT_SZ], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec),
            addr=VA4, size=SORT_IDX_BYTES)
        sort_dst = plm.make_tile(
            plm.TileType(shape=[1, SORT_DST_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA5, size=SORT_DST_BYTES)
        mrgsort_src = plm.make_tile(
            plm.TileType(shape=[1, MRGSORT_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA6, size=MRGSORT_BYTES)
        mrgsort_dst = plm.make_tile(
            plm.TileType(shape=[1, MRGSORT_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA7, size=MRGSORT_BYTES)
        key_fp_tmp = plm.make_tile(
            plm.TileType(shape=[1, SORT_SZ], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA8, size=SORT_SZ_BYTES)
        sort_mask = plm.make_tile(
            plm.TileType(shape=[1, SORT_SZ], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=VA9, size=SORT_SZ_BYTES)
        # Cross-tile merge: buffers scale with topk (constants computed in outer scope)
        merge_lo  = plm.make_tile(
            plm.TileType(shape=[1, _topk_pairs], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=_va_mrg_lo, size=_topk_pairs_sz)
        merge_hi  = plm.make_tile(
            plm.TileType(shape=[1, _topk_pairs], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=_va_mrg_hi, size=_topk_pairs_sz)
        merge_src = plm.make_tile(
            plm.TileType(shape=[1, _topk_quad], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=_va_mrg_lo, size=_topk_quad_sz)   # overlaps lo+hi
        merge_out = plm.make_tile(
            plm.TileType(shape=[1, _topk_quad], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=_va_mrg_out, size=_topk_quad_sz)
        # merge_out_lo: first topk*2 FP32 of merge_out — copy to merge_lo as new accumulator
        merge_out_lo = plm.make_tile(
            plm.TileType(shape=[1, _topk_pairs], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
            addr=_va_mrg_out, size=_topk_pairs_sz)
        # Alias tiles: ALWAYS created unconditionally (never via Python if/else).
        # Stubs use size=4/[1,1] — their dedup key never matches any real tile.
        merge_lo_w = plm.make_tile(
            plm.TileType(shape=[1, _alias_lo_cols], dtype=pl.FP32,
                         target_memory=pl.MemorySpace.Vec),
            addr=_alias_lo_addr, size=_alias_lo_sz)
        merge_hi_w = plm.make_tile(
            plm.TileType(shape=[1, _alias_hi_cols], dtype=pl.FP32,
                         target_memory=pl.MemorySpace.Vec),
            addr=_alias_hi_addr, size=_alias_hi_sz)
        mrgsort_dst_w = plm.make_tile(
            plm.TileType(shape=[1, _mrgsort_dst_w_cols], dtype=pl.FP32,
                         target_memory=pl.MemorySpace.Vec),
            addr=_mrgsort_dst_w_addr, size=_mrgsort_dst_w_sz)
        return (scores_vec, tmp_vec, weights_fp32,
                sort_src, sort_idx, sort_dst, mrgsort_src, mrgsort_dst,
                key_fp_tmp, sort_mask,
                merge_lo, merge_hi, merge_src, merge_out, merge_out_lo,
                merge_lo_w, merge_hi_w, mrgsort_dst_w)

    @fe.kernel(name=kernel_name)
    def lightning_indexer_kernel(
        query: pl.Tensor[[B, N, Sq, D], pl.FP16],
        key_tensor: pl.Tensor[[B, G, Sk, D], pl.FP16],
        weights: pl.Tensor[[B, N, Sq, 1], pl.FP32],
        scores_workspace: pl.Tensor[[B, N, Sq, Sk], pl.FP32],
        idx_workspace: pl.Tensor[[B, N, Sq, Sk], pl.UINT32],
        key_idx_fp32: pl.Tensor[[1, Sk], pl.FP32],
        sorted_workspace: pl.Tensor[[B, N, Sq, Sk2], pl.FP32],
        causal_dims: pl.Tensor[[SqReal, SkReal], pl.FP32],
    ) -> pl.Tensor[[B, N, Sq, Sk2], pl.FP32]:
        """Lightning Indexer kernel.

        sparse_mode captured at compile time: 0=no mask, 3=rightDownCausal.
        causal_dims: dummy tensor whose shape [SqReal, SkReal] encodes the original
            (pre-padding) Sq and Sk. Used in sparseMode=3 to compute the correct causal
            offset (SkReal - SqReal) instead of the padded (Sk - Sq).
        topk captured at compile time: controls cross-tile merge output size.
        sorted_workspace shape: [B, N, Sq, topk*2] — [val,idx] pairs, NPU-merged top-k.
        For sparseMode=3: position q attends to key [0, q+(SkReal-SqReal)].
        """
        b_dim    = B
        n_dim    = N
        g_dim    = G
        sq_dim   = Sq
        sk_dim   = Sk
        sq_real  = SqReal   # original (pre-padding) Sq — used for causal offset
        sk_real  = SkReal   # original (pre-padding) Sk — used for causal offset
        sq_tiles = (sq_dim + (TS - 1)) // TS
        sk_tiles = (sk_dim + (TS - 1)) // TS
        num_cores = pl.block.index_cast(pl.block.get_block_num())
        core_id   = pl.block.index_cast(pl.block.get_block_idx())

        pl.system.bar_all()

        # =================== CUBE SECTION ===================
        with pl.section_cube():
            q_mat, k_mat_buf, left_buf, right_buf, acc = alloc_cube_buffer()

            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_src(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_src(set_pipe=pl.PipeType.FIX,  wait_pipe=pl.PipeType.M,    event_id=0)

            sqi_count = 0
            for b_id in pl.range(0, b_dim):
                for n_id in pl.range(0, n_dim):
                    g_id = n_id * g_dim // n_dim

                    for sqi in pl.range(core_id, sq_tiles, num_cores):
                        sq_off = sqi * TS

                        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                        plm.load(q_mat, query, [b_id, n_id, sq_off, 0])
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                        pl.system.sync_src(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=2)

                        for ski in pl.range(0, sk_tiles):
                            sk_off  = ski * TS
                            buf_idx = (sqi_count * sk_tiles + ski) % 2
                            compute_li_k(
                                b_id, g_id, sk_off, buf_idx,
                                key_tensor, scores_workspace,
                                q_mat, k_mat_buf, left_buf, right_buf, acc,
                                [b_id, n_id, sq_off, sk_off],
                            )

                        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=2)
                        sqi_count = sqi_count + 1

            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.FIX,  wait_pipe=pl.PipeType.M,    event_id=0)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=SCORES_READY)

        # =================== VECTOR SECTION ===================
        with pl.section_vector():
            (scores_vec, tmp_vec, weights_fp32,
             sort_src, sort_idx, sort_dst, mrgsort_src, mrgsort_dst,
             key_fp_tmp, sort_mask,
             merge_lo, merge_hi, merge_src, merge_out, merge_out_lo,
             merge_lo_w, merge_hi_w, mrgsort_dst_w) = alloc_vector_buffer()

            sub_id  = pl.block.index_cast(pl.block.get_subblock_idx())
            row_off = sub_id * TS_HALF

            # 等 Cube 全部写完 scores_workspace 后再开始（1次，精确 0→1→0）
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=SCORES_READY)

            for b_id in pl.range(0, b_dim):
                for n_id in pl.range(0, n_dim):
                    for sqi in pl.range(core_id, sq_tiles, num_cores):
                        sq_off = sqi * TS

                        plm.load(weights_fp32, weights, [b_id, n_id, sq_off + row_off, 0], layout="dn")
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                        for ski in pl.range(0, sk_tiles):
                            sk_off = ski * TS
                            plm.load(scores_vec, scores_workspace, [b_id, n_id, sq_off + row_off, sk_off])
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                            plm.relu(tmp_vec, scores_vec)
                            pl.system.bar_v()
                            plm.row_expand_mul(tmp_vec, tmp_vec, weights_fp32)
                            pl.system.bar_v()
                            pl.system.sync_src(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=0)
                            plm.store(scores_workspace, tmp_vec, [b_id, n_id, sq_off + row_off, sk_off])
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

                        # ---- sort32 + within-tile mrgsort, then cross-tile sequential merge ----
                        for row in pl.range(0, TS_HALF):
                            row_abs = sq_off + row_off + row

                            # --- ski=0 peeled: sort32×4 + mrgsort → accumulator (merge_lo) ---
                            for blk in pl.range(0, SORT_BLOCKS_PER_TILE):
                                blk_off = blk * SORT_SZ
                                plm.load(sort_src, scores_workspace, [b_id, n_id, row_abs, blk_off])
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                   wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                   wait_pipe=pl.PipeType.V, event_id=0)
                                plm.load(sort_idx, idx_workspace, [b_id, n_id, row_abs, blk_off])
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                   wait_pipe=pl.PipeType.V, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                   wait_pipe=pl.PipeType.V, event_id=1)
                                if sparse_mode == 3:
                                    thresh_p1 = row_abs + sk_real - sq_real + 1
                                    plm.load(key_fp_tmp, key_idx_fp32, [0, blk_off])
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                       wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                       wait_pipe=pl.PipeType.V, event_id=0)
                                    plm.expands(sort_mask, thresh_p1)
                                    pl.system.bar_v()
                                    plm.sub(sort_mask, sort_mask, key_fp_tmp)
                                    pl.system.bar_v()
                                    plm.relu(sort_mask, sort_mask)
                                    pl.system.bar_v()
                                    plm.mins(sort_mask, sort_mask, 1.0)
                                    pl.system.bar_v()
                                    plm.mul(sort_src, sort_mask, sort_src)
                                    pl.system.bar_v()
                                plm.sort32(sort_dst, sort_src, sort_idx)
                                pl.system.bar_v()
                                pl.system.sync_src(set_pipe=pl.PipeType.V,
                                                   wait_pipe=pl.PipeType.MTE3, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.V,
                                                   wait_pipe=pl.PipeType.MTE3, event_id=0)
                                # store sort_dst to scratch area of sorted_workspace (ski=0 region)
                                plm.store(sorted_workspace, sort_dst,
                                          [b_id, n_id, row_abs, blk * SORT_DST_COLS])
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3,
                                                   wait_pipe=pl.PipeType.MTE2, event_id=1)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3,
                                                   wait_pipe=pl.PipeType.MTE2, event_id=1)

                            # within-tile mrgsort for ski=0
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3,
                                               wait_pipe=pl.PipeType.MTE2, event_id=2)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3,
                                               wait_pipe=pl.PipeType.MTE2, event_id=2)
                            plm.load(mrgsort_src, sorted_workspace, [b_id, n_id, row_abs, 0])
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                               wait_pipe=pl.PipeType.V, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                               wait_pipe=pl.PipeType.V, event_id=0)
                            # mrgsort_dst = sorted 128 pairs of ski=0; top-topk_padded become accumulator
                            plm.mrgsort(mrgsort_dst, mrgsort_src, block_len=64)
                            pl.system.bar_v()
                            if _topk_pairs > MRGSORT_COLS:
                                plm.muls(merge_lo, merge_lo, 0.0)
                                pl.system.bar_v()
                                plm.muls(merge_lo_w, mrgsort_dst, 1.0)
                            elif _topk_pairs == MRGSORT_COLS:
                                plm.muls(merge_lo, mrgsort_dst, 1.0)
                            else:
                                plm.muls(merge_lo, mrgsort_dst_w, 1.0)
                            pl.system.bar_v()

                            # --- ski=1..sk_tiles-1: sort32×4 + mrgsort → merge_hi, merge with accum ---
                            for ski in pl.range(1, sk_tiles):
                                sk_off = ski * TS
                                for blk in pl.range(0, SORT_BLOCKS_PER_TILE):
                                    blk_off = sk_off + blk * SORT_SZ
                                    plm.load(sort_src, scores_workspace,
                                             [b_id, n_id, row_abs, blk_off])
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                       wait_pipe=pl.PipeType.V, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                       wait_pipe=pl.PipeType.V, event_id=0)
                                    plm.load(sort_idx, idx_workspace,
                                             [b_id, n_id, row_abs, blk_off])
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                       wait_pipe=pl.PipeType.V, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                       wait_pipe=pl.PipeType.V, event_id=1)
                                    if sparse_mode == 3:
                                        thresh_p1 = row_abs + sk_real - sq_real + 1
                                        plm.load(key_fp_tmp, key_idx_fp32, [0, blk_off])
                                        pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                           wait_pipe=pl.PipeType.V, event_id=0)
                                        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                           wait_pipe=pl.PipeType.V, event_id=0)
                                        plm.expands(sort_mask, thresh_p1)
                                        pl.system.bar_v()
                                        plm.sub(sort_mask, sort_mask, key_fp_tmp)
                                        pl.system.bar_v()
                                        plm.relu(sort_mask, sort_mask)
                                        pl.system.bar_v()
                                        plm.mins(sort_mask, sort_mask, 1.0)
                                        pl.system.bar_v()
                                        plm.mul(sort_src, sort_mask, sort_src)
                                        pl.system.bar_v()
                                    plm.sort32(sort_dst, sort_src, sort_idx)
                                    pl.system.bar_v()
                                    pl.system.sync_src(set_pipe=pl.PipeType.V,
                                                       wait_pipe=pl.PipeType.MTE3, event_id=0)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.V,
                                                       wait_pipe=pl.PipeType.MTE3, event_id=0)
                                    # scratch: reuse sorted_workspace[row, 0] area temporarily
                                    plm.store(sorted_workspace, sort_dst,
                                              [b_id, n_id, row_abs, blk * SORT_DST_COLS])
                                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3,
                                                       wait_pipe=pl.PipeType.MTE2, event_id=1)
                                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3,
                                                       wait_pipe=pl.PipeType.MTE2, event_id=1)

                                # within-tile mrgsort → merge_hi
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE3,
                                                   wait_pipe=pl.PipeType.MTE2, event_id=2)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3,
                                                   wait_pipe=pl.PipeType.MTE2, event_id=2)
                                plm.load(mrgsort_src, sorted_workspace, [b_id, n_id, row_abs, 0])
                                pl.system.sync_src(set_pipe=pl.PipeType.MTE2,
                                                   wait_pipe=pl.PipeType.V, event_id=0)
                                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2,
                                                   wait_pipe=pl.PipeType.V, event_id=0)
                                # within-tile mrgsort → mrgsort_dst (full 256 cols), then copy top to merge_hi
                                plm.mrgsort(mrgsort_dst, mrgsort_src, block_len=64)
                                pl.system.bar_v()
                                if _topk_pairs > MRGSORT_COLS:
                                    plm.muls(merge_hi, merge_hi, 0.0)
                                    pl.system.bar_v()
                                    plm.muls(merge_hi_w, mrgsort_dst, 1.0)
                                elif _topk_pairs == MRGSORT_COLS:
                                    plm.muls(merge_hi, mrgsort_dst, 1.0)
                                else:
                                    plm.muls(merge_hi, mrgsort_dst_w, 1.0)
                                pl.system.bar_v()

                                # cross-tile merge: block_len=topk_padded (must be multiple of 64)
                                plm.mrgsort(merge_out, merge_src, block_len=topk_padded)
                                pl.system.bar_v()

                                # update accumulator: merge_lo ← top-topk*2 of merge_out
                                plm.muls(merge_lo, merge_out_lo, 1.0)
                                pl.system.bar_v()

                            # write final accumulator → sorted_workspace[row, 0:TOPK*2]
                            pl.system.sync_src(set_pipe=pl.PipeType.V,
                                               wait_pipe=pl.PipeType.MTE3, event_id=1)
                            pl.system.sync_dst(set_pipe=pl.PipeType.V,
                                               wait_pipe=pl.PipeType.MTE3, event_id=1)
                            plm.store(sorted_workspace, merge_lo, [b_id, n_id, row_abs, 0])
                            pl.system.sync_src(set_pipe=pl.PipeType.MTE3,
                                               wait_pipe=pl.PipeType.MTE2, event_id=0)
                            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3,
                                               wait_pipe=pl.PipeType.MTE2, event_id=0)
        return sorted_workspace

    return lightning_indexer_kernel


# ================================================================
#  Reference + Tests
# ================================================================

def reference_lightning_indexer(weighted, topk=TOPK, sparse_mode=0):
    """Reference: sorted values and indices of top-k over last dim of weighted scores.

    weighted: [..., Sk] — arbitrary leading dims (B, N, Sq supported)
    sparse_mode: 0=no mask, 3=rightDownCausal
    Returns values [..., topk] and indices [..., topk] sorted descending over last dim.
    """
    if sparse_mode == 3:
        sq, sk = weighted.shape[-2], weighted.shape[-1]
        offset = sk - sq
        mask = torch.ones_like(weighted)
        for i in range(sq):
            valid = i + offset + 1
            if valid < sk:
                mask[..., i, valid:] = 0.0
        weighted = weighted * mask
    sorted_values, sorted_indices = torch.sort(weighted, dim=-1, descending=True, stable=True)
    return sorted_values[..., :topk], sorted_indices[..., :topk]

def _verify_outputs(tag, sorted_workspace, ref_weighted, ref_values, ref_indices, topk=TOPK):
    """Verify sorted_workspace against reference.

    sorted_workspace : [..., topk*2] FP32 — [value, idx] FP32 pairs, NPU top-k merged.
    ref_weighted : [..., Sk] CPU FP32 — W ⊙ ReLU(Q @ K^T) (after mask)
    ref_values   : [..., topk] CPU FP32 — top-k values (sorted descending)
    ref_indices  : [..., topk] CPU INT32 — top-k indices
    """
    raw_cpu = sorted_workspace.cpu()

    # Flatten all leading dims into a single batch dimension
    *leading, total_cols = raw_cpu.shape
    batch = 1
    for ld in leading:
        batch *= ld
    raw_cpu         = raw_cpu.reshape(batch, total_cols)
    ref_weighted_2d = ref_weighted.reshape(batch, -1)
    ref_values_2d   = ref_values.reshape(batch, -1)
    ref_indices_2d  = ref_indices.reshape(batch, -1)

    # sorted_workspace is already fully merged: [batch, MRGSORT_COLS=256], first topk*2 are valid
    valid_cols = topk * 2
    npu_vals = raw_cpu[:, :valid_cols:2].contiguous()                   # [batch, topk]
    raw_idx  = raw_cpu[:, 1:valid_cols:2].contiguous().view(torch.int32)  # [batch, topk]
    low  = raw_idx & 0xFFFF
    high = (raw_idx >> 16) & 0xFFFF
    indices_out    = (high << 16) | low                               # [batch, topk]
    merged_values  = npu_vals

    # Verify sort order
    assert (merged_values[:, :-1] >= merged_values[:, 1:]).all(), \
        f"[{tag}] output is not sorted descending"

    # Filter masked positions (value=0): applies to sparseMode=3 causal mask
    npu_valid = merged_values > 0
    ref_valid = ref_values_2d > 0

    # Diagnostics (row 0)
    print(f"  [{tag}] NPU indices shape : {indices_out.shape}  values shape : {merged_values.shape}")
    print(f"  [ref] ref indices shape  : {ref_indices_2d.shape}  values shape : {ref_values_2d.shape}")
    print(f"  [{tag}] NPU values[:32]  : {merged_values[0, :32].tolist()}")
    print(f"  [{tag}] NPU indices[:32] : {indices_out[0, :32].tolist()}")
    print(f"  [ref] ref values[:32]  : {ref_values_2d[0, :32].tolist()}")
    print(f"  [ref] ref indices[:32] : {ref_indices_2d[0, :32].tolist()}")

    # Set-based index match over valid positions
    match_count = 0
    total_count = 0
    valid_count_diff = 0
    for i in range(batch):
        npu_set = set(indices_out[i][npu_valid[i]].tolist())
        ref_set = set(ref_indices_2d[i][ref_valid[i]].tolist())
        valid_count_diff += abs(len(npu_set) - len(ref_set))
        if ref_set:
            match_count += len(npu_set & ref_set)
            total_count += len(ref_set)
    match = match_count / total_count if total_count > 0 else 1.0
    avg_count_diff = valid_count_diff / batch
    print(f"  index match rate={match:.4f} (valid-only set match), avg valid count diff={avg_count_diff:.1f}")
    assert avg_count_diff < topk * 0.01, \
        f"lightning_indexer {tag} failed: npu valid count differs from ref by {avg_count_diff:.1f} on average"
    assert match > 0.99, f"lightning_indexer {tag} failed: index match={match:.4f}"

    # Values check
    npu_ref_values = ref_weighted_2d.gather(1, indices_out.long())
    values_diff = 0.0
    for i in range(batch):
        npu_v = npu_ref_values[i][npu_valid[i]].sort(descending=True).values
        ref_v = ref_values_2d[i][ref_valid[i]].sort(descending=True).values
        n = min(npu_v.numel(), ref_v.numel())
        if n > 0:
            values_diff = max(values_diff, torch.abs(npu_v[:n] - ref_v[:n]).max().item())
    print(f"  values max diff (ref@npu_idx vs ref_topk)={values_diff:.6f}")
    assert values_diff < 0.5, f"lightning_indexer {tag} failed: values max diff {values_diff}"

    npu_sort_vs_ref = torch.abs(merged_values - npu_ref_values)
    sort_val_diff = npu_sort_vs_ref[npu_valid].max().item() if npu_valid.any() else 0.0
    print(f"  sort values diff (npu_sort vs ref@npu_idx)={sort_val_diff:.6f}")
    assert sort_val_diff < 0.5, f"lightning_indexer {tag} failed: sort values diff {sort_val_diff}"

    print(f"{tag} PASS")

def test_lightning_indexer():
    device = "npu:7"
    torch.npu.set_device(device)

    # (b, n, g, s1, s2, d, topk, num_cores, sparse_mode)
    # g divides n (GQA constraint: n % g == 0)
    for b, n, g, s1, s2, d, topk, num_cores, sparse_mode in [
        # === sparse_mode=0 ===
        (1, 1, 1, 128, 128,  TD, TOPK,  1, 0),   # 最小 baseline
        (1, 1, 1, 128, 256,  TD, 150,  1, 0),   # Sk > Sq (2 sk_tiles)
        (1, 1, 1, 256, 128,  TD, 100,  1, 0),   # Sq > Sk (2 sq_tiles, 1 sk_tile)
        (1, 1, 1, 256, 256,  TD, 256,   2, 0),   # topk=256, 2 tiles × 2 tiles
        (1, 1, 1, 512, 512,  TD, TOPK,  4, 0),   # 4 tiles × 4 tiles
        (1, 1, 1, 512, 1024, TD, 512,   4, 0),   # topk=512, 8 sk_tiles
        (1, 1, 1, 128, 1024, TD, 1000,  1, 0),   # 1 sq_tile, 8 sk_tiles (最多 cross-tile merge)
        (2, 4, 2, 128, 256,  TD, TOPK,  2, 0),   # BNG: B=2, N=4, G=2 (GQA 2:1)
        (1, 4, 1, 128, 256,  TD, 255,  2, 0),   # GQA 4:1
        (1, 8, 2, 256, 256,  TD, TOPK,  4, 0),   # GQA 4:1, 更大
        (1, 1, 1, 4096, 4096, TD, 2048, 16, 0),  # large
        (1, 1, 1, 128, 128,  TD, 64,   1, 0),    # 不同 topk
        # --- 尾块（非 128 整倍数）---
        (1, 1, 1,  96,  96,  TD, 50,   1, 0),   # Sq=Sk=96 (尾块)
        (1, 1, 1,  64, 200,  TD, 64,   1, 0),   # Sq=64, Sk=200 尾块
        (1, 1, 1, 160, 320,  TD, 200,  2, 0),   # Sq=160, Sk=320 尾块
        (1, 2, 1, 100, 150,  TD, 64,   1, 0),   # BN + 尾块

        # === sparse_mode=3 (rightDownCausal) ===
        (1, 1, 1, 128, 128,  TD, TOPK,  1, 3),   # Sk=Sq (row0 只有 1 个 valid)
        (1, 1, 1, 128, 256,  TD, 200,  1, 3),   # Sk>Sq (row0 有 129 个 valid)
        (1, 1, 1, 256, 256,  TD, TOPK,  2, 3),   # Sq=Sk=256
        (1, 1, 1, 256, 512,  TD, TOPK,  2, 3),   # Sk=2×Sq
        (1, 1, 1, 256, 512,  TD, 256,   2, 3),   # topk=256 + causal
        (1, 1, 1, 128, 1024, TD, TOPK,  1, 3),   # Sk=8×Sq (大量 mask)
        (2, 4, 2, 128, 256,  TD, 60,  2, 3),   # BNG + causal
        (1, 1, 1, 2048, 2048,  TD, 2048,  4, 3),   # large + causal
        # --- 尾块 + causal ---
        (1, 1, 1,  96, 192,  TD, 64,   1, 3),   # Sq=96, Sk=192 尾块 + causal
    ]:
        print(f"\nLightning-Indexer B={b} N={n} G={g} Sq={s1} Sk={s2} D={d} "
              f"topk={topk} cores={num_cores} sparseMode={sparse_mode}")

        kernel = make_lightning_indexer_kernel(sparse_mode, topk)
        compiled_cce = fe.compile(kernel, arch="a3", codegen_mode="cce")
        compiled_pto = fe.compile(kernel, arch="a3", codegen_mode="pto")

        torch.manual_seed(42 + sparse_mode * 4)
        query   = torch.randn([b, n, s1, d], device=device, dtype=torch.float16)
        key     = torch.randn([b, g, s2, d], device=device, dtype=torch.float16)
        weights = torch.rand([b, n, s1, 1], device=device, dtype=torch.float32).abs() + 0.1

        group_size   = n // g
        key_expanded = key.repeat_interleave(group_size, dim=1)
        ref_weighted = (
            torch.relu(torch.matmul(query.float(), key_expanded.float().transpose(-2, -1)))
            * weights
        )
        ref_values, ref_indices = reference_lightning_indexer(ref_weighted, topk, sparse_mode)
        ref_weighted = ref_weighted.cpu()
        ref_values   = ref_values.cpu()
        ref_indices  = ref_indices.cpu().int()

        # Pad Sq and Sk to multiples of TS so the kernel always sees full tiles.
        # Padded regions are zero: ReLU(0)=0, so padding entries never enter top-k.
        sq_pad = ((s1 + TS - 1) // TS) * TS
        sk_pad = ((s2 + TS - 1) // TS) * TS

        def pad_rows(t, padded):
            """Zero-pad axis -2 (row dimension) of t."""
            p = padded - t.shape[-2]
            return torch.nn.functional.pad(t, (0, 0, 0, p)) if p > 0 else t

        query_p   = pad_rows(query,   sq_pad)   # [b, n, sq_pad, d]
        key_p     = pad_rows(key,     sk_pad)   # [b, g, sk_pad, d]
        weights_p = pad_rows(weights, sq_pad)   # [b, n, sq_pad, 1]

        idx_workspace = (
            torch.arange(sk_pad, device=device, dtype=torch.int32)
            .view(1, 1, 1, sk_pad).expand(b, n, sq_pad, -1).contiguous()
        )
        key_idx_fp32 = torch.arange(sk_pad, device=device, dtype=torch.float32).unsqueeze(0)

        # causal_dims: dummy [s1, s2] FP32 tensor — shape encodes real Sq/Sk for causal offset.
        # Data is never read by the kernel; only the shape (SqReal=s1, SkReal=s2) is used.
        causal_dims = torch.zeros([s1, s2], device=device, dtype=torch.float32)

        topk_padded = ((topk + 63) // 64) * 64
        sorted_cols = max(topk_padded * 2, MRGSORT_COLS)

        def make_workspaces():
            return (
                torch.zeros([b, n, sq_pad, sk_pad], device=device, dtype=torch.float32),
                torch.zeros([b, n, sq_pad, sorted_cols], device=device, dtype=torch.float32),
            )

        # --- CCE ---
        scores_workspace, sorted_workspace = make_workspaces()
        fe.launch(None, num_cores, compiled_cce,
                  query_p, key_p, weights_p,
                  scores_workspace, idx_workspace, key_idx_fp32, sorted_workspace, causal_dims)
        torch.npu.synchronize()
        _verify_outputs("CCE", sorted_workspace[:, :, :s1, :], ref_weighted, ref_values, ref_indices, topk)

        # --- PTO ---
        scores_workspace, sorted_workspace = make_workspaces()
        fe.launch(None, num_cores, compiled_pto,
                  query_p, key_p, weights_p,
                  scores_workspace, idx_workspace, key_idx_fp32, sorted_workspace, causal_dims)
        torch.npu.synchronize()
        _verify_outputs("PTO", sorted_workspace[:, :, :s1, :], ref_weighted, ref_values, ref_indices, topk)

if __name__ == "__main__":
    print("Lightning Indexer: sort32 + mrgsort + cross-tile merge, BNG + sparseMode=0/3 support")
    print("=" * 60)
    test_lightning_indexer()
    print("\nAll lightning_indexer tests passed!")