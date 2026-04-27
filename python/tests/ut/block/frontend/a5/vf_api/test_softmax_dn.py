"""Softmax DN VF API test — ProcessVec1DnNoUpdateVF (fp16 branch, full version).

Full version with Cast + DeInterleave + StoreAlign(DATA_BLOCK_COPY).
Constants: m=64, ubN=128, dScale=1.0

UB layout (DN mode / transposed):
  input_x_local_UB: fp32 [ubN, m] = [128, 64]
  x_exp:            fp16 output (softmax result, NZ layout)
  new_global_max:   fp32 [64] row max
  new_global_sum:   fp32 [64] row sum
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm
import pypto_block.language.op.vf_api as vf

# ================================================================
# Constants
# ================================================================
m = 64              # elements per register (row stride)
ubN = 128           # columns
dScale = 1.0        # scale factor
minValue = -1e9
blockStride = ubN >> 1  # = 64 (不加1，先不管bank冲突)
repeatStride = 1

# UB address layout
INPUT_SIZE = ubN * m * 4        # fp32 [128, 64] = 32KB
XEXP_SIZE = ubN * m * 2         # fp16 [128, 64] = 16KB
MAX_SIZE = m * 4                # fp32 [64] = 256B
SUM_SIZE = m * 4                # fp32 [64] = 256B

VA_INPUT = 0
VA_XEXP = VA_INPUT + INPUT_SIZE                 # 0x8000
VA_MAX = VA_XEXP + XEXP_SIZE                    # 0xC000
VA_SUM = VA_MAX + MAX_SIZE                      # 0xC100

M_DYN = pl.DynVar('M')
N_DYN = pl.DynVar('N')
M_OUT = pl.DynVar('M_OUT')
M_EXP = pl.DynVar('M_EXP')
N_EXP = pl.DynVar('N_EXP')


@pl.inline
def process_vec1_dn_no_update_vf(input_tile, x_exp_tile, max_tile, sum_tile):
    """Softmax DN VF kernel — full version with Cast + DeInterleave + Store."""
    with vf.vf_scope(name="softmax_dn"):
        # ---- fp32 registers ----
        src0 = vf.RegTensor(dtype=pl.FP32)
        src1 = vf.RegTensor(dtype=pl.FP32)
        src2 = vf.RegTensor(dtype=pl.FP32)
        src3 = vf.RegTensor(dtype=pl.FP32)
        max0 = vf.RegTensor(dtype=pl.FP32)
        max1 = vf.RegTensor(dtype=pl.FP32)
        max2 = vf.RegTensor(dtype=pl.FP32)
        max3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_2 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_f32_3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_2 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_exp_3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_1 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_2 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum_3 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum0 = vf.RegTensor(dtype=pl.FP32)
        vreg_x_sum1 = vf.RegTensor(dtype=pl.FP32)

        # ---- fp16 registers (for Cast output) ----
        vreg_x_exp_even_f16 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_odd_f16 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_even_f16_1 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_odd_f16_1 = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_pack = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_packa = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_1_pack = vf.RegTensor(dtype=pl.FP16)
        vreg_x_exp_f16_1_packa = vf.RegTensor(dtype=pl.FP16)

        # ---- Masks ----
        preg_108 = vf.CreateMask(pattern="ALL", dtype=pl.FP32)   # b32, for fp32 ops
        preg_134 = vf.CreateMask(pattern="ALL", dtype=pl.FP32)   # b32, for fp32 exp/add
        preg_135 = vf.CreateMask(pattern="ALL", dtype=pl.FP32)   # b32, for Cast
        preg_136 = vf.UpdateMask(128, dtype=pl.FP16)  # for DATA_BLOCK_COPY store (fp16 data)

        # ---- Pointer offsets ----
        src_ub0 = input_tile
        src_ub1 = input_tile + m
        src_ub2 = input_tile + m * 2
        src_ub3 = input_tile + m * 3
        x_exp_1 = x_exp_tile + ubN * 4   # second half of x_exp output

        # ============================================================
        # Phase 1: ReduceMax (4-way parallel)
        # ============================================================
        vf.Duplicate(max0, minValue)
        vf.Duplicate(max1, minValue)
        vf.Duplicate(max2, minValue)
        vf.Duplicate(max3, minValue)

        for iter_m in pl.range(0, ubN // 4):
            vf.LoadAlign(src0, src_ub0, iter_m * m * 4)
            vf.LoadAlign(src1, src_ub1, iter_m * m * 4)
            vf.LoadAlign(src2, src_ub2, iter_m * m * 4)
            vf.LoadAlign(src3, src_ub3, iter_m * m * 4)
            vf.Max(max0, max0, src0, preg_108)
            vf.Max(max1, max1, src1, preg_108)
            vf.Max(max2, max2, src2, preg_108)
            vf.Max(max3, max3, src3, preg_108)

        vf.Max(max0, max0, max2, preg_108)
        vf.Max(max1, max1, max3, preg_108)
        vf.Max(max0, max0, max1, preg_108)
        vf.Muls(max0, max0, dScale, preg_108)

        vf.StoreAlign(max_tile, max0, preg_108)

        # ============================================================
        # Phase 2: ExpSub + Cast + DeInterleave + Store + Sum
        # ============================================================
        vf.Duplicate(vreg_x_sum_0, 0.0, preg_134)
        vf.Duplicate(vreg_x_sum_1, 0.0, preg_134)
        vf.Duplicate(vreg_x_sum_2, 0.0, preg_134)
        vf.Duplicate(vreg_x_sum_3, 0.0, preg_134)

        for i0 in pl.range(0, ubN // 4):
            # Load from 4 quadrants
            vf.LoadAlign(vreg_x_f32_0, input_tile, i0 * m)
            vf.LoadAlign(vreg_x_f32_1, input_tile, ubN * m // 4 + i0 * m)
            vf.LoadAlign(vreg_x_f32_2, input_tile, ubN * m // 2 + i0 * m)
            vf.LoadAlign(vreg_x_f32_3, input_tile, ubN * m // 2 + ubN * m // 4 + i0 * m)

            # Scale
            vf.Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108)
            vf.Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108)
            vf.Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108)
            vf.Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108)

            # FusedExpSub: exp(x * dScale - max)
            vf.FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134)
            vf.FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134)
            vf.FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134)
            vf.FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134)

            # Cast fp32 → fp16 (castTraitZero: layout=ZERO, round=CAST_ROUND)
            vf.Cast(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135, layout="ZERO")
            vf.Cast(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135, layout="ZERO")
            vf.DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa,
                            vreg_x_exp_even_f16, vreg_x_exp_odd_f16)

            vf.Cast(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135, layout="ZERO")
            vf.Cast(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135, layout="ZERO")
            vf.DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa,
                            vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1)

            # Store fp16 result (DATA_BLOCK_COPY + POST_UPDATE)
            vf.StoreAlign(x_exp_tile, vreg_x_exp_f16_pack, preg_136,
                          block_stride=blockStride, repeat_stride=repeatStride,
                          data_copy_mode="DATA_BLOCK_COPY", post_update=True)
            vf.Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134)
            vf.Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134)

            vf.StoreAlign(x_exp_1, vreg_x_exp_f16_1_pack, preg_136,
                          block_stride=blockStride, repeat_stride=repeatStride,
                          data_copy_mode="DATA_BLOCK_COPY", post_update=True)
            vf.Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134)
            vf.Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134)

        # ============================================================
        # Phase 3: Sum merge + store
        # ============================================================
        vf.Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134)
        vf.Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134)
        vf.Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134)
        vf.StoreAlign(sum_tile, vreg_x_sum0, preg_134)


@fe.kernel
def softmax_dn_vf_kernel(
    x: pl.Tensor[[M_DYN, N_DYN], pl.FP32],
    out_xexp: pl.Tensor[[M_EXP, N_EXP], pl.FP16],
    out_max: pl.Tensor[[M_OUT, N_DYN], pl.FP32],
    out_sum: pl.Tensor[[M_OUT, N_DYN], pl.FP32],
) -> pl.Tensor[[M_OUT, N_DYN], pl.FP32]:
    # UB tiles
    input_tile_type = plm.TileType(
        shape=[ubN, m], dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    xexp_tile_type = plm.TileType(
        shape=[ubN, m], dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    max_tile_type = plm.TileType(
        shape=[1, m], dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    sum_tile_type = plm.TileType(
        shape=[1, m], dtype=pl.FP32,
        target_memory=pl.MemorySpace.Vec,
    )

    input_tile = plm.make_tile(input_tile_type, addr=VA_INPUT, size=INPUT_SIZE)
    x_exp_tile = plm.make_tile(xexp_tile_type, addr=VA_XEXP, size=XEXP_SIZE)
    max_tile = plm.make_tile(max_tile_type, addr=VA_MAX, size=MAX_SIZE)
    sum_tile = plm.make_tile(sum_tile_type, addr=VA_SUM, size=SUM_SIZE)

    with pl.section_vector():
        plm.load(input_tile, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        process_vec1_dn_no_update_vf(input_tile, x_exp_tile, max_tile, sum_tile)

        # Sync and store results
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(out_xexp, x_exp_tile, [0, 0])
        plm.store(out_max, max_tile, [0, 0])
        plm.store(out_sum, sum_tile, [0, 0])

    return out_sum


def compute_golden(x_dn, device):
    """Compute golden results for softmax DN kernel.

    Returns:
        golden_max: fp32 [64]
        golden_sum: fp32 [64]
        golden_xexp_store: fp16 [128*64] — what plm.store outputs from x_exp_tile
    """
    golden_max = x_dn.max(dim=0).values * dScale
    x_scaled = x_dn * dScale
    x_exp_fp32 = torch.exp(x_scaled - golden_max.unsqueeze(0))
    golden_sum = x_exp_fp32.sum(dim=0)

    # Simulate vsstb layout in UB
    x_exp_fp16 = x_exp_fp32.to(torch.float16)
    golden_xexp_store = simulate_vsstb_layout(x_exp_fp16, device)

    return golden_max, golden_sum, golden_xexp_store


def simulate_vsstb_layout(x_exp_fp16, device):
    """Simulate vsstb writes into UB, then return what plm.store would output.

    UB x_exp region: XEXP_SIZE = 16640 bytes = 8320 fp16 elements
    plm.store outputs: x_exp_tile shape [128, 64] = 8192 fp16 elements (first 16384 bytes)

    vsstb writes 8 blocks of 16 fp16 per call:
      block[i] at ptr + i * blockStride * 16 fp16
      POST_UPDATE: ptr += repeatStride * 16 fp16

    Two streams:
      x_exp:   ptr0 starts at offset 0
      x_exp_1: ptr1 starts at offset ubN*4 = 512 fp16 elements
    """
    BLOCK_SIZE = 16       # fp16 elements per block
    NUM_BLOCKS = 8
    BS_ELEM = blockStride * BLOCK_SIZE   # 65 * 16 = 1040 fp16
    RS_ELEM = repeatStride * BLOCK_SIZE  # 1 * 16 = 16 fp16
    XEXP1_OFFSET = ubN * 4              # 512 fp16

    # UB buffer (in fp16 elements) — use full XEXP_SIZE
    buf_size = XEXP_SIZE // 2  # 8320 fp16 elements
    ub_buf = torch.zeros(buf_size, dtype=torch.float16, device=device)

    x_flat = x_exp_fp16.reshape(-1)  # [8192]

    ptr0 = 0             # x_exp stream
    ptr1 = XEXP1_OFFSET  # x_exp_1 stream

    num_iters = ubN // 4  # 32
    for i0 in range(num_iters):
        # 4 quadrants
        quad0 = x_flat[i0 * m                                : i0 * m + m]
        quad1 = x_flat[ubN * m // 4 + i0 * m                : ubN * m // 4 + i0 * m + m]
        quad2 = x_flat[ubN * m // 2 + i0 * m                : ubN * m // 2 + i0 * m + m]
        quad3 = x_flat[ubN * m // 2 + ubN * m // 4 + i0 * m : ubN * m // 2 + ubN * m // 4 + i0 * m + m]

        # Cast(PART_EVEN) + DeInterleave → pack = cat(quad_even, quad_odd)
        pack0 = torch.cat([quad0, quad2])  # [128]
        pack1 = torch.cat([quad1, quad3])  # [128]

        # vsstb pack0 → x_exp stream
        for blk in range(NUM_BLOCKS):
            src_s = blk * BLOCK_SIZE
            dst_s = ptr0 + blk * BS_ELEM
            if dst_s + BLOCK_SIZE <= buf_size:
                ub_buf[dst_s : dst_s + BLOCK_SIZE] = pack0[src_s : src_s + BLOCK_SIZE]
        ptr0 += RS_ELEM

        # vsstb pack1 → x_exp_1 stream
        for blk in range(NUM_BLOCKS):
            src_s = blk * BLOCK_SIZE
            dst_s = ptr1 + blk * BS_ELEM
            if dst_s + BLOCK_SIZE <= buf_size:
                ub_buf[dst_s : dst_s + BLOCK_SIZE] = pack1[src_s : src_s + BLOCK_SIZE]
        ptr1 += RS_ELEM

    # plm.store outputs 128*64 = 8192 fp16 elements from UB
    store_size = ubN * m  # 8192
    return ub_buf[:store_size]


def simulate_vsstb_layout(x_exp_fp16, device):
    """Simulate Cast + DeInterleave + vsstb(DATA_BLOCK_COPY, POST_UPDATE).

    The VF kernel loop (32 iterations) does:
      1. Load 4 quadrants → Muls → FusedExpSub → 4 fp32 exp regs (each 64 elements)
      2. Cast fp32→fp16 (PART_EVEN): exp_0→even_f16[64], exp_2→odd_f16[64]
      3. DeInterleave(even_f16, odd_f16) → pack[128], packa[128]
         DeInterleave takes odd-indexed elements from src0+src1 into dst0,
         even-indexed elements into dst1:
           src0=[e0,e1,e2,...], src1=[o0,o1,o2,...]
           dst0(pack) =[e0,e2,e4,...,o0,o2,o4,...]  (stored)
           dst1(packa)=[e1,e3,e5,...,o1,o3,o5,...]  (discarded)
      4. vsstb(pack, ptr, (65<<16)|1, mask, POST_UPDATE)
         128 fp16 = 8 blocks x 16 elements
         block[i] at ptr + i * blockStride * 16 fp16
         POST_UPDATE: ptr += repeatStride * 16 fp16

    Args:
        x_exp_fp16: fp16 tensor [ubN, m] = [128, 64]
        device: torch device

    Returns:
        flat fp16 tensor [8192] matching UB x_exp layout
    """
    BLOCK_SIZE = 16
    NUM_BLOCKS = 8
    BS_ELEM = blockStride * BLOCK_SIZE   # 65 * 16 = 1040 fp16 elements
    RS_ELEM = repeatStride * BLOCK_SIZE  # 1 * 16 = 16 fp16 elements
    XEXP1_OFFSET = ubN * 4               # 512 fp16 elements

    buf_size = XEXP_SIZE // 2  # 8192 fp16 elements
    out_buf = torch.zeros(buf_size, dtype=torch.float16, device=device)

    x_flat = x_exp_fp16.reshape(-1)  # [8192]

    ptr0 = 0             # x_exp stream
    ptr1 = XEXP1_OFFSET  # x_exp_1 stream

    num_iters = ubN // 4  # 32
    for i0 in range(num_iters):
        # 4 quadrants (each 64 fp16 elements)
        quad0 = x_flat[i0 * m                                : i0 * m + m]
        quad1 = x_flat[ubN * m // 4 + i0 * m                : ubN * m // 4 + i0 * m + m]
        quad2 = x_flat[ubN * m // 2 + i0 * m                : ubN * m // 2 + i0 * m + m]
        quad3 = x_flat[ubN * m // 2 + ubN * m // 4 + i0 * m : ubN * m // 2 + ubN * m // 4 + i0 * m + m]

        # Cast PART_EVEN: fp32[64] → fp16, values placed at even indices of 128-element reg
        # even_f16 = [q0_0, 0, q0_1, 0, ..., q0_63, 0]  (128 fp16, odd slots = 0)
        # odd_f16  = [q2_0, 0, q2_1, 0, ..., q2_63, 0]
        #
        # DeInterleave(even_f16, odd_f16):
        #   dst0(pack) = even_f16[0::2] concat odd_f16[0::2] = [q0_0..q0_63, q2_0..q2_63]
        #   dst1(packa) = even_f16[1::2] concat odd_f16[1::2] = [0..0, 0..0] (zeros)
        # So pack = simple concatenation of the valid fp16 values
        pack0 = torch.cat([quad0, quad2])  # [128]
        pack1 = torch.cat([quad1, quad3])  # [128]

        # vsstb pack0 → x_exp stream
        for blk in range(NUM_BLOCKS):
            src_s = blk * BLOCK_SIZE
            dst_s = ptr0 + blk * BS_ELEM
            if dst_s + BLOCK_SIZE <= buf_size:
                out_buf[dst_s : dst_s + BLOCK_SIZE] = pack0[src_s : src_s + BLOCK_SIZE]
        ptr0 += RS_ELEM

        # vsstb pack1 → x_exp_1 stream
        for blk in range(NUM_BLOCKS):
            src_s = blk * BLOCK_SIZE
            dst_s = ptr1 + blk * BS_ELEM
            if dst_s + BLOCK_SIZE <= buf_size:
                out_buf[dst_s : dst_s + BLOCK_SIZE] = pack1[src_s : src_s + BLOCK_SIZE]
        ptr1 += RS_ELEM

    return out_buf


@fe.jit()
def test_softmax_dn():
    compiled_lib = fe.compile(softmax_dn_vf_kernel, arch="a5", codegen_mode="cce")
    print("Compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x_dn = torch.rand([ubN, m], device=device, dtype=torch.float32)
    out_xexp = torch.empty([ubN, m], device=device, dtype=torch.float16)
    out_max = torch.empty([1, m], device=device, dtype=torch.float32)
    out_sum = torch.empty([1, m], device=device, dtype=torch.float32)

    fe.launch(None, 1, compiled_lib, x_dn, out_xexp, out_max, out_sum)
    torch.npu.synchronize()

    golden_max, golden_sum, golden_xexp = compute_golden(x_dn, device)

    npu_max = out_max.squeeze(0)
    npu_sum = out_sum.squeeze(0)
    npu_xexp = out_xexp.reshape(-1)

    # ---- Max check ----
    print("***********npu max***********")
    print(npu_max[:8])
    print("***********golden max***********")
    print(golden_max[:8])
    torch.testing.assert_close(npu_max, golden_max, rtol=1e-3, atol=1e-3)
    print("Max check PASSED")

    # ---- Sum check ----
    print("***********npu sum***********")
    print(npu_sum[:8])
    print("***********golden sum***********")
    print(golden_sum[:8])
    torch.testing.assert_close(npu_sum, golden_sum, rtol=1e-2, atol=1e-2)
    print("Sum check PASSED")

    # ---- x_exp layout check ----
    print("***********npu x_exp (first 32)***********")
    print(npu_xexp[:32])
    print("***********golden x_exp (first 32)***********")
    print(golden_xexp[:32])
    torch.testing.assert_close(npu_xexp, golden_xexp, rtol=1e-2, atol=1e-2)
    print("x_exp layout check PASSED")

    print("Softmax DN VF API (full version) test PASSED!")


if __name__ == "__main__":
    test_softmax_dn()
    print("\nAll tests passed!")
