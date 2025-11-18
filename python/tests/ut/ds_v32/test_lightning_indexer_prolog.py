from dataclasses import dataclass, field
from typing import List, Set
import logging
import pytest
import pto

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

NUM_NEG1 = -1
NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4 = 4
NUM_16 = 16
NUM_28 = 28
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_448 = 448
NUM_1024 = 1024
NUM_1536 = 1536
NUM_2048 = 2048
NUM_7168 = 7168


@dataclass
class LightningIndexerPrologTileConfig:
    c1_tile: List[List[int]]
    v1_tile: List[int]
    c2_tile: List[List[int]]
    v2_tile: List[int]
    rope_2d: List[int]
    rope_3d: List[int]
    rope_4d: List[int]


@dataclass
class LightningIndexerPrologParams:
    b: int
    s1: int
    dim: int
    q_lora_rank: int
    head_dim: int
    head_num: int
    rope_head_dim: int
    block_size: int
    block_num: int
    n_kv: int
    s2: int
    tile_bs: int = -1


@dataclass
class LightningIndexerPrologArgs:
    x: pto.Tensor
    qr: pto.Tensor
    q_w: pto.Tensor
    k_w: pto.Tensor
    proj_w: pto.Tensor
    ln_w: pto.Tensor
    ln_b: pto.Tensor
    cos: pto.Tensor
    sin: pto.Tensor
    k_cache: pto.Tensor
    k_cache_index: pto.Tensor
    block_table: pto.Tensor
    query: pto.Tensor
    weight: pto.Tensor
    k_cache_out: pto.Tensor
    tile_config: LightningIndexerPrologTileConfig
    unroll_list: List[int]
    params: LightningIndexerPrologParams


@dataclass
class LightningIndexerPrologBuildConfig:
    b: int = NUM_28
    s1: int = NUM_1
    dim: int = NUM_7168
    rope_head_dim: int = NUM_64
    n_kv: int = NUM_1
    head_dim: int = NUM_128
    head_num: int = NUM_64
    q_lora_rank: int = NUM_1536
    s2_tile: int = NUM_2048
    block_size: int = NUM_128
    block_num: int = NUM_448
    tile_bs: int = NUM_NEG1
    c1_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_16, NUM_16],
            [NUM_256, NUM_256],
            [NUM_128, NUM_128],
        ]
    )
    v1_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_256, NUM_128, NUM_128]
    )
    c2_tile: List[List[int]] = field(
        default_factory=lambda: [
            [NUM_16, NUM_16],
            [NUM_256, NUM_256],
            [NUM_128, NUM_128],
        ]
    )
    v2_tile: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_128, NUM_128, NUM_128]
    )
    rope_2d: List[int] = field(default_factory=lambda: [NUM_128, NUM_256])
    rope_3d_vals: List[int] = field(default_factory=lambda: [NUM_32, NUM_128, NUM_128])
    rope_4d: List[int] = field(
        default_factory=lambda: [NUM_1, NUM_64, NUM_128, NUM_128]
    )


def layer_norm(
    x: pto.Tensor, weight: pto.Tensor, bias: pto.Tensor, dim: int
) -> pto.Tensor:
    assert dim == (len(x.shape) - 1) or dim == -1
    assert x.dtype == pto.DataType.DT_FP32
    eps = 1e-6
    actual_dim = dim + len(x.shape) if dim < 0 else dim
    x_scaled = x / (x.shape[actual_dim])
    mean = pto.sum(x_scaled, -1)
    diff = x - mean
    squared_diff = diff * diff
    square_diff_scaled = squared_diff / (x.shape[actual_dim])
    var = pto.sum(square_diff_scaled, -1)
    std_var = pto.sqrt(var + eps)
    res32 = diff / std_var
    weight32 = pto.cast(weight, pto.DataType.DT_FP32)
    bias32 = pto.cast(bias, pto.DataType.DT_FP32)
    return res32 * weight32 + bias32


def rotate_half(input_tensor: pto.Tensor) -> pto.Tensor:
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % NUM_2 == 0
    shape[shape_size - 1] //= NUM_2
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pto.view(input_tensor, shape, offset1)
    x2 = pto.view(input_tensor, shape, offset2)
    return pto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rotate_half_valid_shape(input_tensor: pto.Tensor) -> pto.Tensor:
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % NUM_2 == 0
    shape[shape_size - 1] //= NUM_2
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    valid_shape = input_tensor.dynamic_valid_shape
    valid_shape[shape_size - 1] //= NUM_2
    x1 = pto.view(input_tensor, shape, offset1, valid_shape=valid_shape)
    x2 = pto.view(input_tensor, shape, offset2, valid_shape=valid_shape)
    return pto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(
    x: pto.Tensor,
    cos: pto.Tensor,
    sin: pto.Tensor,
    tile_config: LightningIndexerPrologTileConfig,
) -> pto.Tensor:
    assert (
        len(x.shape) == SHAPE_DIM_3
        and len(cos.shape) == SHAPE_DIM_2
        and len(sin.shape) == SHAPE_DIM_2
    )
    pto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128)
    cast_x = pto.cast(x, pto.DataType.DT_FP32)
    if x.dtype == pto.DataType.DT_FP32:
        cast_x = cast_x + 0.0
    cast_cos = pto.cast(cos, pto.DataType.DT_FP32)
    cast_sin = pto.cast(sin, pto.DataType.DT_FP32)
    cast_cos[:] = pto.reshape(cast_cos, [x.shape[NUM_0], 1, x.shape[NUM_2]])
    cast_sin[:] = pto.reshape(cast_sin, [x.shape[NUM_0], 1, x.shape[NUM_2]])
    x_valid_shape = x.dynamic_valid_shape
    x_view = pto.reshape(
        cast_x,
        [x.shape[NUM_0], x.shape[NUM_1], x.shape[NUM_2] // NUM_2, NUM_2],
        valid_shape=[
            x_valid_shape[NUM_0],
            x_valid_shape[NUM_1],
            x_valid_shape[NUM_2] // NUM_2,
            NUM_2,
        ],
    )
    pto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128, NUM_128)
    x_trans = pto.transpose(x_view, NUM_2, NUM_3)
    x_re_second = pto.reshape(x_trans, x.shape, valid_shape=x_valid_shape)
    pto.set_vec_tile_shapes(NUM_1, NUM_32, NUM_128, NUM_128)
    x_embed = x_re_second * cast_cos + rotate_half_valid_shape(x_re_second) * cast_sin
    return pto.cast(x_embed, x.dtype)


def rope(
    x: pto.Tensor,
    cos: pto.Tensor,
    sin: pto.Tensor,
    tile_config: LightningIndexerPrologTileConfig,
) -> pto.Tensor:
    assert (
        len(x.shape) == SHAPE_DIM_2
        and len(cos.shape) == SHAPE_DIM_2
        and len(sin.shape) == SHAPE_DIM_2
    )
    seq_size = x.shape[NUM_0]
    d_r = x.shape[NUM_1]
    x_dtype = x.dtype
    pto.set_vec_tile_shapes(
        tile_config.rope_2d[NUM_0],
        tile_config.rope_2d[NUM_1],
    )
    cast_x = pto.cast(x, pto.DataType.DT_FP32)
    if x_dtype == pto.DataType.DT_FP32:
        cast_x = cast_x + 0.0
    cast_cos = pto.cast(cos, pto.DataType.DT_FP32)
    cast_sin = pto.cast(sin, pto.DataType.DT_FP32)
    x_view = pto.reshape(cast_x, [1, seq_size, d_r // NUM_2, NUM_2])
    pto.set_vec_tile_shapes(
        tile_config.rope_4d[0],
        tile_config.rope_4d[1],
        tile_config.rope_4d[2],
        tile_config.rope_4d[3],
    )
    x_trans = pto.transpose(x_view, NUM_2, NUM_3)
    x_re_second = pto.reshape(x_trans, [seq_size, d_r])
    pto.set_vec_tile_shapes(
        tile_config.rope_2d[NUM_0],
        tile_config.rope_2d[NUM_1],
    )
    x_embed = x_re_second * cast_cos + rotate_half(x_re_second) * cast_sin
    return pto.cast(x_embed, x_dtype)


def lightning_indexer_prolog_impl(args: LightningIndexerPrologArgs):
    x = args.x
    qr = args.qr
    q_w = args.q_w
    k_w = args.k_w
    proj_w = args.proj_w
    weight = args.weight
    ln_w = args.ln_w
    ln_b = args.ln_b
    cos = args.cos
    sin = args.sin
    k_cache = args.k_cache
    k_cache_index = args.k_cache_index
    block_table = args.block_table
    query = args.query
    k_cache_out = args.k_cache_out
    tile_cfg = args.tile_config
    unroll_list = args.unroll_list
    params = args.params

    b = x.shape[0]
    seq = x.shape[1]
    head_dim = params.head_dim
    rope_head_dim = params.rope_head_dim
    q_lora_rank = params.q_lora_rank
    dim = params.dim
    head_num = params.head_num

    x_2d = pto.tensor(dtype=x.dtype, shape=[b * seq, dim], name="x_2d")
    qr_2d = pto.tensor(dtype=qr.dtype, shape=[b * seq, q_lora_rank], name="qr_2d")
    cos_2d = pto.tensor(dtype=cos.dtype, shape=[b * seq, rope_head_dim], name="cos_2d")
    sin_2d = pto.tensor(dtype=sin.dtype, shape=[b * seq, rope_head_dim], name="sin_2d")
    ln_w_2d = pto.tensor(dtype=ln_w.dtype, shape=[1, ln_w.shape[0]], name="ln_w_2d")
    ln_b_2d = pto.tensor(dtype=ln_b.dtype, shape=[1, ln_b.shape[0]], name="ln_b_2d")

    for _ in pto.loop(0, 1, 1, name="LOOP_RESHAPE_IN", idx_name="dummy"):
        x_2d[:] = pto.reshape(x, [b * seq, dim], inplace=True)
        qr_2d[:] = pto.reshape(qr, [b * seq, q_lora_rank], inplace=True)
        cos_2d[:] = pto.reshape(cos, [b * seq, rope_head_dim], inplace=True)
        sin_2d[:] = pto.reshape(sin, [b * seq, rope_head_dim], inplace=True)
        ln_w_2d[:] = pto.reshape(ln_w, [1, ln_w.shape[0]], inplace=True)
        ln_b_2d[:] = pto.reshape(ln_b, [1, ln_w.shape[0]], inplace=True)

    for b_idx, unroll_length in pto.loop_unroll(
        0,
        b * seq,
        1,
        name="IndexerPrologLoop",
        idx_name="bsIdx",
        unroll_list=unroll_list,
    ):

        def inner(b_idx, unroll_length):
            tile_bs = unroll_length
            act_bs = tile_bs

            pto.set_semantic_label("QMatmul")
            pto.set_cube_tile_shapes(
                tile_cfg.c1_tile[0],
                tile_cfg.c1_tile[1],
                tile_cfg.c1_tile[2],
                True,
            )

            qr_block = pto.view(
                qr_2d,
                [tile_bs, q_lora_rank],
                [b_idx, 0],
                valid_shape=[act_bs, q_lora_rank],
            )

            q_32 = pto.matmul(qr_block, q_w, pto.DataType.DT_FP32)

            pto.set_semantic_label("QCast")
            pto.set_vec_tile_shapes(
                pto.symbolic_scalar(tile_bs).min(4),
                NUM_64,
                tile_cfg.v1_tile[NUM_1],
            )

            q = pto.cast(
                pto.reshape(q_32, [tile_bs, head_num, head_dim]),
                qr_block.dtype,
            )

            q_rope = pto.view(
                q,
                [tile_bs, head_num, rope_head_dim],
                [0, 0, 0],
                valid_shape=[act_bs, head_num, rope_head_dim],
            )
            q_nope = pto.view(
                q,
                [tile_bs, head_num, head_dim - rope_head_dim],
                [0, 0, rope_head_dim],
                valid_shape=[
                    act_bs,
                    head_num,
                    head_dim - rope_head_dim,
                ],
            )

            q_nope[:] = pto.cast(
                pto.cast(q_nope, pto.DataType.DT_FP32), q_nope.dtype
            )

            pto.set_semantic_label("KMatmul")
            pto.set_cube_tile_shapes(
                tile_cfg.c2_tile[0],
                tile_cfg.c2_tile[1],
                tile_cfg.c2_tile[2],
                True,
            )

            pto.set_vec_tile_shapes(
                tile_cfg.v1_tile[NUM_0],
                tile_cfg.v1_tile[NUM_1],
                tile_cfg.v1_tile[NUM_1],
            )

            x_block = pto.view(
                x_2d,
                [tile_bs, dim],
                [b_idx, 0],
                valid_shape=[act_bs, dim],
            )

            weights = pto.matmul(x_block, proj_w, x_block.dtype)
            pto.assemble(weights, [b_idx, 0], weight)

            k = pto.matmul(x_block, k_w, pto.DataType.DT_FP32)

            k[:] = pto.cast(
                layer_norm(k, ln_w_2d, ln_b_2d, -1),
                x_block.dtype,
            )

            k_rope = pto.view(
                k,
                [tile_bs, rope_head_dim],
                [0, 0],
                valid_shape=[act_bs, rope_head_dim],
            )
            k_nope = pto.view(
                k,
                [tile_bs, head_dim - rope_head_dim],
                [0, rope_head_dim],
                valid_shape=[
                    act_bs,
                    head_dim - rope_head_dim,
                ],
            )

            pto.set_vec_tile_shapes(
                tile_cfg.v1_tile[NUM_0],
                tile_cfg.v1_tile[NUM_1],
                tile_cfg.v1_tile[NUM_2],
            )
            cos_2d[:] = pto.view(
                cos_2d,
                [tile_bs, rope_head_dim],
                [b_idx, 0],
                valid_shape=[act_bs, rope_head_dim],
            )
            sin_2d[:] = pto.view(
                sin_2d,
                [tile_bs, rope_head_dim],
                [b_idx, 0],
                valid_shape=[act_bs, rope_head_dim],
            )

            pto.set_semantic_label("QRope")
            q_roped = rope_3d(
                q_rope,
                cos_2d,
                sin_2d,
                tile_cfg,
            )

            pto.set_semantic_label("KRope")
            pto.set_vec_tile_shapes(
                tile_cfg.v1_tile[NUM_0],
                tile_cfg.v1_tile[NUM_1],
            )
            k_roped = rope(
                k_rope,
                cos_2d,
                sin_2d,
                tile_cfg,
            )

            pto.set_semantic_label("KAssemble")
            pto.set_vec_tile_shapes(
                tile_bs,
                NUM_128,
                NUM_128,
                NUM_128,
            )
            pto.assemble(
                q_roped,
                [b_idx, 0, 0],
                query,
            )
            pto.assemble(
                q_nope,
                [b_idx, 0, rope_head_dim],
                query,
            )

            pto.set_vec_tile_shapes(tile_bs, NUM_128 * NUM_2)
            k_type = k_nope.dtype
            k_nope[:] = pto.cast(
                pto.cast(k_nope, pto.DataType.DT_FP32),
                k_type,
            )

            k_update = pto.concat([k_roped, k_nope], -1)
            k_update_4d = pto.reshape(
                k_update,
                [tile_bs, 1, 1, head_dim],
            )

            index = pto.view(
                k_cache_index,
                [tile_bs, 1],
                [b_idx, 0],
                valid_shape=[act_bs, 1],
            )

            pto.set_vec_tile_shapes(
                tile_bs,
                NUM_128,
                NUM_128,
                NUM_128,
            )
            k_cache_out[:] = pto.scatter_update(
                k_cache,
                -2,
                index,
                k_update_4d,
            )

        inner(b_idx, unroll_length)


def lightning_indexer_prolog_inner(args: LightningIndexerPrologArgs):
    input_tensors = [
        args.x,
        args.qr,
        args.q_w,
        args.k_w,
        args.proj_w,
        args.ln_w,
        args.ln_b,
        args.cos,
        args.sin,
        args.k_cache,
        args.k_cache_index,
        args.block_table,
    ]
    output_tensors = [args.query, args.weight]
    inplace_tensors = [[args.k_cache_out, args.k_cache]]

    with pto.function(
        "LightningIndexerProlog",
        input_tensors,
        output_tensors,
        inplace_tensors,
    ):

        def inside_main():
            lightning_indexer_prolog_impl(args)

        inside_main()


def setup_lightning_indexer_prolog_config():
    pto.set_codegen_options(support_dynamic_unaligned=True)
    pto.set_pass_options(nbuffer_merge_mode=NUM_1,
                         l1_reuse=NUM_4,
                         cube_nbuffer_map={NUM_3: NUM_4},
                         copyin_threshold=NUM_2 * NUM_1024 * NUM_1024)


def build_lightning_indexer_prolog_args(
    cfg: LightningIndexerPrologBuildConfig = LightningIndexerPrologBuildConfig(),
):
    d_bf16 = pto.DataType.DT_BF16
    d_i32 = pto.DataType.DT_INT32

    x = pto.tensor(dtype=d_bf16, shape=[cfg.b, cfg.s1, cfg.dim], name="x")
    qr = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.b, cfg.s1, cfg.q_lora_rank],
        name="qr",
    )
    q_w = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.q_lora_rank, cfg.head_num * cfg.head_dim],
        name="q_w",
    )
    k_w = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.dim, cfg.head_dim],
        name="k_w",
    )
    proj_w = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.dim, cfg.head_num],
        name="proj_w",
    )
    ln_w = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.head_dim],
        name="ln_w",
    )
    ln_b = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.head_dim],
        name="ln_b",
    )
    cos = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.b, cfg.s1, cfg.rope_head_dim],
        name="cos",
    )
    sin = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.b, cfg.s1, cfg.rope_head_dim],
        name="sin",
    )
    k_cache = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.block_num, cfg.block_size, cfg.n_kv, cfg.head_dim],
        name="k_cache",
    )
    k_cache_index = pto.tensor(
        dtype=d_i32,
        shape=[cfg.b, cfg.s1],
        name="k_cache_index",
    )
    block_table = pto.tensor(
        dtype=d_i32,
        shape=[cfg.b, cfg.s2_tile // cfg.block_size],
        name="block_table",
    )

    query = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.b * cfg.s1, cfg.head_num, cfg.head_dim],
        name="qOut",
    )
    weight = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.b * cfg.s1, cfg.head_num],
        name="weightOut",
    )
    k_cache_out = pto.tensor(
        dtype=d_bf16,
        shape=[cfg.block_num, cfg.block_size, cfg.n_kv, cfg.head_dim],
        name="kCacheOut",
    )

    tile_cfg = LightningIndexerPrologTileConfig(
        c1_tile=cfg.c1_tile,
        v1_tile=cfg.v1_tile,
        c2_tile=cfg.c2_tile,
        v2_tile=cfg.v2_tile,
        rope_2d=cfg.rope_2d,
        rope_3d=cfg.rope_3d_vals,
        rope_4d=cfg.rope_4d,
    )

    params = LightningIndexerPrologParams(
        b=cfg.b,
        s1=cfg.s1,
        dim=cfg.dim,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        head_num=cfg.head_num,
        rope_head_dim=cfg.rope_head_dim,
        block_size=cfg.block_size,
        block_num=cfg.block_num,
        n_kv=cfg.n_kv,
        s2=cfg.s2_tile,
        tile_bs=cfg.tile_bs,
    )

    unroll_list: List[int] = [1, 2, 4, 8, 16, 32]

    args = LightningIndexerPrologArgs(
        x=x,
        qr=qr,
        q_w=q_w,
        k_w=k_w,
        proj_w=proj_w,
        ln_w=ln_w,
        ln_b=ln_b,
        cos=cos,
        sin=sin,
        k_cache=k_cache,
        k_cache_index=k_cache_index,
        block_table=block_table,
        query=query,
        weight=weight,
        k_cache_out=k_cache_out,
        tile_config=tile_cfg,
        unroll_list=unroll_list,
        params=params,
    )

    meta = {
        "b": cfg.b,
        "s1": cfg.s1,
        "head_num": cfg.head_num,
        "head_dim": cfg.head_dim,
        "rope_head_dim": cfg.rope_head_dim,
        "q_lora_rank": cfg.q_lora_rank,
        "dim": cfg.dim,
        "n_kv": cfg.n_kv,
        "cache": {
            "blockSize": cfg.block_size,
            "block_num": cfg.block_num,
        },
        "tiles": {
            "c1Tile": cfg.c1_tile,
            "v1Tile": cfg.v1_tile,
            "c2Tile": cfg.c2_tile,
            "v2Tile": cfg.v2_tile,
            "rope2D": cfg.rope_2d,
            "rope3D": cfg.rope_3d_vals,
            "rope4D": cfg.rope_4d,
        },
        "dims": {
            "x": [cfg.b, cfg.s1, cfg.dim],
            "qr": [cfg.b, cfg.s1, cfg.q_lora_rank],
            "q_w": [cfg.q_lora_rank, cfg.head_num * cfg.head_dim],
            "k_w": [cfg.dim, cfg.head_dim],
            "proj_w": [cfg.dim, cfg.head_num],
            "ln_w": [cfg.head_dim],
            "ln_b": [cfg.head_dim],
            "cos/sin": [cfg.b, cfg.s1, cfg.rope_head_dim],
            "k_cache": [
                cfg.block_num,
                cfg.block_size,
                cfg.n_kv,
                cfg.head_dim,
            ],
            "k_cache_index": [cfg.b, cfg.s1],
            "block_table": [cfg.b, cfg.s2_tile // cfg.block_size],
            "query(out)": [
                cfg.b * cfg.s1,
                cfg.head_num,
                cfg.head_dim,
            ],
            "weight(out)": [cfg.b * cfg.s1, cfg.head_num],
            "k_cache_out(out)": [
                cfg.block_num,
                cfg.block_size,
                cfg.n_kv,
                cfg.head_dim,
            ],
        },
        "dtype": str(d_bf16),
        "tile_bs": cfg.tile_bs,
        "unrollList": sorted(list(unroll_list)),
    }

    return args, meta


@pytest.mark.skip(reason="There is a probability of failure")
def test_lightning_indexer_prolog():
    logging.basicConfig(level=logging.INFO)
    setup_lightning_indexer_prolog_config()
    args, meta = build_lightning_indexer_prolog_args()
    logging.info({"Sanity": meta})
    lightning_indexer_prolog_inner(args)
    assert True
