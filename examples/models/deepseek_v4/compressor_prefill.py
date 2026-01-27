import pypto
from compressor_decode import (
    softmax,
    rms_norm,
    interleaved_rope_3d,
    scatter_update_3d,
    Rope3dTileConfig,
)


def overlap_trans_dynamic(x: pypto.Tensor, y: pypto.Tensor):
    b, s, r, d = x.shape
    d = d // 2
    x1 = pypto.view(x, [b, s, r, d], [0, 0, 0, d])
    x2 = pypto.view(x, [b, s - 1, r, d], [0, 0, 0, 0])
    res = pypto.view(x, [b, 1, r, d], [0, s - 1, 0, 0])
    # y = pypto.full([b, 1, r, d], value, x.dtype)
    z = pypto.concat([y, x2], 1)
    z = pypto.concat([z, x1], 2)
    return z, res


@pypto.jit(pass_options={})
def compressor_prefill_ratio_4_rotate(
    x,
    kv_state,
    score_state,
    cache_index_2d,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    out,
    kv_state_out,
    score_state_out,
    ratio,
    kv_len_int_dy,
    rope_head_dim,
):
    # pypto.set_debug_options(runtime_debug_mode=1)

    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    overlap = ratio == 4
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    s_tile = 128
    kv_len_int = kv_len_int_dy.concrete()
    s_loop = (kv_len_int + s_tile - 1) // s_tile

    for b_idx in pypto.loop(b):
        if overlap:
            pypto.set_vec_tile_shapes(1, 1, 4, d)
            kv_full = pypto.full([1, 1, ratio, d], 0, pypto.DT_FP32)
            score_full = pypto.full([1, 1, ratio, d], float("-inf"), pypto.DT_FP32)

        pypto.set_vec_tile_shapes(1, 8)
        cache_index_2d_tile = pypto.view(
            cache_index_2d, [1, cache_index_2d.shape[1]], [b_idx, 0]
        )
        pypto.set_vec_tile_shapes(1, 8, 128)
        kv_state_tile = pypto.view(
            kv_state, [1, kv_state.shape[1], kv_state.shape[2]], [b_idx, 0, 0]
        )
        score_state_tile = pypto.view(
            score_state, [1, score_state.shape[1], score_state.shape[2]], [b_idx, 0, 0]
        )

        for s_idx in pypto.loop(s_loop):
            act_s_tile = (kv_len_int - s_idx * s_tile).min(s_tile)
            pypto.set_vec_tile_shapes(1, 64, 64)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
            x_tile = pypto.view(
                x,
                [1, s_tile, h],
                [b_idx, s_idx * s_tile, 0],
                valid_shape=[b, act_s_tile, h],
            )

            x_tile = pypto.reshape(x_tile, [s_tile, h], valid_shape=[act_s_tile, h])
            pypto.set_vec_tile_shapes(64, 64)

            x_tile = pypto.cast(x_tile, pypto.DT_FP32)  ## b,s,h
            kv = pypto.matmul(x_tile, wkv, pypto.DT_FP32)  ## b*s,2d
            kv = pypto.reshape(
                kv, [1, s_tile, wkv.shape[1]], valid_shape=[1, act_s_tile, wkv.shape[1]]
            )  ## b,s,2d
            score = pypto.matmul(x_tile, wgate, pypto.DT_FP32)
            score = pypto.reshape(
                score,
                [1, s_tile, wgate.shape[1]],
                valid_shape=[1, act_s_tile, wgate.shape[1]],
            )  ## b,s,2d

            should_compute = True
            remainder = kv_len_int % ratio
            cutoff = kv_len_int - remainder
            offset = ratio * overlap

            cutoff_128 = ((kv_len_int // ratio) * ratio) % s_tile
            update_state = cutoff >= ratio

            if pypto.cond(pypto.is_loop_end(s_idx)):
                if cutoff_128 >= ratio:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(
                        kv, [1, ratio, wkv.shape[1]], [0, cutoff_128 - ratio, 0]
                    )
                    score_view = pypto.view(
                        score, [1, ratio, wgate.shape[1]], [0, cutoff_128 - ratio, 0]
                    )

                    pypto.set_vec_tile_shapes(1, 8, coff * d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(
                        score_state_tile, index, score_view
                    )

                    if remainder > 0:
                        index = pypto.view(
                            cache_index_2d_tile,
                            [1, ratio],
                            [0, offset],
                            valid_shape=[1, remainder],
                        )
                        kv_view = pypto.view(
                            kv,
                            [1, ratio, wkv.shape[1]],
                            [0, cutoff_128, 0],
                            valid_shape=[1, remainder, wkv.shape[1]],
                        )
                        score_view = pypto.view(
                            score,
                            [1, ratio, wgate.shape[1]],
                            [0, cutoff_128, 0],
                            valid_shape=[1, remainder, wgate.shape[1]],
                        )
                        ape_view = pypto.view(
                            ape,
                            [ratio, coff * d],
                            [0, 0],
                            valid_shape=[remainder, coff * d],
                        )

                        pypto.set_vec_tile_shapes(1, 8, coff * d)
                        score_view = pypto.add(score_view, ape_view)
                        kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                        score_state_tile = scatter_update_3d(
                            score_state_tile, index, score_view
                        )

                elif update_state and cutoff_128 == 0 and remainder > 0:
                    index = pypto.view(
                        cache_index_2d_tile,
                        [1, ratio],
                        [0, offset],
                        valid_shape=[1, remainder],
                    )
                    kv_view = pypto.view(
                        kv,
                        [1, ratio, wkv.shape[1]],
                        [0, cutoff_128, 0],
                        valid_shape=[1, remainder, wkv.shape[1]],
                    )
                    score_view = pypto.view(
                        score,
                        [1, ratio, wgate.shape[1]],
                        [0, cutoff_128, 0],
                        valid_shape=[1, remainder, wgate.shape[1]],
                    )
                    ape_view = pypto.view(
                        ape,
                        [ratio, coff * d],
                        [0, 0],
                        valid_shape=[remainder, coff * d],
                    )

                    pypto.set_vec_tile_shapes(1, 8, coff * d)
                    score_view = pypto.add(score_view, ape_view)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(
                        score_state_tile, index, score_view
                    )
                    should_compute = False  # 跳过

                elif update_state and cutoff_128 == 0 and remainder == 0:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(
                        kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0]
                    )
                    score_view = pypto.view(
                        score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0]
                    )

                    pypto.set_vec_tile_shapes(1, 8, coff * d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(
                        score_state_tile, index, score_view
                    )
                if update_state:
                    ## 更新kv_state和score_state
                    pypto.set_vec_tile_shapes(1, 128, d)
                    pypto.assemble(kv_state_tile, [b_idx, 0, 0], kv_state_out)
                    pypto.assemble(score_state_tile, [b_idx, 0, 0], score_state_out)

            if (
                update_state
                and cutoff_128 == 0
                and remainder > 0
                and pypto.cond(s_idx == s_loop - 2)
            ):
                index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                kv_view = pypto.view(
                    kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0]
                )
                score_view = pypto.view(
                    score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0]
                )

                pypto.set_vec_tile_shapes(1, 8, coff * d)
                score_view = pypto.add(score_view, ape)
                kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                score_state_tile = scatter_update_3d(
                    score_state_tile, index, score_view
                )

            if should_compute:
                pypto.set_vec_tile_shapes(1, 4, d)
                kv = pypto.reshape(
                    kv, [1, kv.shape[1] // ratio, ratio, coff * d]
                )  ## b,cut,4,2d
                score = pypto.reshape(
                    score, [1, score.shape[1] // ratio, ratio, coff * d]
                )  ## b,cut,4,2d
                pypto.set_vec_tile_shapes(1, 4, 4, d)
                score = pypto.add(score, ape)  ## b,cut,4,2d

                if overlap:
                    ## overlap_trans\softmax\mul\sum
                    ## 两个overlap_trans分成不同图的时候，精度才对
                    pypto.set_pass_options(sg_set_scope=1)
                    kv, kv_full_tmp = overlap_trans_dynamic(kv, kv_full)  ## b,cut,8,d
                    kv_full[:] = kv_full_tmp
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_pass_options(sg_set_scope=2)
                    score, score_full_tmp = overlap_trans_dynamic(
                        score, score_full
                    )  ## b,cut,8,d
                    score_full[:] = score_full_tmp
                    score = softmax(score, 2)  ## b,cut,8,d
                    pypto.set_pass_options(sg_set_scope=-1)
                else:
                    score = softmax(score, 2)  ## b,cut,8,d

                ## 不用scope和前面隔开会有kv*score的精度问题
                pypto.set_pass_options(sg_set_scope=3)
                kv = kv * score  ## b,cut,8,d
                kv = pypto.sum(kv, 2)  ## b,cut,d
                pypto.set_pass_options(sg_set_scope=-1)

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = rms_norm(pypto.cast(kv, dtype), weight)  ## b,cut,d

                kv_nope = kv[:, :, : d - rope_head_dim]
                kv_rope = kv[:, :, d - rope_head_dim :]
                sin_tile = pypto.view(
                    sin, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0]
                )  ## b, cut, 64
                cos_tile = pypto.view(
                    cos, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0]
                )
                rope3d_tile_config = Rope3dTileConfig([1, 32, 32], [1, 32, 32, 32])
                kv_rope = interleaved_rope_3d(
                    kv_rope, cos_tile, sin_tile, rope3d_tile_config
                )
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1)  ## b,cut,d

                kv = pypto.reshape(kv, [kv.shape[1], d])
                kv = pypto.matmul(kv, hadamard, pypto.DT_BF16)  ## b*cut,d
                kv = pypto.reshape(kv, [1, kv.shape[0], d])  ## b,cut,d
                if pypto.cond(pypto.is_loop_end(s_idx)):
                    kv = pypto.view(
                        kv,
                        [1, s_tile // ratio, d],
                        [0, 0, 0],
                        valid_shape=[1, cutoff_128 // ratio, d],
                    )
                    kv = kv + 0

                pypto.assemble(kv, [b_idx, s_idx * s_tile // ratio, 0], out)
                ## 更新kv_cache和score_cache


@pypto.jit(pass_options={})
def compressor_prefill_ratio_4(
    x,
    kv_state,
    score_state,
    cache_index_2d,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    out,
    kv_state_out,
    score_state_out,
    ratio,
    kv_len_int_dy,
    rope_head_dim,
):
    # pypto.set_debug_options(runtime_debug_mode=1)

    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    overlap = ratio == 4
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    s_tile = 128
    kv_len_int = kv_len_int_dy.concrete()
    s_loop = (kv_len_int + s_tile - 1) // s_tile

    for b_idx in pypto.loop(b):
        if overlap:
            pypto.set_vec_tile_shapes(1, 1, 4, d)
            kv_full = pypto.full([1, 1, ratio, d], 0, pypto.DT_FP32)
            score_full = pypto.full([1, 1, ratio, d], float("-inf"), pypto.DT_FP32)

        pypto.set_vec_tile_shapes(1, 8)
        cache_index_2d_tile = pypto.view(
            cache_index_2d, [1, cache_index_2d.shape[1]], [b_idx, 0]
        )
        pypto.set_vec_tile_shapes(1, 8, 128)
        kv_state_tile = pypto.view(
            kv_state, [1, kv_state.shape[1], kv_state.shape[2]], [b_idx, 0, 0]
        )
        score_state_tile = pypto.view(
            score_state, [1, score_state.shape[1], score_state.shape[2]], [b_idx, 0, 0]
        )

        for s_idx in pypto.loop(s_loop):
            act_s_tile = (kv_len_int - s_idx * s_tile).min(s_tile)
            pypto.set_vec_tile_shapes(1, 64, 64)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
            x_tile = pypto.view(
                x,
                [1, s_tile, h],
                [b_idx, s_idx * s_tile, 0],
                valid_shape=[b, act_s_tile, h],
            )

            x_tile = pypto.reshape(x_tile, [s_tile, h], valid_shape=[act_s_tile, h])
            pypto.set_vec_tile_shapes(64, 64)

            x_tile = pypto.cast(x_tile, pypto.DT_FP32)  ## b,s,h
            kv = pypto.matmul(x_tile, wkv, pypto.DT_FP32)  ## b*s,2d
            kv = pypto.reshape(
                kv, [1, s_tile, wkv.shape[1]], valid_shape=[1, act_s_tile, wkv.shape[1]]
            )  ## b,s,2d
            score = pypto.matmul(x_tile, wgate, pypto.DT_FP32)
            score = pypto.reshape(
                score,
                [1, s_tile, wgate.shape[1]],
                valid_shape=[1, act_s_tile, wgate.shape[1]],
            )  ## b,s,2d

            should_compute = True
            remainder = kv_len_int % ratio
            cutoff = kv_len_int - remainder
            offset = ratio * overlap

            cutoff_128 = ((kv_len_int // ratio) * ratio) % s_tile
            update_state = cutoff >= ratio

            if pypto.cond(pypto.is_loop_end(s_idx)):
                if cutoff_128 >= ratio:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(
                        kv, [1, ratio, wkv.shape[1]], [0, cutoff_128 - ratio, 0]
                    )
                    score_view = pypto.view(
                        score, [1, ratio, wgate.shape[1]], [0, cutoff_128 - ratio, 0]
                    )

                    pypto.set_vec_tile_shapes(1, 8, coff * d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(
                        score_state_tile, index, score_view
                    )

                    if remainder > 0:
                        index = pypto.view(
                            cache_index_2d_tile,
                            [1, ratio],
                            [0, offset],
                            valid_shape=[1, remainder],
                        )
                        kv_view = pypto.view(
                            kv,
                            [1, ratio, wkv.shape[1]],
                            [0, cutoff_128, 0],
                            valid_shape=[1, remainder, wkv.shape[1]],
                        )
                        score_view = pypto.view(
                            score,
                            [1, ratio, wgate.shape[1]],
                            [0, cutoff_128, 0],
                            valid_shape=[1, remainder, wgate.shape[1]],
                        )
                        ape_view = pypto.view(
                            ape,
                            [ratio, coff * d],
                            [0, 0],
                            valid_shape=[remainder, coff * d],
                        )

                        pypto.set_vec_tile_shapes(1, 8, coff * d)
                        score_view = pypto.add(score_view, ape_view)
                        kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                        score_state_tile = scatter_update_3d(
                            score_state_tile, index, score_view
                        )

                elif update_state and cutoff_128 == 0 and remainder > 0:
                    index = pypto.view(
                        cache_index_2d_tile,
                        [1, ratio],
                        [0, offset],
                        valid_shape=[1, remainder],
                    )
                    kv_view = pypto.view(
                        kv,
                        [1, ratio, wkv.shape[1]],
                        [0, cutoff_128, 0],
                        valid_shape=[1, remainder, wkv.shape[1]],
                    )
                    score_view = pypto.view(
                        score,
                        [1, ratio, wgate.shape[1]],
                        [0, cutoff_128, 0],
                        valid_shape=[1, remainder, wgate.shape[1]],
                    )
                    ape_view = pypto.view(
                        ape,
                        [ratio, coff * d],
                        [0, 0],
                        valid_shape=[remainder, coff * d],
                    )

                    pypto.set_vec_tile_shapes(1, 8, coff * d)
                    score_view = pypto.add(score_view, ape_view)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(
                        score_state_tile, index, score_view
                    )
                    should_compute = False  # 跳过

                elif update_state and cutoff_128 == 0 and remainder == 0:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(
                        kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0]
                    )
                    score_view = pypto.view(
                        score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0]
                    )

                    pypto.set_vec_tile_shapes(1, 8, coff * d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(
                        score_state_tile, index, score_view
                    )
                if update_state:
                    ## 更新kv_state和score_state
                    pypto.set_vec_tile_shapes(1, 128, d)
                    pypto.assemble(kv_state_tile, [b_idx, 0, 0], kv_state_out)
                    pypto.assemble(score_state_tile, [b_idx, 0, 0], score_state_out)

            if (
                update_state
                and cutoff_128 == 0
                and remainder > 0
                and pypto.cond(s_idx == s_loop - 2)
            ):
                index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                kv_view = pypto.view(
                    kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0]
                )
                score_view = pypto.view(
                    score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0]
                )

                pypto.set_vec_tile_shapes(1, 8, coff * d)
                score_view = pypto.add(score_view, ape)
                kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                score_state_tile = scatter_update_3d(
                    score_state_tile, index, score_view
                )

            if should_compute:
                pypto.set_vec_tile_shapes(1, 4, d)
                kv = pypto.reshape(
                    kv, [1, kv.shape[1] // ratio, ratio, coff * d]
                )  ## b,cut,4,2d
                score = pypto.reshape(
                    score, [1, score.shape[1] // ratio, ratio, coff * d]
                )  ## b,cut,4,2d
                pypto.set_vec_tile_shapes(1, 4, 4, d)
                score = pypto.add(score, ape)  ## b,cut,4,2d

                if overlap:
                    ## overlap_trans\softmax\mul\sum
                    ## 两个overlap_trans分成不同图的时候，精度才对
                    pypto.set_pass_options(sg_set_scope=1)
                    kv, kv_full_tmp = overlap_trans_dynamic(kv, kv_full)  ## b,cut,8,d
                    kv_full[:] = kv_full_tmp
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_pass_options(sg_set_scope=2)
                    score, score_full_tmp = overlap_trans_dynamic(
                        score, score_full
                    )  ## b,cut,8,d
                    score_full[:] = score_full_tmp
                    score = softmax(score, 2)  ## b,cut,8,d
                    pypto.set_pass_options(sg_set_scope=-1)
                else:
                    score = softmax(score, 2)  ## b,cut,8,d

                ## 不用scope和前面隔开会有kv*score的精度问题
                pypto.set_pass_options(sg_set_scope=3)
                kv = kv * score  ## b,cut,8,d
                kv = pypto.sum(kv, 2)  ## b,cut,d
                pypto.set_pass_options(sg_set_scope=-1)

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = rms_norm(pypto.cast(kv, dtype), weight)  ## b,cut,d

                kv_nope = kv[:, :, : d - rope_head_dim]
                kv_rope = kv[:, :, d - rope_head_dim :]
                sin_tile = pypto.view(
                    sin, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0]
                )  ## b, cut, 64
                cos_tile = pypto.view(
                    cos, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0]
                )
                rope3d_tile_config = Rope3dTileConfig([1, 32, 32], [1, 32, 32, 32])
                kv_rope = interleaved_rope_3d(
                    kv_rope, cos_tile, sin_tile, rope3d_tile_config
                )
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1)  ## b,cut,d

                if pypto.cond(pypto.is_loop_end(s_idx)):
                    kv = pypto.view(
                        kv,
                        [1, s_tile // ratio, d],
                        [0, 0, 0],
                        valid_shape=[1, cutoff_128 // ratio, d],
                    )
                    kv = kv + 0

                pypto.assemble(kv, [b_idx, s_idx * s_tile // ratio, 0], out)
                ## 更新kv_cache和score_cache


@pypto.jit(pass_options={})
def compressor_prefill_ratio_128(
    x,
    kv_state,
    score_state,
    cache_index_2d,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    out,
    kv_state_out,
    score_state_out,
    ratio,
    kv_len_int_dy,
    rope_head_dim,
):
    # pypto.set_debug_options(runtime_debug_mode=1)

    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    overlap = ratio == 4
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    s_tile = 128
    kv_len_int = kv_len_int_dy.concrete()
    s_loop = (kv_len_int + s_tile - 1) // s_tile

    for b_idx in pypto.loop(b):
        pypto.set_vec_tile_shapes(1, 128)
        cache_index_2d_tile = pypto.view(
            cache_index_2d, [1, cache_index_2d.shape[1]], [b_idx, 0]
        )
        pypto.set_vec_tile_shapes(1, 128, 128)
        kv_state_tile = pypto.view(
            kv_state, [1, kv_state.shape[1], kv_state.shape[2]], [b_idx, 0, 0]
        )
        score_state_tile = pypto.view(
            score_state, [1, score_state.shape[1], score_state.shape[2]], [b_idx, 0, 0]
        )
        for s_idx in pypto.loop(s_loop):
            act_s_tile = (kv_len_int - s_idx * s_tile).min(s_tile)
            pypto.set_vec_tile_shapes(1, 64, 64)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
            x_tile = pypto.view(
                x,
                [1, s_tile, h],
                [b_idx, s_idx * s_tile, 0],
                valid_shape=[b, act_s_tile, h],
            )

            x_tile = pypto.reshape(x_tile, [s_tile, h], valid_shape=[act_s_tile, h])
            pypto.set_vec_tile_shapes(64, 64)

            x_tile = pypto.cast(x_tile, pypto.DT_FP32)  ## b,s,h
            kv = pypto.matmul(x_tile, wkv, pypto.DT_FP32)  ## b*s,2d
            kv = pypto.reshape(
                kv, [1, s_tile, wkv.shape[1]], valid_shape=[1, act_s_tile, wkv.shape[1]]
            )  ## b,s,2d
            score = pypto.matmul(x_tile, wgate, pypto.DT_FP32)
            score = pypto.reshape(
                score,
                [1, s_tile, wgate.shape[1]],
                valid_shape=[1, act_s_tile, wgate.shape[1]],
            )  ## b,s,2d

            ## Cut
            remainder = kv_len_int % ratio
            if remainder > 0 and pypto.cond(pypto.is_loop_end(s_idx)):
                pypto.set_vec_tile_shapes(1, 8, coff * d)
                index = pypto.view(
                    cache_index_2d_tile,
                    [1, s_tile],
                    [0, 0],
                    valid_shape=[1, act_s_tile],
                )
                ape_view = pypto.view(
                    ape, [s_tile, d], [0, 0], valid_shape=[act_s_tile, d]
                )
                score = pypto.add(score, ape_view)
                kv_state_tile = scatter_update_3d(kv_state_tile, index, kv)
                score_state_tile = scatter_update_3d(score_state_tile, index, score)

                ## 更新kv_state和score_state
                pypto.set_vec_tile_shapes(1, 128, d)
                pypto.assemble(kv_state_tile, [b_idx, 0, 0], kv_state_out)
                pypto.assemble(score_state_tile, [b_idx, 0, 0], score_state_out)
            else:
                pypto.set_vec_tile_shapes(1, 64, d)
                kv = pypto.reshape(kv, [1, 1, ratio, coff * d])  ## b,cut,4,2d
                score = pypto.reshape(score, [1, 1, ratio, coff * d])  ## b,cut,4,2d

                pypto.set_vec_tile_shapes(1, 4, 4, d)
                score = pypto.add(score, ape)  ## b,cut,4,2d
                score = softmax(score, 2)  ## b,cut,8,d

                ## 不用scope和前面隔开会有kv*score的精度问题
                pypto.set_pass_options(sg_set_scope=3)
                kv = kv * score  ## b,cut,8,d
                kv = pypto.sum(kv, 2)  ## b,cut,d
                pypto.set_pass_options(sg_set_scope=-1)

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = rms_norm(pypto.cast(kv, dtype), weight)  ## b,cut,d

                kv_nope = kv[:, :, : d - rope_head_dim]
                kv_rope = kv[:, :, d - rope_head_dim :]
                sin_tile = pypto.view(
                    sin, kv_rope.shape, [b_idx, s_idx, 0]
                )  ## b, cut, 64
                cos_tile = pypto.view(cos, kv_rope.shape, [b_idx, s_idx, 0])
                rope3d_tile_config = Rope3dTileConfig([1, 32, 32], [1, 32, 32, 32])
                kv_rope = interleaved_rope_3d(
                    kv_rope, cos_tile, sin_tile, rope3d_tile_config
                )
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1)  ## b,cut,d

                pypto.assemble(kv, [b_idx, s_idx, 0], out)

                ## 更新kv_cache和score_cache
