import pypto
import torch
import os
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import List

# Constants for shape dimensions
SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

@dataclass
class Rope3dTileConfig:
    # two_dim_tile: List[int]
    three_dim_tile: List[int]
    four_dim_tile: List[int]

def compressor(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ratio, start_pos, kv_len_int, rope_head_dim, rotate, **kwargs):
    inputs = {
        x: [0],
        kv_state: [0],
        score_state: [0],
        cache_index_2d: [],
        sin: [],
        cos: [],
        wkv: [],
        wgate: [],
        ape: [],
        weight: []
    }
    outputs = {
        out: [0],
        kv_state_out: [0],
        score_state_out: [0]
    }
    kv_len_int_dy = pypto.SymbolicScalar(kv_len_int)

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    if start_pos == 0 and rotate and ratio == 4:
        hadamard = kwargs.get('hadamard')
        hadamard = pypto.from_torch(hadamard, dynamic_axis=[])
        compressor_prefill_ratio_4_rotate(*pto_inputs, hadamard, *pto_outputs, ratio, kv_len_int_dy, rope_head_dim)
    elif start_pos == 0 and not rotate and ratio == 4:
        compressor_prefill_ratio_4(*pto_inputs, *pto_outputs, ratio, kv_len_int_dy, rope_head_dim)
    elif start_pos == 0 and not rotate and ratio == 128:
        compressor_prefill_ratio_128(*pto_inputs, *pto_outputs, ratio, kv_len_int_dy, rope_head_dim)


def overlap_trans(x: pypto.Tensor, value):
    b, s, r, d = x.shape
    d = d // 2
    x1 = pypto.view(x, [b, s, r, d], [0, 0, 0, d])
    x2 = pypto.view(x, [b, s - 1, r, d], [0, 0, 0, 0])
    y = pypto.full([b, 1, r, d], value, x.dtype)
    z = pypto.concat([y, x2], 1)
    z = pypto.concat([z, x1], 2)
    return z

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

def softmax(x: pypto.Tensor, dim) -> pypto.Tensor:
    xmax = pypto.amax(x, dim, keepdim=True)
    xsub = pypto.sub(x, xmax)
    xexp = pypto.exp(xsub)
    xsum = pypto.sum(xexp, dim, keepdim=True)
    xdiv = pypto.div(xexp, xsum)
    return xdiv

def rms_norm(input_tensor: pypto.Tensor, gamma: pypto.Tensor, epsilon=1e-6) -> pypto.Tensor:
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim = True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y

def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1], -1)

def interleaved_rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor, rope_3d_config: Rope3dTileConfig) -> pypto.Tensor:
    pypto.set_vec_tile_shapes(*rope_3d_config.three_dim_tile) # (1, 64, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(*rope_3d_config.four_dim_tile)  # (1, 64, 128, 128)
    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    x_trans = pypto.transpose(x_view, 2, 3)
    x_trans = pypto.reshape(x_trans, x.shape)
    x_trans = rotate_half(x_trans)
    x_trans_reshape = pypto.reshape(x_trans, [x.shape[0], x.shape[1], 2, x.shape[2] // 2])
    x_trans_embed = pypto.transpose(x_trans_reshape, 2, 3)
    x_second = pypto.reshape(x_trans_embed, x.shape)

    x_embed = cast_x * cast_cos + x_second * cast_sin

    return pypto.cast(x_embed, x.dtype)

def scatter_update_3d(input, index, src):
    input_shape = input.shape
    d = src.shape[2]
    pypto.set_vec_tile_shapes(1, 64, d)
    src = pypto.reshape(src, [src.shape[0]*src.shape[1], src.shape[2]])
    input = pypto.reshape(input, [input.shape[0]*input.shape[1], input.shape[2]])
    pypto.set_vec_tile_shapes(1, 64, d)
    index = pypto.reshape(index, [1, index.shape[0]*index.shape[1]])
    pypto.set_vec_tile_shapes(64, d)
    output = pypto.scatter_update(input, -2, index, src)
    return pypto.reshape(output, input_shape)


@pypto.jit(
    pass_options={}
)
def compressor_prefill_ratio_4_rotate(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, hadamard, out, kv_state_out, score_state_out, ratio, kv_len_int_dy, rope_head_dim):
    # pypto.set_debug_options(runtime_debug_mode=1)

    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    overlap = (ratio == 4)
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    s_tile = 128
    kv_len_int = kv_len_int_dy.concrete()
    s_loop = (kv_len_int + s_tile -1 ) // s_tile

    for b_idx in pypto.loop(b):
        if overlap:
            pypto.set_vec_tile_shapes(1, 1, 4, d)
            kv_full = pypto.full([1, 1, ratio, d], 0, pypto.DT_FP32)
            score_full = pypto.full([1, 1, ratio, d], float("-inf"), pypto.DT_FP32)

        pypto.set_vec_tile_shapes(1, 8)
        cache_index_2d_tile = pypto.view(cache_index_2d, [1, cache_index_2d.shape[1]], [b_idx, 0])
        pypto.set_vec_tile_shapes(1, 8, 128)
        kv_state_tile = pypto.view(kv_state, [1, kv_state.shape[1], kv_state.shape[2]], [b_idx, 0, 0])
        score_state_tile = pypto.view(score_state, [1, score_state.shape[1], score_state.shape[2]], [b_idx, 0, 0])

        for s_idx in pypto.loop(s_loop):
            act_s_tile = (kv_len_int - s_idx*s_tile).min(s_tile)
            pypto.set_vec_tile_shapes(1, 64, 64)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
            x_tile = pypto.view(x, [1, s_tile, h], [b_idx, s_idx*s_tile, 0], valid_shape=[b, act_s_tile, h])

            x_tile = pypto.reshape(x_tile, [s_tile, h], valid_shape=[act_s_tile, h])
            pypto.set_vec_tile_shapes(64, 64)

            x_tile = pypto.cast(x_tile, pypto.DT_FP32) ## b,s,h
            kv = pypto.matmul(x_tile, wkv, pypto.DT_FP32) ## b*s,2d
            kv = pypto.reshape(kv, [1, s_tile, wkv.shape[1]], valid_shape=[1, act_s_tile, wkv.shape[1]]) ## b,s,2d
            score = pypto.matmul(x_tile, wgate, pypto.DT_FP32)
            score = pypto.reshape(score, [1, s_tile, wgate.shape[1]], valid_shape=[1, act_s_tile, wgate.shape[1]]) ## b,s,2d

            should_compress = (kv_len_int >= ratio)
            should_compute = True
            remainder = kv_len_int % ratio
            cutoff = kv_len_int - remainder
            offset = ratio * overlap

            cutoff_128 = ((kv_len_int // ratio) * ratio) % s_tile
            update_state = cutoff >= ratio

            if pypto.cond(pypto.is_loop_end(s_idx)):
                if cutoff_128 >= ratio:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, cutoff_128-ratio, 0])
                    score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, cutoff_128-ratio, 0])

                    pypto.set_vec_tile_shapes(1, 8, coff*d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(score_state_tile, index, score_view)

                    if remainder > 0: 
                        index = pypto.view(cache_index_2d_tile, [1, ratio], [0, offset], valid_shape=[1, remainder])
                        kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wkv.shape[1]])
                        score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wgate.shape[1]])
                        ape_view = pypto.view(ape, [ratio, coff*d], [0, 0], valid_shape=[remainder, coff*d])

                        pypto.set_vec_tile_shapes(1, 8, coff*d)
                        score_view = pypto.add(score_view, ape_view)
                        kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                        score_state_tile = scatter_update_3d(score_state_tile, index, score_view)

                elif update_state and cutoff_128 == 0 and remainder > 0:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, offset], valid_shape=[1, remainder])
                    kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wkv.shape[1]])
                    score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wgate.shape[1]])
                    ape_view = pypto.view(ape, [ratio, coff*d], [0, 0], valid_shape=[remainder, coff*d])

                    pypto.set_vec_tile_shapes(1, 8, coff*d)
                    score_view = pypto.add(score_view, ape_view)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(score_state_tile, index, score_view)
                    should_compute = False # 跳过

                elif update_state and cutoff_128 == 0 and remainder == 0:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0])
                    score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0])

                    pypto.set_vec_tile_shapes(1, 8, coff*d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(score_state_tile, index, score_view)
                if update_state:
                    ## 更新kv_state和score_state
                    pypto.set_vec_tile_shapes(1, 128, d)
                    pypto.assemble(kv_state_tile, [b_idx, 0, 0], kv_state_out)
                    pypto.assemble(score_state_tile, [b_idx, 0, 0], score_state_out)


            if update_state and cutoff_128 == 0 and remainder > 0 and pypto.cond(s_idx == s_loop - 2):
                index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0])
                score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0])

                pypto.set_vec_tile_shapes(1, 8, coff*d)
                score_view = pypto.add(score_view, ape)
                kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                score_state_tile = scatter_update_3d(score_state_tile, index, score_view)

            if should_compute:
                pypto.set_vec_tile_shapes(1, 4, d)
                kv = pypto.reshape(kv, [1, kv.shape[1]//ratio, ratio, coff*d]) ## b,cut,4,2d
                score = pypto.reshape(score, [1, score.shape[1]//ratio, ratio, coff*d]) ## b,cut,4,2d
                pypto.set_vec_tile_shapes(1, 4, 4, d)
                score = pypto.add(score, ape) ## b,cut,4,2d

                if overlap:
                    ## overlap_trans\softmax\mul\sum
                    ## 两个overlap_trans分成不同图的时候，精度才对
                    pypto.set_pass_options(sg_set_scope=1)
                    kv, kv_full_tmp = overlap_trans_dynamic(kv, kv_full) ## b,cut,8,d
                    kv_full[:] = kv_full_tmp
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_pass_options(sg_set_scope=2)
                    score, score_full_tmp = overlap_trans_dynamic(score, score_full) ## b,cut,8,d
                    score_full[:] = score_full_tmp
                    score = softmax(score, 2) ## b,cut,8,d
                    pypto.set_pass_options(sg_set_scope=-1)
                else:
                    score = softmax(score, 2) ## b,cut,8,d

                ## 不用scope和前面隔开会有kv*score的精度问题
                pypto.set_pass_options(sg_set_scope=3)
                kv = kv * score ## b,cut,8,d
                kv = pypto.sum(kv, 2) ## b,cut,d
                pypto.set_pass_options(sg_set_scope=-1)

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,cut,d

                kv_nope = kv[:, :, :d-rope_head_dim]
                kv_rope = kv[:, :, d-rope_head_dim:]
                sin_tile = pypto.view(sin, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0]) ## b, cut, 64
                cos_tile = pypto.view(cos, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0])
                rope3d_tile_config = Rope3dTileConfig(
                    [1, 32, 32],
                    [1, 32, 32, 32]
                )
                kv_rope = interleaved_rope_3d(kv_rope, cos_tile, sin_tile, rope3d_tile_config)
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,cut,d

                kv = pypto.reshape(kv, [kv.shape[1], d])
                kv = pypto.matmul(kv, hadamard, pypto.DT_BF16) ## b*cut,d
                kv = pypto.reshape(kv, [1, kv.shape[0], d]) ## b,cut,d
                if pypto.cond(pypto.is_loop_end(s_idx)):
                    kv = pypto.view(kv, [1, s_tile // ratio, d], [0, 0, 0], valid_shape=[1, cutoff_128 // ratio, d])
                    kv = kv + 0

                pypto.assemble(kv, [b_idx, s_idx * s_tile // ratio, 0], out)
                ## 更新kv_cache和score_cache

@pypto.jit(
    pass_options={}
)
def compressor_prefill_ratio_4(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ratio, kv_len_int_dy, rope_head_dim):
    # pypto.set_debug_options(runtime_debug_mode=1)

    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    overlap = (ratio == 4)
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    s_tile = 128
    kv_len_int = kv_len_int_dy.concrete()
    s_loop = (kv_len_int + s_tile -1 ) // s_tile

    for b_idx in pypto.loop(b):
        if overlap:
            pypto.set_vec_tile_shapes(1, 1, 4, d)
            kv_full = pypto.full([1, 1, ratio, d], 0, pypto.DT_FP32)
            score_full = pypto.full([1, 1, ratio, d], float("-inf"), pypto.DT_FP32)

        pypto.set_vec_tile_shapes(1, 8)
        cache_index_2d_tile = pypto.view(cache_index_2d, [1, cache_index_2d.shape[1]], [b_idx, 0])
        pypto.set_vec_tile_shapes(1, 8, 128)
        kv_state_tile = pypto.view(kv_state, [1, kv_state.shape[1], kv_state.shape[2]], [b_idx, 0, 0])
        score_state_tile = pypto.view(score_state, [1, score_state.shape[1], score_state.shape[2]], [b_idx, 0, 0])

        for s_idx in pypto.loop(s_loop):
            act_s_tile = (kv_len_int - s_idx*s_tile).min(s_tile)
            pypto.set_vec_tile_shapes(1, 64, 64)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
            x_tile = pypto.view(x, [1, s_tile, h], [b_idx, s_idx*s_tile, 0], valid_shape=[b, act_s_tile, h])

            x_tile = pypto.reshape(x_tile, [s_tile, h], valid_shape=[act_s_tile, h])
            pypto.set_vec_tile_shapes(64, 64)

            x_tile = pypto.cast(x_tile, pypto.DT_FP32) ## b,s,h
            kv = pypto.matmul(x_tile, wkv, pypto.DT_FP32) ## b*s,2d
            kv = pypto.reshape(kv, [1, s_tile, wkv.shape[1]], valid_shape=[1, act_s_tile, wkv.shape[1]]) ## b,s,2d
            score = pypto.matmul(x_tile, wgate, pypto.DT_FP32)
            score = pypto.reshape(score, [1, s_tile, wgate.shape[1]], valid_shape=[1, act_s_tile, wgate.shape[1]]) ## b,s,2d

            should_compress = (kv_len_int >= ratio)
            should_compute = True
            remainder = kv_len_int % ratio
            cutoff = kv_len_int - remainder
            offset = ratio * overlap

            cutoff_128 = ((kv_len_int // ratio) * ratio) % s_tile
            update_state = cutoff >= ratio

            if pypto.cond(pypto.is_loop_end(s_idx)):
                if cutoff_128 >= ratio:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, cutoff_128-ratio, 0])
                    score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, cutoff_128-ratio, 0])

                    pypto.set_vec_tile_shapes(1, 8, coff*d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(score_state_tile, index, score_view)

                    if remainder > 0: 
                        index = pypto.view(cache_index_2d_tile, [1, ratio], [0, offset], valid_shape=[1, remainder])
                        kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wkv.shape[1]])
                        score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wgate.shape[1]])
                        ape_view = pypto.view(ape, [ratio, coff*d], [0, 0], valid_shape=[remainder, coff*d])

                        pypto.set_vec_tile_shapes(1, 8, coff*d)
                        score_view = pypto.add(score_view, ape_view)
                        kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                        score_state_tile = scatter_update_3d(score_state_tile, index, score_view)

                elif update_state and cutoff_128 == 0 and remainder > 0:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, offset], valid_shape=[1, remainder])
                    kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wkv.shape[1]])
                    score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, cutoff_128, 0], valid_shape=[1, remainder, wgate.shape[1]])
                    ape_view = pypto.view(ape, [ratio, coff*d], [0, 0], valid_shape=[remainder, coff*d])

                    pypto.set_vec_tile_shapes(1, 8, coff*d)
                    score_view = pypto.add(score_view, ape_view)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(score_state_tile, index, score_view)
                    should_compute = False # 跳过

                elif update_state and cutoff_128 == 0 and remainder == 0:
                    index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                    kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0])
                    score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0])

                    pypto.set_vec_tile_shapes(1, 8, coff*d)
                    score_view = pypto.add(score_view, ape)
                    kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                    score_state_tile = scatter_update_3d(score_state_tile, index, score_view)
                if update_state:
                    ## 更新kv_state和score_state
                    pypto.set_vec_tile_shapes(1, 128, d)
                    pypto.assemble(kv_state_tile, [b_idx, 0, 0], kv_state_out)
                    pypto.assemble(score_state_tile, [b_idx, 0, 0], score_state_out)


            if update_state and cutoff_128 == 0 and remainder > 0 and pypto.cond(s_idx == s_loop - 2):
                index = pypto.view(cache_index_2d_tile, [1, ratio], [0, 0])
                kv_view = pypto.view(kv, [1, ratio, wkv.shape[1]], [0, s_tile - ratio, 0])
                score_view = pypto.view(score, [1, ratio, wgate.shape[1]], [0, s_tile - ratio, 0])

                pypto.set_vec_tile_shapes(1, 8, coff*d)
                score_view = pypto.add(score_view, ape)
                kv_state_tile = scatter_update_3d(kv_state_tile, index, kv_view)
                score_state_tile = scatter_update_3d(score_state_tile, index, score_view)

            if should_compute:
                pypto.set_vec_tile_shapes(1, 4, d)
                kv = pypto.reshape(kv, [1, kv.shape[1]//ratio, ratio, coff*d]) ## b,cut,4,2d
                score = pypto.reshape(score, [1, score.shape[1]//ratio, ratio, coff*d]) ## b,cut,4,2d
                pypto.set_vec_tile_shapes(1, 4, 4, d)
                score = pypto.add(score, ape) ## b,cut,4,2d

                if overlap:
                    ## overlap_trans\softmax\mul\sum
                    ## 两个overlap_trans分成不同图的时候，精度才对
                    pypto.set_pass_options(sg_set_scope=1)
                    kv, kv_full_tmp = overlap_trans_dynamic(kv, kv_full) ## b,cut,8,d
                    kv_full[:] = kv_full_tmp
                    pypto.set_pass_options(sg_set_scope=-1)

                    pypto.set_pass_options(sg_set_scope=2)
                    score, score_full_tmp = overlap_trans_dynamic(score, score_full) ## b,cut,8,d
                    score_full[:] = score_full_tmp
                    score = softmax(score, 2) ## b,cut,8,d
                    pypto.set_pass_options(sg_set_scope=-1)
                else:
                    score = softmax(score, 2) ## b,cut,8,d

                ## 不用scope和前面隔开会有kv*score的精度问题
                pypto.set_pass_options(sg_set_scope=3)
                kv = kv * score ## b,cut,8,d
                kv = pypto.sum(kv, 2) ## b,cut,d
                pypto.set_pass_options(sg_set_scope=-1)

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,cut,d

                kv_nope = kv[:, :, :d-rope_head_dim]
                kv_rope = kv[:, :, d-rope_head_dim:]
                sin_tile = pypto.view(sin, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0]) ## b, cut, 64
                cos_tile = pypto.view(cos, kv_rope.shape, [b_idx, s_idx * s_tile // ratio, 0])
                rope3d_tile_config = Rope3dTileConfig(
                    [1, 32, 32],
                    [1, 32, 32, 32]
                )
                kv_rope = interleaved_rope_3d(kv_rope, cos_tile, sin_tile, rope3d_tile_config)
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,cut,d

                if pypto.cond(pypto.is_loop_end(s_idx)):
                    kv = pypto.view(kv, [1, s_tile // ratio, d], [0, 0, 0], valid_shape=[1, cutoff_128 // ratio, d])
                    kv = kv + 0

                pypto.assemble(kv, [b_idx, s_idx * s_tile // ratio, 0], out)
                ## 更新kv_cache和score_cache

@pypto.jit(
    pass_options={}
)
def compressor_prefill_ratio_128(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ratio, kv_len_int_dy, rope_head_dim):
    # pypto.set_debug_options(runtime_debug_mode=1)

    shape_x = x.shape
    dtype = x.dtype
    b = shape_x[0]
    s = shape_x[1]
    h = shape_x[2]
    overlap = (ratio == 4)
    coff = 1 + overlap
    d = wkv.shape[1] // coff
    s_tile = 128
    kv_len_int = kv_len_int_dy.concrete()
    s_loop = (kv_len_int + s_tile -1 ) // s_tile

    for b_idx in pypto.loop(b):
        pypto.set_vec_tile_shapes(1, 128)
        cache_index_2d_tile = pypto.view(cache_index_2d, [1, cache_index_2d.shape[1]], [b_idx, 0])
        pypto.set_vec_tile_shapes(1, 128, 128)
        kv_state_tile = pypto.view(kv_state, [1, kv_state.shape[1], kv_state.shape[2]], [b_idx, 0, 0])
        score_state_tile = pypto.view(score_state, [1, score_state.shape[1], score_state.shape[2]], [b_idx, 0, 0])
        for s_idx in pypto.loop(s_loop):
            act_s_tile = (kv_len_int - s_idx*s_tile).min(s_tile)
            pypto.set_vec_tile_shapes(1, 64, 64)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
            x_tile = pypto.view(x, [1, s_tile, h], [b_idx, s_idx*s_tile, 0], valid_shape=[b, act_s_tile, h])

            x_tile = pypto.reshape(x_tile, [s_tile, h], valid_shape=[act_s_tile, h])
            pypto.set_vec_tile_shapes(64, 64)

            x_tile = pypto.cast(x_tile, pypto.DT_FP32) ## b,s,h
            kv = pypto.matmul(x_tile, wkv, pypto.DT_FP32) ## b*s,2d
            kv = pypto.reshape(kv, [1, s_tile, wkv.shape[1]], valid_shape=[1, act_s_tile, wkv.shape[1]]) ## b,s,2d
            score = pypto.matmul(x_tile, wgate, pypto.DT_FP32)
            score = pypto.reshape(score, [1, s_tile, wgate.shape[1]], valid_shape=[1, act_s_tile, wgate.shape[1]]) ## b,s,2d

            ## Cut
            remainder = kv_len_int % ratio
            if remainder > 0 and pypto.cond(pypto.is_loop_end(s_idx)):
                pypto.set_vec_tile_shapes(1, 8, coff*d)
                index = pypto.view(cache_index_2d_tile, [1, s_tile], [0, 0], valid_shape=[1, act_s_tile])
                ape_view = pypto.view(ape, [s_tile, d], [0, 0], valid_shape=[act_s_tile, d])
                score = pypto.add(score, ape_view)
                kv_state_tile = scatter_update_3d(kv_state_tile, index, kv)
                score_state_tile = scatter_update_3d(score_state_tile, index, score)

                ## 更新kv_state和score_state
                pypto.set_vec_tile_shapes(1, 128, d)
                pypto.assemble(kv_state_tile, [b_idx, 0, 0], kv_state_out)
                pypto.assemble(score_state_tile, [b_idx, 0, 0], score_state_out)
            else:
                pypto.set_vec_tile_shapes(1, 64, d)
                kv = pypto.reshape(kv, [1, 1, ratio, coff*d]) ## b,cut,4,2d
                score = pypto.reshape(score, [1, 1, ratio, coff*d]) ## b,cut,4,2d

                pypto.set_vec_tile_shapes(1, 4, 4, d)
                score = pypto.add(score, ape) ## b,cut,4,2d
                score = softmax(score, 2) ## b,cut,8,d

                ## 不用scope和前面隔开会有kv*score的精度问题
                pypto.set_pass_options(sg_set_scope=3)
                kv = kv * score ## b,cut,8,d
                kv = pypto.sum(kv, 2) ## b,cut,d
                pypto.set_pass_options(sg_set_scope=-1)

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,cut,d

                kv_nope = kv[:, :, :d-rope_head_dim]
                kv_rope = kv[:, :, d-rope_head_dim:]
                sin_tile = pypto.view(sin, kv_rope.shape, [b_idx, s_idx, 0]) ## b, cut, 64
                cos_tile = pypto.view(cos, kv_rope.shape, [b_idx, s_idx, 0])
                rope3d_tile_config = Rope3dTileConfig(
                    [1, 32, 32],
                    [1, 32, 32, 32]
                )
                kv_rope = interleaved_rope_3d(kv_rope, cos_tile, sin_tile, rope3d_tile_config)
                pypto.set_vec_tile_shapes(1, 8, d)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,cut,d

                pypto.assemble(kv, [b_idx, s_idx, 0], out)

                ## 更新kv_cache和score_cache


def overlap_transform(tensor: torch.Tensor, value):
    # tensor: [b,s,r,2d]
    b, s, ratio, d = tensor.size()
    d = d//2
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor

def RMSNorm(x, eps, weight):
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)

def apply_rotary_pos_emb_v2(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, mode="half"):
    input_dtype = x.dtype
    if input_dtype != torch.float32:
        x = x.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
    if mode == "half":
        b, s, d = x.shape
        x = x.reshape(b, s, d // 2, 2).permute(0, 1, 3, 2).reshape(b, s, d)

        x1, x2 = x.chunk(2, dim=-1)
        p = torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        p = torch.stack((-x2, x1), dim=-1).flatten(-2)
    x_embed = (x * cos) + (p * sin)
    x_embed = x_embed.to(input_dtype)

    return x_embed

def golden_compress(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard, ratio, start_pos, rope_head_dim, rotate, kv_len_int, eps=1e-6):
    bsz, _, _ = x.size()
    overlap = (ratio == 4)
    d = wkv.size(1) // (1 + overlap)
    dtype = x.dtype
    x = x.float() ## b,s,h
    kv = torch.matmul(x, wkv) ## b,s,2d
    score = torch.matmul(x, wgate) ## b,s,2d
    if start_pos == 0:
        should_compress = kv_len_int >= ratio
        remainder = kv_len_int % ratio
        cutoff = kv_len_int - remainder

        ##跳跃采样未适配
        sin = sin[:,:max(1, cutoff//ratio)] ## b, cut, 64
        cos = cos[:,:max(1, cutoff//ratio)] ## b, cut, 64

        offset = ratio if overlap else 0
        if overlap and cutoff >= ratio:
            kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff] ## b,4,2d
            score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + ape ## b,4,2d
        if remainder > 0:
            kv, kv_state[:bsz, offset : offset+remainder] = kv[:, :kv_len_int].split([cutoff, remainder], dim=1) ## b,4*cut,2d
            score_state[:bsz, offset : offset+remainder] = score[:, cutoff:cutoff + remainder] + ape[:remainder]
            score = score[:, :cutoff] ## b,4*cut,2d
        kv = kv.unflatten(1, (-1, ratio)) ## b,cut,4,2d
        score = score.unflatten(1, (-1, ratio)) + ape ## b,cut,4,2d
        if overlap:
            kv = overlap_transform(kv, 0) ## b,cut,8,d
            score = overlap_transform(score, float("-inf")) ## b,cut,8,d
        kv = (kv * score.softmax(dim=2)).sum(dim=2) ## b,cut,d
    else:
        should_compress = (start_pos + 1) % ratio == 0
        score += ape[start_pos % ratio]
        if overlap:
            kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
            score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv_state_tmp = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1) ## b,8,d
                score_state_tmp = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1) ## b,8,d
                kv = (kv_state_tmp * score_state_tmp.softmax(dim=1)).sum(dim=1, keepdim=True) ## b,1,d
                kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
                score_state[:bsz, :ratio] = score_state[:bsz, ratio:]
        else:
            kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            score_state[:bsz, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True) ## b,1,d

    if not should_compress:
        return
    kv = RMSNorm(kv.to(dtype), eps, weight) ## b,cut,d
    kv_rope = kv[..., -rope_head_dim:].clone()
    kv = kv.clone()
    kv[..., -rope_head_dim:] = apply_rotary_pos_emb_v2(kv_rope, sin, cos, "interleave")
    if rotate:
        kv = torch.matmul(kv, hadamard) ## b,cut,d
    # if start_pos == 0:
    #       kv_cache[:bsz, :kv_len_int // ratio] = kv
    # else:
    #       kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
    return kv

def gen_inputs(bsz, seq, h, d, rope_head_dim, ratio, device):
    torch.manual_seed(42)
    overlap = (ratio == 4)
    coff = 1 + overlap
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device)
    sin = torch.rand((bsz, max(1, seq//ratio), rope_head_dim), dtype=torch.bfloat16, device=device)
    cos = torch.rand((bsz, max(1, seq//ratio), rope_head_dim), dtype=torch.bfloat16, device=device)
    wkv = torch.rand((h, coff*d), dtype=torch.float32, device=device)
    wgate = torch.rand((h, coff*d), dtype=torch.float32, device=device)
    ape = torch.rand((ratio, coff*d), dtype=torch.float32, device=device)
    weight = torch.ones(d, dtype=torch.float32, device=device)
    kv_state = torch.zeros((bsz, coff*ratio, coff*d), dtype=torch.float32, device=device)
    score_state = torch.full((bsz, coff*ratio, coff*d), float("-inf"), dtype=torch.float32, device=device)
    hadamard = torch.rand((d, d), dtype=torch.bfloat16, device=device)*(d ** -0.5)
    return x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard


def test_prefill():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    kv_len_int_ori = [255]
    ratio_4 = [4] * len(kv_len_int_ori)
    ratio_128 = [128] * len(kv_len_int_ori)
    
    rotate = [False, True] * len(ratio_4) + [False] * len(ratio_128)
    ratio = ratio_4 * 2 + ratio_128
    kv_len_int = kv_len_int_ori * (len(rotate) // 1)

    for kv_len, ra, ro in zip(kv_len_int, ratio, rotate):
        print(f"test_compressor_prefill (kv_len_int: {kv_len}, ratio: {ra}, rotate: {ro}) begin!")
        bsz = 1
        h = 4096
        d = 512
        rope_head_dim = 64
        overlap = (ra == 4)
        coff = 1 + overlap
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = gen_inputs(bsz, kv_len, h, d, rope_head_dim, ra, device)
        cache_index_2d = torch.arange(bsz*coff*ra, dtype=torch.int32, device=device).reshape(bsz, coff*ra)

        out = torch.zeros((bsz, max(1, kv_len // ra), d), dtype=torch.bfloat16, device=device) 
        kv_state_out = torch.zeros((bsz, coff*ra, coff*d), dtype=torch.float32, device=device)
        score_state_out = torch.full((bsz, coff*ra, coff*d), float("-inf"), dtype=torch.float32, device=device)

        compressor(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ra, 0, kv_len, rope_head_dim, ro, hadamard=hadamard)
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = gen_inputs(bsz, kv_len, h, d, rope_head_dim, ra, device)
        kv = golden_compress(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard, ra, 0, rope_head_dim, ro, kv_len)
        
        assert_allclose(kv_state_out.cpu().float().numpy(), kv_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(score_state_out.cpu().float().numpy(), score_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
        if kv is not None:
            assert_allclose(out.cpu().float().numpy(), kv.cpu().float().numpy(), rtol=0.0078125, atol=1e-4)

        print(f"test_compressor_prefill passed!")


if __name__ == "__main__":
    test_prefill()