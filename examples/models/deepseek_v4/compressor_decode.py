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
    start_pos_dy = pypto.SymbolicScalar(start_pos)

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    if start_pos != 0 and rotate and ratio == 4:
        hadamard = kwargs.get('hadamard')
        hadamard = pypto.from_torch(hadamard, dynamic_axis=[])
        compressor_decode_ratio_4_rotate(*pto_inputs, hadamard, *pto_outputs, ratio, start_pos_dy, rope_head_dim)
    elif start_pos != 0 and not rotate and ratio == 4:
        compressor_decode_ratio_4(*pto_inputs, *pto_outputs, ratio, start_pos_dy, rope_head_dim)
    elif start_pos != 0 and not rotate and ratio == 128:
        compressor_decode_ratio_128(*pto_inputs, *pto_outputs, ratio, start_pos_dy, rope_head_dim)

def overlap_trans(x: pypto.Tensor, value):
    b, s, r, d = x.shape
    d = d // 2
    x1 = pypto.view(x, [b, s, r, d], [0, 0, 0, d])
    x2 = pypto.view(x, [b, s - 1, r, d], [0, 0, 0, 0])
    y = pypto.full([b, 1, r, d], value, x.dtype)
    z = pypto.concat([y, x2], 1)
    z = pypto.concat([z, x1], 2)
    return z

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

def concat_index_3d(input, src, start, end):
    if start > 0:
        if end < input.shape[1]:
            return pypto.concat([input[:,:start,:], src, input[:,end:,:]], dim=1)+0.0
        else:
            return pypto.concat([input[:,:start,:], src], dim=1)+0.0
    if start == 0:
        return pypto.concat([src, input[:,end:,:]], dim=1)+0.0

@pypto.jit(
    pass_options={}
)
def compressor_decode_ratio_4(x, kv_state_t, score_state_t, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ratio, start_pos_dy, rope_head_dim):
    # pypto.set_debug_options(runtime_debug_mode=1)

    dtype = x.dtype
    bsz, s, h = x.shape
    overlap = (ratio == 4)
    coff = 1 + overlap
    start_pos = start_pos_dy.concrete()
    should_compress = ((start_pos + 1) % ratio == 0)
    pos = start_pos % ratio
    d = wkv.shape[1] // coff

    kv_t = pypto.Tensor([bsz, s, coff*d], pypto.DT_FP32)
    score_t = pypto.Tensor([bsz, s, coff*d], pypto.DT_FP32)

    b = 16
    b_loop = (bsz + b - 1) // b
    for b_idx in pypto.loop(b_loop, name="LOOP_COMP_1", idx_name="b_idx"):
        b_valid = (bsz-b_idx*b).min(b)

        pypto.set_vec_tile_shapes(2, 1, h)
        x_view = pypto.view(x, [b, s, h], [b_idx*b, 0, 0], valid_shape = [b_valid, s, h])
        x_cast = pypto.cast(x_view, pypto.DT_FP32) ## b,s,h
        x_cast = pypto.reshape(x_cast, [b * s, h], inplace=True)

        ## Matmul
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
        pypto.set_vec_tile_shapes(2, 1, 512)
        kv_t = pypto.matmul(x_cast, wkv, pypto.DT_FP32) ## b*s,2d
        kv_t = pypto.reshape(kv_t, [b, s, wkv.shape[1]]) + 0.0 ## b,s,2d
        score_t = pypto.matmul(x_cast, wgate, pypto.DT_FP32)
        score_t = pypto.reshape(score_t, [b, s, wgate.shape[1]]) + 0.0 ## b,s,2d

        c = 1
        c_loop = (b_valid + c - 1) // c
        for c_idx in pypto.loop(c_loop, name="LOOP_COMP_2", idx_name="c_idx"):
            c_valid = (b_valid-c_idx*c).min(c)
            pypto.set_vec_tile_shapes(1, 1, 512)
            kv = pypto.view(kv_t, [c, s, coff*d], [c_idx*c, 0, 0], valid_shape = [c_valid, s, coff*d])
            score = pypto.view(score_t, [c, s, coff*d], [c_idx*c, 0, 0], valid_shape = [c_valid, s, coff*d])
            kv_state = pypto.view(kv_state_t, [c, coff*ratio, coff*d], [b_idx*b+c_idx*c, 0, 0], valid_shape = [c_valid, coff*ratio, coff*d])
            score_state = pypto.view(score_state_t, [c, coff*ratio, coff*d], [b_idx*b+c_idx*c, 0, 0], valid_shape = [c_valid, coff*ratio, coff*d])

            pypto.set_vec_tile_shapes(1, 512)
            ape_view = ape[pos, :]
            pypto.set_vec_tile_shapes(1, 1, 512)
            score = pypto.add(score, ape_view) ## b,1,2d
            if overlap:
                pypto.set_vec_tile_shapes(c, 8)
                index = cache_index_2d[:c, ratio+pos : ratio+pos+1]
                kv_state = scatter_update_3d(kv_state, index, kv)
                score_state = scatter_update_3d(score_state, index, score)
                if should_compress:
                    pypto.set_vec_tile_shapes(c, 4, 1024)
                    kv_state_tmp = pypto.concat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], 1) ## b,8,d
                    score_state_tmp = pypto.concat([score_state[:, :ratio, :d], score_state[:, ratio:, d:]], 1) ## b,8,d
                    kv = kv_state_tmp * softmax(score_state_tmp, 1) ## b,8,d
                    kv = pypto.sum(kv, 1, keepdim=True) ## b,1,d
                    # pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)
                    kv_state_view = kv_state[:, ratio:, :]
                    score_state_view = score_state[:, ratio:, :]
                    kv_state = pypto.concat([kv_state_view, kv_state_view], dim=1)
                    score_state = pypto.concat([score_state_view, score_state_view], dim=1)
            else:
                pypto.set_vec_tile_shapes(c, 8)
                index = cache_index_2d[:c, pos : pos+1]
                kv_state = scatter_update_3d(kv_state, index, kv)
                score_state = scatter_update_3d(score_state, index, score)
                if should_compress:
                    pypto.set_vec_tile_shapes(1, 256, 64)
                    kv = kv_state * softmax(score_state, 1)
                    kv = pypto.sum(kv, 1, keepdim=True) ## b,1,d

            ## 更新kv_state和score_state
            pypto.assemble(kv_state, [b_idx*b+c_idx*c, 0, 0], kv_state_out)
            pypto.assemble(score_state, [b_idx*b+c_idx*c, 0, 0], score_state_out)

            if should_compress:
                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, 512)
                kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,1,d
                # pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)

                kv_nope = kv[:, :, :d-rope_head_dim]
                kv_rope = kv[:, :, d-rope_head_dim:]
                sin = pypto.view(sin, kv_rope.shape, [b_idx*b+c_idx*c, 0, 0]) ## b, 1, 64
                cos = pypto.view(cos, kv_rope.shape, [b_idx*b+c_idx*c, 0, 0]) ## b, 1, 64
                rope3d_tile_config = Rope3dTileConfig(
                    [1, 64, 64],
                    [1, 64, 128, 128]
                )
                kv_rope = interleaved_rope_3d(kv_rope, cos, sin, rope3d_tile_config)
                pypto.set_vec_tile_shapes(1, 8, 512)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,1,d
                pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)

        ## 更新kv_cache和score_cache


@pypto.jit(
    pass_options={}
)
def compressor_decode_ratio_4_rotate(x, kv_state_t, score_state_t, cache_index_2d, sin, cos, wkv, wgate, ape, weight, hadamard, out, kv_state_out, score_state_out, ratio, start_pos_dy, rope_head_dim):
    # pypto.set_debug_options(runtime_debug_mode=1)

    dtype = x.dtype
    bsz, s, h = x.shape
    overlap = (ratio == 4)
    coff = 1 + overlap
    start_pos = start_pos_dy.concrete()
    should_compress = ((start_pos + 1) % ratio == 0)
    pos = start_pos % ratio
    d = wkv.shape[1] // coff

    kv_t = pypto.Tensor([bsz, s, coff*d], pypto.DT_FP32)
    score_t = pypto.Tensor([bsz, s, coff*d], pypto.DT_FP32)

    b = 16
    b_loop = (bsz + b - 1) // b
    for b_idx in pypto.loop(b_loop, name="LOOP_COMP_1", idx_name="b_idx"):
        b_valid = (bsz-b_idx*b).min(b)

        pypto.set_vec_tile_shapes(2, 1, h)
        x_view = pypto.view(x, [b, s, h], [b_idx*b, 0, 0], valid_shape = [b_valid, s, h])
        x_cast = pypto.cast(x_view, pypto.DT_FP32) ## b,s,h
        x_cast = pypto.reshape(x_cast, [b * s, h], inplace=True)

        ## Matmul
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
        pypto.set_vec_tile_shapes(2, 1, 512)
        kv_t = pypto.matmul(x_cast, wkv, pypto.DT_FP32) ## b*s,2d
        kv_t = pypto.reshape(kv_t, [b, s, wkv.shape[1]]) + 0.0 ## b,s,2d
        score_t = pypto.matmul(x_cast, wgate, pypto.DT_FP32)
        score_t = pypto.reshape(score_t, [b, s, wgate.shape[1]]) + 0.0 ## b,s,2d

        out_t = pypto.Tensor([b, s, d], pypto.DT_BF16)

        c = 1
        c_loop = (b_valid + c - 1) // c
        for c_idx in pypto.loop(c_loop, name="LOOP_COMP_2", idx_name="c_idx"):
            c_valid = (b_valid-c_idx*c).min(c)
            pypto.set_vec_tile_shapes(1, 1, 512)
            kv = pypto.view(kv_t, [c, s, coff*d], [c_idx*c, 0, 0], valid_shape = [c_valid, s, coff*d])
            score = pypto.view(score_t, [c, s, coff*d], [c_idx*c, 0, 0], valid_shape = [c_valid, s, coff*d])
            kv_state = pypto.view(kv_state_t, [c, coff*ratio, coff*d], [b_idx*b+c_idx*c, 0, 0], valid_shape = [c_valid, coff*ratio, coff*d])
            score_state = pypto.view(score_state_t, [c, coff*ratio, coff*d], [b_idx*b+c_idx*c, 0, 0], valid_shape = [c_valid, coff*ratio, coff*d])

            pypto.set_vec_tile_shapes(1, 512)
            ape_view = ape[pos, :]
            pypto.set_vec_tile_shapes(1, 1, 512)
            score = pypto.add(score, ape_view) ## b,1,2d
            if overlap:
                pypto.set_vec_tile_shapes(c, 8)
                index = cache_index_2d[:c, ratio+pos : ratio+pos+1]
                kv_state = scatter_update_3d(kv_state, index, kv)
                score_state = scatter_update_3d(score_state, index, score)
                if should_compress:
                    pypto.set_vec_tile_shapes(c, 4, 1024)
                    kv_state_tmp = pypto.concat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], 1) ## b,8,d
                    score_state_tmp = pypto.concat([score_state[:, :ratio, :d], score_state[:, ratio:, d:]], 1) ## b,8,d
                    kv = kv_state_tmp * softmax(score_state_tmp, 1) ## b,8,d
                    kv = pypto.sum(kv, 1, keepdim=True) ## b,1,d
                    # pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)
                    kv_state_view = kv_state[:, ratio:, :]
                    score_state_view = score_state[:, ratio:, :]
                    kv_state = pypto.concat([kv_state_view, kv_state_view], dim=1)
                    score_state = pypto.concat([score_state_view, score_state_view], dim=1)
            else:
                pypto.set_vec_tile_shapes(c, 8)
                index = cache_index_2d[:c, pos : pos+1]
                kv_state = scatter_update_3d(kv_state, index, kv)
                score_state = scatter_update_3d(score_state, index, score)
                if should_compress:
                    pypto.set_vec_tile_shapes(1, 256, 64)
                    kv = kv_state * softmax(score_state, 1)
                    kv = pypto.sum(kv, 1, keepdim=True) ## b,1,d

            ## 更新kv_state和score_state
            pypto.assemble(kv_state, [b_idx*b+c_idx*c, 0, 0], kv_state_out)
            pypto.assemble(score_state, [b_idx*b+c_idx*c, 0, 0], score_state_out)

            if should_compress:
                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, 512)
                kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,1,d
                # pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)

                kv_nope = kv[:, :, :d-rope_head_dim]
                kv_rope = kv[:, :, d-rope_head_dim:]
                sin = pypto.view(sin, kv_rope.shape, [b_idx*b+c_idx*c, 0, 0]) ## b, 1, 64
                cos = pypto.view(cos, kv_rope.shape, [b_idx*b+c_idx*c, 0, 0]) ## b, 1, 64
                rope3d_tile_config = Rope3dTileConfig(
                    [1, 64, 64],
                    [1, 64, 128, 128]
                )
                kv_rope = interleaved_rope_3d(kv_rope, cos, sin, rope3d_tile_config)
                pypto.set_vec_tile_shapes(1, 8, 512)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,1,d

                ## hadamard
                pypto.assemble(kv, [c_idx*c, 0, 0], out_t)
        if should_compress:
            out_t = pypto.reshape(out_t, [b*s, d])
            out_t = pypto.matmul(out_t, hadamard, pypto.DT_BF16) ## b,d
            out_t = pypto.reshape(out_t, [b, s, d], valid_shape=[b_valid, s, d]) ## b,d
            pypto.assemble(out_t, [b_idx*b, 0, 0], out)

        ## 更新kv_cache和score_cache

@pypto.jit(
    pass_options={}
)
def compressor_decode_ratio_128(x, kv_state_t, score_state_t, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ratio, start_pos_dy, rope_head_dim):
    # pypto.set_debug_options(runtime_debug_mode=1)

    dtype = x.dtype
    bsz, s, h = x.shape
    overlap = (ratio == 4)
    coff = 1 + overlap
    start_pos = start_pos_dy.concrete()
    should_compress = ((start_pos + 1) % ratio == 0)
    pos = start_pos % ratio
    d = wkv.shape[1] // coff

    kv_t = pypto.Tensor([bsz, s, coff*d], pypto.DT_FP32)
    score_t = pypto.Tensor([bsz, s, coff*d], pypto.DT_FP32)
    
    b = 16
    b_loop = (bsz + b - 1) // b
    for b_idx in pypto.loop(b_loop, name="LOOP_COMP_1", idx_name="b_idx"):
        b_valid = (bsz-b_idx*b).min(b)

        pypto.set_vec_tile_shapes(2, 1, h)
        x_view = pypto.view(x, [b, s, h], [b_idx*b, 0, 0], valid_shape = [b_valid, s, h])
        x_cast = pypto.cast(x_view, pypto.DT_FP32) ## b,s,h
        x_cast = pypto.reshape(x_cast, [b * s, h], inplace=True)
        
        ## Matmul
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
        pypto.set_vec_tile_shapes(2, 1, 512)
        kv_t = pypto.matmul(x_cast, wkv, pypto.DT_FP32) ## b*s,2d
        kv_t = pypto.reshape(kv_t, [b, s, wkv.shape[1]]) + 0.0 ## b,s,2d
        score_t = pypto.matmul(x_cast, wgate, pypto.DT_FP32)
        score_t = pypto.reshape(score_t, [b, s, wgate.shape[1]]) + 0.0 ## b,s,2d

        c = 1
        c_loop = (b_valid + c - 1) // c
        for c_idx in pypto.loop(c_loop, name="LOOP_COMP_2", idx_name="c_idx"):
            c_valid = (b_valid-c_idx*c).min(c)
            pypto.set_vec_tile_shapes(1, 1, 512)
            kv = pypto.view(kv_t, [c, s, coff*d], [c_idx*c, 0, 0], valid_shape = [c_valid, s, coff*d])
            score = pypto.view(score_t, [c, s, coff*d], [c_idx*c, 0, 0], valid_shape = [c_valid, s, coff*d])
            kv_state = pypto.view(kv_state_t, [c, coff*ratio, coff*d], [b_idx*b+c_idx*c, 0, 0], valid_shape = [c_valid, coff*ratio, coff*d])
            score_state = pypto.view(score_state_t, [c, coff*ratio, coff*d], [b_idx*b+c_idx*c, 0, 0], valid_shape = [c_valid, coff*ratio, coff*d])

            pypto.set_vec_tile_shapes(1, 512)
            ape_view = ape[pos, :]
            pypto.set_vec_tile_shapes(1, 1, 512)
            score = pypto.add(score, ape_view) ## b,1,2d
            
            pypto.set_vec_tile_shapes(c, 8)
            index = cache_index_2d[:c, pos : pos+1]
            kv_state = scatter_update_3d(kv_state, index, kv)
            score_state = scatter_update_3d(score_state, index, score)
            if should_compress:
                pypto.set_vec_tile_shapes(1, 256, 64)
                kv = kv_state * softmax(score_state, 1)
                kv = pypto.sum(kv, 1, keepdim=True) ## b,1,d
        
            ## 更新kv_state和score_state
            pypto.assemble(kv_state, [b_idx*b+c_idx*c, 0, 0], kv_state_out)
            pypto.assemble(score_state, [b_idx*b+c_idx*c, 0, 0], score_state_out)
            
            if should_compress:

                # RMSNorm\RoPE
                pypto.set_vec_tile_shapes(1, 8, 512)
                kv = rms_norm(pypto.cast(kv, dtype), weight) ## b,1,d
                # pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)
                
                kv_nope = kv[:, :, :d-rope_head_dim]
                kv_rope = kv[:, :, d-rope_head_dim:]
                sin = pypto.view(sin, kv_rope.shape, [b_idx*b+c_idx*c, 0, 0]) ## b, 1, 64
                cos = pypto.view(cos, kv_rope.shape, [b_idx*b+c_idx*c, 0, 0]) ## b, 1, 64
                rope3d_tile_config = Rope3dTileConfig(
                    [1, 32, 32],
                    [1, 32, 128, 128]
                )
                kv_rope = interleaved_rope_3d(kv_rope, cos, sin, rope3d_tile_config)
                pypto.set_vec_tile_shapes(1, 8, 512)
                kv = pypto.concat([kv_nope, kv_rope], dim=-1) ## b,1,d 
                pypto.assemble(kv, [b_idx*b+c_idx*c, 0, 0], out)
                
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
    kv = RMSNorm(kv.to(dtype), eps, weight) ## b,1,d
    kv_rope = kv[..., -rope_head_dim:].clone()
    kv_new = kv.clone()
    kv_new[..., -rope_head_dim:] = apply_rotary_pos_emb_v2(kv_rope, sin, cos, "interleave")
    if rotate:
        kv = torch.matmul(kv_new, hadamard) ## b,1,d
    else:
        kv = kv_new ## b,1,d
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


def test_decode():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)

    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    start_pos_ori = [255]
    ratio_4 = [4] * len(start_pos_ori)
    ratio_128 = [128] * len(start_pos_ori)
    
    rotate = [False, True] * len(ratio_4) + [False] * len(ratio_128)
    ratio = ratio_4 * 2 + ratio_128
    start_pos = start_pos_ori * (len(rotate) // 1)

    for st, ra, ro in zip(start_pos, ratio, rotate):
        print(f"test_compressor_decode (start_pos: {st}, ratio: {ra}, rotate: {ro}) begin!")
        bsz = 3
        seq = 1
        kv_len_int = seq
        h = 4096
        d = 512
        rope_head_dim = 64
        overlap = (ra == 4)
        coff = 1 + overlap
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = gen_inputs(bsz, seq, h, d, rope_head_dim, ra, device)
        cache_index_2d = torch.arange(bsz*coff*ra, dtype=torch.int32, device=device).reshape(bsz, coff*ra)

        out = torch.zeros((bsz, 1, d), dtype=torch.bfloat16, device=device) 
        kv_state_out = torch.zeros((bsz, coff*ra, coff*d), dtype=torch.float32, device=device)
        score_state_out = torch.full((bsz, coff*ra, coff*d), float("-inf"), dtype=torch.float32, device=device)

        compressor(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, out, kv_state_out, score_state_out, ra, st, kv_len_int, rope_head_dim, ro, hadamard=hadamard)
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = gen_inputs(bsz, seq, h, d, rope_head_dim, ra, device)
        kv = golden_compress(x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard, ra, st, rope_head_dim, ro, kv_len_int)
        
        assert_allclose(kv_state_out.cpu().float().numpy(), kv_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(score_state_out.cpu().float().numpy(), score_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
        if kv is not None:
            assert_allclose(out.cpu().float().numpy(), kv.cpu().float().numpy(), rtol=0.0078125, atol=1e-4)

        print(f"test_compressor_decode passed!")


if __name__ == "__main__":
    test_decode()