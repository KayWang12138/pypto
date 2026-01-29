#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import pypto
import torch
from torch._dynamo import allow_in_graph
from compressor_decode import (
    compressor_decode_ratio_4_rotate,
    compressor_decode_ratio_4,
    compressor_decode_ratio_128,
)


def check_args(
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
    start_pos,
    kv_len_int,
    rope_head_dim,
    rotate,
    hadamard,
):
    overlap = ratio == 4
    coff = 1 + overlap
    d = weight.shape[0]
    assert ratio == 4 or ratio == 128, f"ratio is {ratio}, expected 4 or 128"
    assert rope_head_dim == 64, f"rope_head_dim is {rope_head_dim}, expected 64"
    assert start_pos >= 0, (
        f"start_pos is {start_pos}, expected greater than or equal to 0"
    )

    assert kv_len_int >= start_pos, (
        f"kv_len_int is {kv_len_int}, start_pos is {start_pos}, expected kv_len_int greater than or equal to start_pos"
    )
    cmp_seq = max(1, (kv_len_int - start_pos) // ratio)

    assert weight.dim() == 1 and ((d == 128 and rotate) or (d == 512 and not rotate)), (
        f"weight dim num is {weight.dim()}, weight axis1 is {d}, \
        expected 1, (d = 512 and rotate = False) or (d = 128 and rotate = True)"
    )

    assert x.dim() == 3 and x.size(1) <= kv_len_int and x.size(2) == 4096, (
        f"x dim num is {x.dim()}, x axis1 is {x.size(1)}, x axis2 is {x.size(2)}, \
        expected 3, less than or equal to {kv_len_int}, 4096"
    )

    assert (
        kv_state.dim() == 3
        and kv_state.size(1) == coff * ratio
        and kv_state.size(2) == coff * d
    ), (
        f"kv_state dim num is {kv_state.dim()}, kv_state axis1 is {kv_state.size(1)}, \
        kv_state axis2 is {kv_state.size(2)}, expected 3, {coff * ratio}, {coff * d}"
    )

    assert (
        score_state.dim() == 3
        and score_state.size(1) == coff * ratio
        and score_state.size(2) == coff * d
    ), (
        f"score_state dim num is {score_state.dim()}, score_state axis1 is {score_state.size(1)}, \
        score_state axis2 is {score_state.size(2)}, expected 3, {coff * ratio}, {coff * d}"
    )

    assert cache_index_2d.dim() == 2 and cache_index_2d.size(1) == coff * ratio, (
        f"cache_index_2d dim num is {cache_index_2d.dim()}, cache_index_2d axis1 is {cache_index_2d.size(1)}, \
        expected 2, {coff * ratio}, {coff * ratio}"
    )

    assert sin.dim() == 3 and sin.size(1) == cmp_seq and sin.size(2) == rope_head_dim, (
        f"sin dim num is {sin.dim()}, sin axis1 is {sin.size(1)}, sin axis2 is {sin.size(2)}, \
        expected 3, {cmp_seq}, {rope_head_dim}"
    )

    assert cos.dim() == 3 and cos.size(1) == cmp_seq and cos.size(2) == rope_head_dim, (
        f"cos dim num is {cos.dim()}, cos axis1 is {cos.size(1)}, cos axis2 is {cos.size(2)}, \
        expected 3, {cmp_seq}, {rope_head_dim}"
    )

    assert wkv.dim() == 2 and wkv.size(0) == 4096 and wkv.size(1) == coff * d, (
        f"wkv dim num is {wkv.dim()}, wkv axis0 is {wkv.size(0)}, wkv axis1 is {wkv.size(1)}, \
        expected 2, 4096, {coff * d}"
    )

    assert wgate.dim() == 2 and wgate.size(0) == 4096 and wgate.size(1) == coff * d, (
        f"wgate dim num is {wgate.dim()}, wgate axis0 is {wgate.size(0)}, wgate axis1 is {wgate.size(1)}, \
        expected 2, 4096, {coff * d}"
    )

    assert ape.dim() == 2 and ape.size(0) == ratio and wgate.size(1) == coff * d, (
        f"ape dim num is {ape.dim()}, ape axis0 is {ape.size(0)}, ape axis1 is {ape.size(1)}, \
        expected 2, {ratio}, {coff * d}"
    )

    assert out.dim() == 3 and out.size(1) == cmp_seq and out.size(2) == d, (
        f"out dim num is {out.dim()}, out axis1 is {out.size(1)}, out axis2 is {out.size(2)}, \
        expected 3, {cmp_seq}, {d}"
    )

    assert (
        kv_state_out.dim() == 3
        and kv_state_out.size(1) == coff * ratio
        and kv_state_out.size(2) == coff * d
    ), (
        f"kv_state_out dim num is {kv_state_out.dim()}, kv_state_out axis1 is {kv_state_out.size(1)}, \
        kv_state_out axis2 is {kv_state_out.size(2)}, expected 3, {coff * ratio}, {coff * d}"
    )

    assert (
        score_state_out.dim() == 3
        and score_state_out.size(1) == coff * ratio
        and score_state_out.size(2) == coff * d
    ), (
        f"score_state_out dim num is {score_state_out.dim()}, score_state_out axis1 is {score_state_out.size(1)}, \
        score_state_out axis2 is {score_state_out.size(2)}, expected 3, {coff * ratio}, {coff * d}"
    )

    assert hadamard.dim() == 2 and hadamard.size(0) == d and hadamard.size(1) == d, (
        f"hadamard dim num is {hadamard.dim()}, hadamard axis0 is {hadamard.size(0)}, \
        hadamard axis1 is {hadamard.size(1)}, expected 2, {d}, {d}"
    )

    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert cos.dtype == torch.bfloat16, (
        f"cos.dtype is {cos.dtype}, expected torch.bfloat16"
    )
    assert sin.dtype == torch.bfloat16, (
        f"sin.dtype is {sin.dtype}, expected torch.bfloat16"
    )
    assert hadamard.dtype == torch.bfloat16, (
        f"hadamard.dtype is {hadamard.dtype}, expected torch.bfloat16"
    )
    assert out.dtype == torch.bfloat16, (
        f"out.dtype is {out.dtype}, expected torch.bfloat16"
    )

    assert kv_state.dtype == torch.float32, (
        f"kv_state.dtype is {kv_state.dtype}, expected torch.float32"
    )
    assert score_state.dtype == torch.float32, (
        f"score_state.dtype is {score_state.dtype}, expected torch.float32"
    )
    assert cache_index_2d.dtype == torch.int32, (
        f"cache_index_2d.dtype is {cache_index_2d.dtype}, expected torch.int32"
    )
    assert wkv.dtype == torch.float32, (
        f"wkv.dtype is {wkv.dtype}, expected torch.float32"
    )
    assert wgate.dtype == torch.float32, (
        f"wgate.dtype is {wgate.dtype}, expected torch.float32"
    )
    assert ape.dtype == torch.float32, (
        f"ape.dtype is {ape.dtype}, expected torch.float32"
    )
    assert kv_state_out.dtype == torch.float32, (
        f"kv_state_out.dtype is {kv_state_out.dtype}, expected torch.float32"
    )
    assert score_state_out.dtype == torch.float32, (
        f"score_state_out.dtype is {score_state_out.dtype}, expected torch.float32"
    )


@allow_in_graph
def npu_compressor_decode(
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
    start_pos,
    kv_len_int,
    rope_head_dim,
    rotate,
    **kwargs,
):
    check_args(
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
        start_pos,
        kv_len_int,
        rope_head_dim,
        rotate,
        kwargs["hadamard"]
    )
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
        weight: [],
    }
    outputs = {out: [0], kv_state_out: [0], score_state_out: [0]}
    start_pos_dy = pypto.SymbolicScalar(start_pos)

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    if start_pos != 0 and rotate and ratio == 4:
        hadamard = kwargs.get("hadamard")
        hadamard = pypto.from_torch(hadamard, dynamic_axis=[])
        compressor_decode_ratio_4_rotate(
            *pto_inputs, hadamard, *pto_outputs, ratio, start_pos_dy, rope_head_dim
        )
    elif start_pos != 0 and not rotate and ratio == 4:
        compressor_decode_ratio_4(
            *pto_inputs, *pto_outputs, ratio, start_pos_dy, rope_head_dim
        )
    elif start_pos != 0 and not rotate and ratio == 128:
        compressor_decode_ratio_128(
            *pto_inputs, *pto_outputs, ratio, start_pos_dy, rope_head_dim
        )


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("compressor_decode(Tensor x, Tensor kv_state, Tensor score_state, Tensor cache_index_2d, \
    Tensor sin, Tensor cos, Tensor wkv, Tensor wgate, Tensor ape, Tensor weight, int ratio, int start_pos, int kv_len_int, int rope_head_dim, bool rotate, Tensor hadamard) -> (Tensor, Tensor, Tensor)")


@torch.library.impl(pyptolib, "compressor_decode", "Meta")
def compressor_decode(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, 
        ratio, start_pos,  kv_len_int, rope_head_dim, rotate, hadamard):
    bsz = x.shape[0]
    overlap = ratio == 4
    coff = 1 + overlap
    d = weight.shape[0]
    cmp_seq = max(1, (kv_len_int - start_pos) // ratio)
    out = torch.zeros((bsz, cmp_seq, d), dtype=torch.bfloat16, device=x.device)
    kv_state_out = torch.zeros((bsz, coff * ratio, coff * d), dtype=torch.float32, device=x.device)
    score_state_out = torch.full((bsz, coff * ratio, coff * d), float("-inf"), dtype=torch.float32, device=x.device)
    return out, kv_state_out, score_state_out


@torch.library.impl(pyptolib, "compressor_decode", "NPU")
def compressor_decode(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, 
        ratio, start_pos,  kv_len_int, rope_head_dim, rotate, hadamard):
    bsz = x.shape[0]
    overlap = ratio == 4
    coff = 1 + overlap
    d = weight.shape[0]
    cmp_seq = max(1, (kv_len_int - start_pos) // ratio)
    out = torch.zeros((bsz, cmp_seq, d), dtype=torch.bfloat16, device=x.device)
    kv_state_out = torch.zeros((bsz, coff * ratio, coff * d), dtype=torch.float32, device=x.device)
    score_state_out = torch.full((bsz, coff * ratio, coff * d), float("-inf"), dtype=torch.float32, device=x.device)
    npu_compressor_decode(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, 
        out, kv_state_out, score_state_out, ratio, start_pos,  kv_len_int, rope_head_dim, rotate, hadamard=hadamard)
    return out, kv_state_out, score_state_out

def compressor_decode_graph(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, 
        ratio, start_pos,  kv_len_int, rope_head_dim, rotate, hadamard):
    return torch.ops.pypto.compressor_decode(x, kv_state, score_state, cache_index_2d, sin, cos, wkv, wgate, ape, weight, 
        ratio, start_pos,  kv_len_int, rope_head_dim, rotate, hadamard)
