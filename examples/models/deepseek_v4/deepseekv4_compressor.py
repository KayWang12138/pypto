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

import os
import torch
from numpy.testing import assert_allclose
from deepseekv4_compressor_impl import compressor_decode


def overlap_transform(tensor: torch.Tensor, value):
    # tensor: [b,s,r,2d]
    b, s, ratio, d = tensor.size()
    d = d // 2
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


def apply_rotary_pos_emb_v2(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, mode="half"
):
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


def golden_compress(
    x,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    kv_state,
    score_state,
    hadamard,
    ratio,
    start_pos,
    rope_head_dim,
    rotate,
    kv_len_int,
    eps=1e-6,
):
    bsz, _, _ = x.size()
    overlap = ratio == 4
    d = wkv.size(1) // (1 + overlap)
    dtype = x.dtype
    x = x.float()  ## b,s,h
    kv = torch.matmul(x, wkv)  ## b,s,2d
    score = torch.matmul(x, wgate)  ## b,s,2d
    if start_pos == 0:
        should_compress = kv_len_int >= ratio
        remainder = kv_len_int % ratio
        cutoff = kv_len_int - remainder

        ##跳跃采样未适配
        sin = sin[:, : max(1, cutoff // ratio)]  ## b, cut, 64
        cos = cos[:, : max(1, cutoff // ratio)]  ## b, cut, 64

        offset = ratio if overlap else 0
        if overlap and cutoff >= ratio:
            kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]  ## b,4,2d
            score_state[:bsz, :ratio] = (score[:, cutoff - ratio : cutoff] + ape)  ## b,4,2d
        if remainder > 0:
            kv, kv_state[:bsz, offset : offset + remainder] = kv[:, :kv_len_int].split([cutoff, remainder], dim=1)
            score_state[:bsz, offset : offset + remainder] = (score[:, cutoff : cutoff + remainder] + ape[:remainder])
            score = score[:, :cutoff]  ## b,4*cut,2d
        kv = kv.unflatten(1, (-1, ratio))  ## b,cut,4,2d
        score = score.unflatten(1, (-1, ratio)) + ape  ## b,cut,4,2d
        if overlap:
            kv = overlap_transform(kv, 0)  ## b,cut,8,d
            score = overlap_transform(score, float("-inf"))  ## b,cut,8,d
        kv = (kv * score.softmax(dim=2)).sum(dim=2)  ## b,cut,d
    else:
        should_compress = (start_pos + 1) % ratio == 0
        score += ape[start_pos % ratio]
        if overlap:
            kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
            score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv_state_tmp = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)  ## b,8,d
                score_state_tmp = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
                kv = (kv_state_tmp * score_state_tmp.softmax(dim=1)).sum(dim=1, keepdim=True)  ## b,1,d
                kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
                score_state[:bsz, :ratio] = score_state[:bsz, ratio:]
        else:
            kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            score_state[:bsz, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)  ## b,1,d

    if not should_compress:
        return
    kv = RMSNorm(kv.to(dtype), eps, weight)  ## b,cut,d
    kv_rope = kv[..., -rope_head_dim:].clone()
    kv = kv.clone()
    kv[..., -rope_head_dim:] = apply_rotary_pos_emb_v2(kv_rope, sin, cos, "interleave")
    if rotate:
        kv = torch.matmul(kv, hadamard)  ## b,cut,d
    # if start_pos == 0:
    #       kv_cache[:bsz, :kv_len_int // ratio] = kv
    # else:
    #       kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
    return kv


def gen_inputs(bsz, seq, h, d, rope_head_dim, ratio, device):
    torch.manual_seed(42)
    overlap = ratio == 4
    coff = 1 + overlap
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device)
    sin = torch.rand((bsz, max(1, seq // ratio), rope_head_dim), dtype=torch.bfloat16, device=device)
    cos = torch.rand((bsz, max(1, seq // ratio), rope_head_dim), dtype=torch.bfloat16, device=device)
    wkv = torch.rand((h, coff * d), dtype=torch.float32, device=device)
    wgate = torch.rand((h, coff * d), dtype=torch.float32, device=device)
    ape = torch.rand((ratio, coff * d), dtype=torch.float32, device=device)
    weight = torch.ones(d, dtype=torch.float32, device=device)
    kv_state = torch.zeros((bsz, coff * ratio, coff * d), dtype=torch.float32, device=device)
    score_state = torch.full((bsz, coff * ratio, coff * d), float("-inf"), dtype=torch.float32, device=device)
    hadamard = torch.rand((d, d), dtype=torch.bfloat16, device=device) * (d**-0.5)
    return x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard


def test_decode():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)

    device_id = os.environ.get("TILE_FWK_DEVICE_ID", 0)
    device = f"npu:{device_id}"
    start_pos_ori = [255]
    ratio_4 = [4] * len(start_pos_ori)
    ratio_128 = [128] * len(start_pos_ori)

    rotate = [False, True] * len(ratio_4) + [False] * len(ratio_128)
    ratio = ratio_4 * 2 + ratio_128
    start_pos = start_pos_ori * (len(rotate) // 1)

    for st, ra, ro in zip(start_pos, ratio, rotate):
        print(
            f"test_compressor_decode (start_pos: {st}, ratio: {ra}, rotate: {ro}) begin!"
        )
        bsz = 3
        seq = 1
        kv_len_int = st + seq
        h = 4096
        if ro:
            d = 128
        else:
            d = 512
        rope_head_dim = 64
        overlap = ra == 4
        coff = 1 + overlap
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = (
            gen_inputs(bsz, seq, h, d, rope_head_dim, ra, device)
        )
        cache_index_2d = torch.arange(bsz * coff * ra, dtype=torch.int32, device=device).reshape(bsz, coff * ra)

        out = torch.zeros((bsz, 1, d), dtype=torch.bfloat16, device=device)
        kv_state_out = torch.zeros((bsz, coff * ra, coff * d), dtype=torch.float32, device=device)
        score_state_out = torch.full((bsz, coff * ra, coff * d), float("-inf"), dtype=torch.float32, device=device)

        compressor_decode(
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
            ra,
            st,
            kv_len_int,
            rope_head_dim,
            ro,
            hadamard=hadamard,
        )
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, hadamard = (
            gen_inputs(bsz, seq, h, d, rope_head_dim, ra, device)
        )
        kv = golden_compress(
            x,
            sin,
            cos,
            wkv,
            wgate,
            ape,
            weight,
            kv_state,
            score_state,
            hadamard,
            ra,
            st,
            rope_head_dim,
            ro,
            kv_len_int,
        )

        assert_allclose(kv_state_out.cpu().float().numpy(), kv_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(score_state_out.cpu().float().numpy(), score_state.cpu().float().numpy(),rtol=1e-3,atol=1e-3)
        if kv is not None:
            assert_allclose(out.cpu().float().numpy(), kv.cpu().float().numpy(), rtol=0.0078125, atol=1e-4)

        print(f"test_compressor_decode passed!")


if __name__ == "__main__":
    test_decode()
