#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
from dataclasses import dataclass
import torch
import pypto
import torch.nn.functional as F


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) \
    for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) \
    for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_rms_norm(x, eps: float = 1e-6):
    gamma = torch.full((x.shape[-1],), 0, dtype=torch.int32, device=x.device) + 1

    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = mean_res.sum(dim=-1, keepdim=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def torch_gated_linear(core_attn_out, z, hidden_size):
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

    z_fp32 = z.to(torch.float32)
    sigmoid = torch.nn.Sigmoid()
    z_fp32_silu = z_fp32 * sigmoid(z_fp32)
    core_attn_out = core_attn_out * z_fp32_silu

    linear_weight_shape = [hidden_size, core_attn_out.shape[2]]
    # com
    # com
    linear_weight_fp32 = torch.full(linear_weight_shape, 1.0, dtype=torch.float32).to(device=core_attn_out.device)
    hidden_state_out = torch.matmul(core_attn_out, linear_weight_fp32.transpose(1, 0).unsqueeze(0))
    # com

    return hidden_state_out


def torch_all(inputs, outputs):
    hidden_states = inputs[0]
    qkvz_weight = inputs[1]
    ba_weight = inputs[2]
    a_log = inputs[3]
    dt_bias = inputs[4]
    conv_weight = inputs[5]
    init_state = inputs[6]

    num_k_heads = 2
    num_v_heads = 4
    batch, seq_len, head_k_dim = hidden_states.shape
    head_v_dim = head_k_dim
    conv_dim = 2*num_k_heads*head_k_dim + num_v_heads*head_v_dim
    conv_kernel_size = 4
    value_dim = num_v_heads * head_v_dim
    key_dim = num_k_heads * head_k_dim

    # Calling in_proj_qkvz() and in_proj_ba() in HF
    mixed_qkvz = torch.nn.functional.linear(hidden_states, qkvz_weight.squeeze(0))
    mixed_ba = torch.nn.functional.linear(hidden_states, ba_weight.squeeze(0))

    # Entering fix_query_key_value_ordering() in HF
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (num_k_heads, \
    2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    new_tensor_shape_ba = mixed_ba.size()[:-1] + (
        num_k_heads, 2 * num_v_heads // num_k_heads) # b_size s_size k_h 2*v_h/k_h
    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
    split_arg_list_qkvz = [
        head_k_dim, # d
        head_k_dim, # d
        (num_v_heads // num_k_heads * head_v_dim), # v_h/k_h*d
        (num_v_heads // num_k_heads * head_v_dim), # v_h/k_h*d
    ]
    split_arg_list_ba = [num_v_heads // num_k_heads, num_v_heads // num_k_heads]
    query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
    b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
    # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
    value = value.reshape(value.size(0), value.size(1), -1, head_v_dim)

    b = b.reshape(b.size(0), b.size(1), num_v_heads)
    a = a.reshape(a.size(0), a.size(1), num_v_heads)
    # Exiting fix_query_key_value_ordering() in HF

    query, key, value, z = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value, z))
    mixed_qkv = torch.cat((query, key, value), dim=-1) # b_size n_size KH*d_size+KH*d_size+VH*d_size
    mixed_qkv = mixed_qkv.transpose(1, 2) # b_size KH*d_size+KH*d_size+VH*d_size n_size

    mixed_qkv_after_conv = torch.nn.functional.conv1d(
        input=mixed_qkv,
        weight=conv_weight,
        stride=(1,),
        groups=conv_dim,
        padding=conv_kernel_size-1
    )

    mixed_qkv_after_conv = torch.nn.functional.silu(mixed_qkv_after_conv[:, :, :seq_len])

    mixed_qkv_after_trans = mixed_qkv_after_conv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv_after_trans, [
            key_dim,
            key_dim,
            value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(query.shape[0], query.shape[1], -1, head_k_dim)
    key = key.reshape(key.shape[0], key.shape[1], -1, head_k_dim)
    value = value.reshape(value.shape[0], value.shape[1], -1, head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)

    if num_v_heads // num_k_heads > 1:
        query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = torch_rms_norm(core_attn_out)
    hidden_state_out = torch_gated_linear(core_attn_out, z, hidden_states.shape[-1])

    outputs[0] = hidden_state_out
    outputs[1] = last_recurrent_state


def pto_core_compute_qkvzba(
    hidden_states, # b_size s_size d_size
    qkvz_weight, # 1, d_size d*...
    ba_weight, # d_size ...
    # output
    mixed_qkv, z, # b_size s_size Hd
    b, a, # b_size s_size H
    k_h, v_h, d
):

    b_size, s_size, d_size = hidden_states.shape

    s_step = 1
    d_step = 1

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    # Calling in_proj_qkvz() and in_proj_ba() in HF
    projected_states_qkvz = pypto.tensor([b_size, s_size, d * (2 * k_h + 2 * v_h)], pypto.DT_FP32)
    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP00", idx_name="LOOP00IDX"):
        def f_s(s_idx):
            for d_idx in pypto.loop(0, qkvz_weight.shape[1], d_step, name="LOOP01", idx_name="LOOP01IDX"):
                def f(d_idx):
                    pypto.set_vec_tile_shapes(8, 8, 8)
                    hidden_states_view = hidden_states[:, s_idx:s_idx + s_step, :]
                    qkvz_weight_view = qkvz_weight[:, d_idx:d_idx + d_step, :]
                    projected_view = pypto.matmul(hidden_states_view, qkvz_weight_view, pypto.DT_FP32, b_trans=True)
                    projected_states_qkvz[:, s_idx:s_idx + s_step, d_idx:d_idx + d_step] = projected_view
                f(d_idx)
        f_s(s_idx)

    projected_states_ba = pypto.tensor([b_size, s_size, 2 * v_h], pypto.DT_FP32)
    for s_idx in pypto.loop(0, s_size, 1, name="LOOP02", idx_name="LOOP02IDX"):
        def f_s(s_idx):
            for d_idx in pypto.loop(0, 2 * v_h, d_step, name="LOOP03", idx_name="LOOP03IDX"):
                def f(d_idx):
                    pypto.set_vec_tile_shapes(8, 8, 8)
                    hidden_states_view = hidden_states[:, s_idx:s_idx + 1, :]
                    ba_weight_view = ba_weight[:, d_idx:d_idx + d_step, :]
                    projected_view = pypto.matmul(hidden_states_view, ba_weight_view, pypto.DT_FP32, b_trans=True)
                    projected_states_ba[:, s_idx:s_idx + s_step, d_idx:d_idx + d_step] = projected_view
                f(d_idx)
        f_s(s_idx)

    # qkvz
    middle_states_qkvz = pypto.tensor([b_size, s_size, k_h, (2 * d + 2 * d * v_h // k_h)], pypto.DT_FP32)
    middle_z = pypto.tensor([b_size, s_size, k_h, d * v_h // k_h], pypto.DT_FP32)
    for _ in pypto.loop(0, 1, 1, name="foo_reshape", idx_name="foo_idx"):
        pypto.set_vec_tile_shapes(16, 16, 16)
        middle_states_qkvz[:, :, :, :] = projected_states_qkvz.reshape([b_size, s_size, k_h, \
        (2 * d + 2 * d * v_h // k_h)])

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP04", idx_name="LOOP04IDX"):
        def f(s_idx):
            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            q_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, :d].reshape([b_size, s_step, k_h * d])
            k_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, d:2 * d].reshape([b_size, s_step, k_h * d])
            v_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, 2 * d: 2 * d + v_h // k_h * d].reshape([b_size, \
            s_step, v_h * d])
            z_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, 2 * d + v_h // k_h * d:]
            pypto.set_vec_tile_shapes(16, 16, 16)
            qkv_view = pypto.concat([q_view, k_view, v_view], 2).reshape([b_size, s_step, 2 * d * k_h + v_h * d])
            qkv_transposed = pypto.transpose(qkv_view, 1, 2)
            mixed_qkv[:, :, s_idx:s_idx + s_step] = qkv_transposed
            middle_z[:, s_idx:s_idx + s_step, :, :] = z_view
        f(s_idx)

    for _ in pypto.loop(0, 1, 1, name="foo_reshape", idx_name="foo_idx"):
        pypto.set_vec_tile_shapes(1, 16, 16, 16)
        z[:, :, :] = middle_z.reshape([b_size, s_size, d * v_h])

    middle_ba = pypto.tensor([b_size, s_size, k_h, 2 * v_h // k_h], pypto.DT_FP32)
    for _ in pypto.loop(0, 1, 1, name="foo_reshape", idx_name="foo_idx"):
        pypto.set_vec_tile_shapes(16, 16, 16)
        middle_ba[:, :, :, :] = projected_states_ba.reshape([b_size, s_size, k_h, 2 * v_h // k_h])

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP05", idx_name="LOOP05IDX"):
        def f(s_idx):
            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            b_view = middle_ba[:, s_idx:s_idx + s_step, :, :v_h // k_h].reshape([b_size, s_step, v_h])
            a_view = middle_ba[:, s_idx:s_idx + s_step, :, v_h // k_h:].reshape([b_size, s_step, v_h])

            pypto.set_vec_tile_shapes(1, 16, 16)
            b[:, s_idx:s_idx + s_step, :] = b_view
            a[:, s_idx:s_idx + s_step, :] = a_view
        f(s_idx)


def ab_preproc(a, b, a_log, dt_bias, g, beta):

    s_size = a.shape[1]
    s_step = 1

    for s_idx in pypto.loop(0, s_size, s_step, name="AB_PREPROC_1_S", idx_name="AB_PREPROC_1_S"):
        def f(s_idx):
            pypto.set_vec_tile_shapes(8, 8, 8)
            a_view = a[:, s_idx:s_idx + s_step, :]
            b_view = b[:, s_idx:s_idx + s_step, :]

            a_fp32 = pypto.cast(a_view, pypto.DT_FP32)
            a_biased = pypto.add(a_fp32, dt_bias)
            a_exp = pypto.exp(a_biased)
            a_exp_add = pypto.add(a_exp, 1)
            a_softplus = pypto.log(a_exp_add)

            a_exp = pypto.exp(a_log)
            a_exp_neg = pypto.mul(a_exp, -1.)
            g_view = pypto.mul(a_softplus, a_exp_neg)

            b_sigmoid = pypto.sigmoid(b_view)

            g[:, s_idx:s_idx + s_step, :] = g_view
            beta[:, s_idx:s_idx + s_step, :] = b_sigmoid
        f(s_idx)


class Conv1d():
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, groups=1):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def __call__(self, x: pypto.tensor, weights: pypto.tensor, out_tensor: pypto.tensor, bias: pypto.tensor = None) -> \
    pypto.tensor:
        batch_size, conv_dim, input_length = x.shape
        d_type = x.dtype
        kernel_size = weights.shape[2]


        padded_x = pypto.tensor([x.shape[0], x.shape[1], x.shape[2] + 2 * self.padding], pypto.DT_FP32, "PADDED_X")
        input_length += 2 * self.padding
        pypto.set_vec_tile_shapes(8, 8, 8)
        step = 8
        for i in pypto.loop(0, self.padding, step, name="CONV1D_1", idx_name="CONV1D_1"):
            def f(i):
                padded_x[0:, 0:, i:i+step] = pypto.full([batch_size, conv_dim, step], 0., pypto.DT_FP32)
            f(i)
        for i in pypto.loop(0, self.padding, step, name="CONV1D_2", idx_name="CONV1D_2"):
            def f(i):
                padded_x[0:, 0:, self.padding + input_length + i:self.padding + input_length + i + step] = \
                pypto.full([batch_size, conv_dim, step], 0., pypto.DT_FP32)
            f(i)
        for i in pypto.loop(0, input_length, step, name="CONV1D_3", idx_name="CONV1D_3"):
            def f(i):
                padded_x[0:, 0:, self.padding+i : self.padding+i+step] = x[0:, 0:, i:i+step]
            f(i)

        output_length = (input_length - kernel_size) // self.stride + 1
        in_group_size = self.in_channel // self.groups # 1
        out_group_size = self.out_channel // self.groups # 1
        for group_idx in pypto.loop(self.groups, name="group_loop", idx_name="group_idx"):
            def f_s(group_idx):
                for conv_idx in pypto.loop(output_length, name="conv_loop", idx_name="conv_idx"):
                    def f(conv_idx):
                        pypto.set_vec_tile_shapes(8, 8, 8)
                        weight_window = pypto.view(weights, [out_group_size, in_group_size, self.kernel_size],
                                                [group_idx * out_group_size, 0, 0])
                        start_idx = conv_idx * self.stride

                        x_window = pypto.view(padded_x, [batch_size, in_group_size, self.kernel_size], \
                        [0, group_idx * in_group_size, start_idx])
                        pypto.set_vec_tile_shapes(8, 8, 8, 8)
                        conv_out = pypto.sum(pypto.mul(pypto.unsqueeze(x_window, 1), \
                        pypto.unsqueeze(weight_window, 0)), -1)
                        conv_out = pypto.reshape(pypto.sum(conv_out, -2), [batch_size, out_group_size, 1])
                        pypto.assemble(conv_out, [0, group_idx * out_group_size, conv_idx], out_tensor)
                    f(conv_idx)
            f_s(group_idx)

        if bias is not None:
            for k in pypto.loop(1):
                out_tensor[:] = out_tensor + pypto.reshape(bias, [1, bias.shape[0], 1])


def compute_conv1d(mixed_qkv, weight, out):
    conv_dim = weight.shape[0]
    conv_kernel_size = weight.shape[-1]
    num_batch = mixed_qkv.shape[0]
    seq_len = mixed_qkv.shape[-1]

    conv1d = Conv1d(
        in_channel=conv_dim,
        out_channel=conv_dim,
        kernel_size=conv_kernel_size,
        stride=1,
        padding=conv_kernel_size-1,
        groups=conv_dim
    )
    conv_result = pypto.tensor([num_batch, conv_dim, seq_len+conv_kernel_size-1], pypto.DT_FP32)
    conv1d(mixed_qkv, weight, conv_result) # b_size c_d s_size+k_size-1

    s_step = 1
    for s_idx in pypto.loop(0, seq_len, s_step, name="SILU_1", idx_name="SILU_1"):
        def f(s_idx):
            pypto.set_vec_tile_shapes(8, 8, 8)
            conv_view = conv_result[:, :, s_idx:s_idx + s_step]
            sigm_view = pypto.sigmoid(conv_view)
            out_view = pypto.mul(conv_view, sigm_view)
            out[:, :, s_idx:s_idx + s_step] = out_view
        f(s_idx)


def compute_interleave(mixed_qkv, query, key, value, k_h, v_h, d):

    b_size = mixed_qkv.shape[0]
    s_size = mixed_qkv.shape[2]

    s_step = 1
    replication = v_h // k_h

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP06", idx_name="LOOP06IDX"):
        def f_s(s_idx):
            for h_idx in pypto.loop(0, v_h, 1, name="LOOP07", idx_name="LOOP07IDX"):
                def f(h_idx):
                    pypto.set_vec_tile_shapes(8, 8, 8)
                    query_view = mixed_qkv[:, h_idx // replication * d:h_idx // replication * d + d, \
                    s_idx:s_idx + s_step]
                    key_view = mixed_qkv[:, d * k_h + h_idx // replication * d:d * k_h + h_idx // replication * d + d,
                    s_idx:s_idx + s_step]

                    query_t = pypto.transpose(query_view, 1, 2)
                    key_t = pypto.transpose(key_view, 1, 2)

                    query_reshaped = pypto.reshape(query_t, [b_size, s_step, 1, d])
                    key_reshaped = pypto.reshape(key_t, [b_size, s_step, 1, d])
                    pypto.set_vec_tile_shapes(8, 8, 8, 8)

                    query[:, s_idx:, h_idx:, :] = query_reshaped
                    key[:, s_idx:, h_idx:, :] = key_reshaped
                f(h_idx)
        f_s(s_idx)

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP08", idx_name="LOOP08IDX"):
        def f_s(s_idx):
            for h_idx in pypto.loop(0, v_h * d, d, name="LOOP09", idx_name="LOOP09IDX"):
                def f(h_idx):
                    pypto.set_vec_tile_shapes(8, 8, 8)
                    value_view = mixed_qkv[:, d * 2 * k_h + h_idx:d * 2 * k_h + h_idx + d, s_idx:s_idx + s_step]

                    value_t = pypto.transpose(value_view, 1, 2)

                    value_reshaped = pypto.reshape(value_t, [b_size, s_step, 1, d])
                    pypto.set_vec_tile_shapes(8, 8, 8, 8)

                    value[:, s_idx:, h_idx // d:, :] = value_reshaped
                f(h_idx)
        f_s(s_idx)


def cast_and_reshape(inputs, outputs):
    query, key, value, gate, beta = inputs
    query_result, key_result, value_result, gate_result, beta_result = outputs
    b, s, n, d = value.shape
    bn, c, l, d = value_result.shape

    view_b = min(2, b)
    view_n = min(2, n)

    for b_idx in pypto.loop(0, b, view_b, name="LOOP_B", idx_name="b_idx"):
        for n_idx in pypto.loop(0, n, view_n, name="LOOP_N", idx_name="n_idx"):
            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            query_block = query[b_idx:b_idx+view_b, :, n_idx:n_idx+view_n, :]
            query_result_block = pypto.cast(query_block.transpose(1, 2).reshape([view_b * view_n, c, l, d]), \
            pypto.DT_FP32)
            query_result[b_idx:b_idx+view_b, n_idx:n_idx+view_n, :, :] = query_result_block

            key_block = key[b_idx:b_idx+view_b, :, n_idx:n_idx+view_n, :]
            key_result_block = pypto.cast(key_block.transpose(1, 2).reshape([view_b*view_n, c, l, d]), pypto.DT_FP32)
            key_result[b_idx:b_idx+view_b, n_idx:n_idx+view_n, :, :] = key_result_block

            value_block = value[b_idx:b_idx+view_b, :, n_idx:n_idx+view_n, :]
            value_result_block = pypto.cast(value_block.transpose(1, 2).reshape([view_b * view_n, c, l, d]), \
            pypto.DT_FP32)
            value_result[b_idx:b_idx+view_b, n_idx:n_idx+view_n, :, :] = value_result_block

            pypto.set_vec_tile_shapes(1, 16, 16)
            gate_block = gate[b_idx:b_idx+view_b, :, n_idx:n_idx+view_n]
            gate_result_block = pypto.cast(gate_block.transpose(1, 2).reshape([view_b*view_n, c, l]), pypto.DT_FP32)
            gate_result[b_idx:b_idx+view_b, n_idx:n_idx+view_n, :] = gate_result_block

            beta_block = beta[b_idx:b_idx+view_b, :, n_idx:n_idx+view_n]
            beta_result_block = pypto.cast(beta_block.transpose(1, 2).reshape([view_b*view_n, c, l]), pypto.DT_FP32)
            beta_result[b_idx:b_idx+view_b, n_idx:n_idx+view_n, :] = beta_result_block


def pypto_l2norm(tensor: pypto.Tensor, result_tensor: pypto.Tensor, eps=1e-6):
    bn, c, l, d = tensor.shape
    view_bn = min(2, bn)
    view_c = min(2, c)

    for bn_idx in pypto.loop(0, bn, view_bn, name="LOOP_BN", idx_name="bn_idx"):
        for c_idx in pypto.loop(0, c, view_c, name="LOOP_C", idx_name="c_idx"):
            x = tensor[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :]
            pypto.set_vec_tile_shapes(1, 1, 32, 32)
            result_tensor[bn_idx:bn_idx + view_bn, c_idx:c_idx + view_c, :, :] = \
            x / pypto.sqrt((x * x).sum(-1, keepdim=True).reshape([view_bn, view_c, l, 1]) + eps)


def cal_cumsum(g, triu, cumsum_g):
    bn, c, l = g.shape
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    pypto.set_matrix_size([c, l, l])
    view_bn = min(1, bn)
    for bn_idx in pypto.loop(0, bn, view_bn, name="BN_LOOP", idx_name="bn_idx"):
        g_block = g[bn_idx:bn_idx+view_bn, :, :] # 1, c, l
        pypto.set_vec_tile_shapes(1, 16, 16)
        triu_block = pypto.expand_clone(triu.reshape([1, l, l]), [view_bn, l, l]) # 1, l, l
        cumsum_g[bn_idx:, :, :] = pypto.matmul(g_block, triu_block, pypto.DT_FP32) # 1 vC l_size


def cal_decay_mask(g, tril, decay_mask):
    bn, c, l = g.shape
    view_bn = min(2, bn)
    # comment
    for bn_idx in pypto.loop(0, bn, view_bn, name="BN_LOOP", idx_name="bn_idx"):
        pypto.set_vec_tile_shapes(2, 8, 16, 16)
        g_block = g[bn_idx:bn_idx+view_bn, :, :] # view_bn, c, l
        tril_block = tril[bn_idx:bn_idx+view_bn, :, :, :] # view_bn, c, l, l

        decay_mask_block = ((pypto.expand_clone(g_block.reshape([view_bn, c, l, 1]), [view_bn, c, l, l]) - \
                             pypto.expand_clone(g_block.reshape([view_bn, c, 1, l]), [view_bn, c, l, l])) * \
                             tril_block).exp() * tril_block
        decay_mask[bn_idx:bn_idx+view_bn, :, :, :] = decay_mask_block # view_bn, c, l, l


def cal_pre_attn(k_beta, key, decay_mask, mask, attn):
    bn, c, l, d = k_beta.shape
    view_bn = min(2, bn)
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    pypto.set_vec_tile_shapes(1, 8, 16, 16)
    # comment
    for bn_idx in pypto.loop(0, bn, view_bn, name="BN_LOOP", dx_name="bn_idx"):
        k_beta_block = k_beta[bn_idx:bn_idx+view_bn, :, :, :] # view_bn, c, l, d
        key_block = key[bn_idx:bn_idx+view_bn, :, :, :] # view_bn, c, l, d
        decay_mask_block = decay_mask[bn_idx:bn_idx+view_bn, :, :, :] # view_bn, c, l, l
        mask_block = mask[bn_idx:bn_idx+view_bn, :, :, :] # view_bn, c, l, l

        attn[bn_idx:bn_idx + view_bn, :, :, :] = pypto.matmul(k_beta_block, key_block.transpose(2, 3), \
        pypto.DT_FP32) * decay_mask_block * mask_block


def cal_inverse(attn, eye, tril, output):
    bn, c, l, l = attn.shape
    view_bn = min(2, bn)

    recurrent_attn = pypto.tensor([bn, c, l, l], pypto.DT_FP32, "recurrent_attn")
    for _ in pypto.loop(0, 1, 1, name="foo", idx_name="foo"):
        recurrent_attn[0:, 0:, 0:, 0:] = attn

    for i in pypto.loop(1, l, 1, name="LOOP_CHUNK", idx_name="i", submit_before_loop=True):
        for bn_idx in pypto.loop(0, bn, view_bn, name="LOOP_BN", idx_name="bn_idx", submit_before_loop=True):
            pypto.set_vec_tile_shapes(1, 1, l, l)
            row = recurrent_attn[bn_idx:bn_idx+view_bn, :, i:i+1, :]
            row_mask = tril[0:1, 0:1, i-1:i, :].reshape([1, l])
            sub = recurrent_attn[bn_idx:bn_idx+view_bn, :, :, :]

            left_mask = pypto.expand_clone(row_mask.unsqueeze(0).unsqueeze(-1), [1, 1, l, l])
            right_mask = pypto.expand_clone(row_mask.unsqueeze(0).unsqueeze(0), [1, 1, l, l])
            mask_2d = left_mask * right_mask
            tmp = (row.reshape([view_bn, c, l, 1]) * sub) * pypto.expand_clone(mask_2d, [view_bn, c, l, l])
            summed = tmp.sum(-2, keepdim=True)
            updated = (row + summed) * pypto.expand_clone(row_mask.reshape([1, 1, 1, l]), [view_bn, c, 1, l])
            recurrent_attn[bn_idx:bn_idx+view_bn, :, i:i+1, :] = updated

    for bn_idx in pypto.loop(0, bn, view_bn, name="LOOP_BN", idx_name="bn_idx"):
        attn_block = recurrent_attn[bn_idx:bn_idx+view_bn, :, :, :]
        eye_block = eye[bn_idx:bn_idx+view_bn, :, :, :]
        output[bn_idx:bn_idx+view_bn, :, :, :] = attn_block + eye_block


def cal_value_and_kcumdecay(inputs, outputs):
    attn, v_beta, k_beta, g = inputs
    value, k_cumdecay = outputs
    bn, c, l, d = v_beta.shape
    view_bn = min(2, bn)
    view_c = min(2, c)
    for bn_idx in pypto.loop(0, bn, view_bn, name="BN_LOOP", idx_name="bn_idx"):
        for c_idx in pypto.loop(0, c, view_c, name="C_LOOP", idx_name="c_idx"):
            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
            attn_block = attn[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] # view_bn, view_c, l, l
            v_beta_block = v_beta[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] # view_bn, view_c, l, d
            k_beta_block = k_beta[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] # view_bn, view_c, l, d
            g_block = g[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :] # view_bn, view_c, l

            value[bn_idx:bn_idx + view_bn, c_idx:c_idx + view_c, :, :] = \
            pypto.matmul(attn_block, v_beta_block, pypto.DT_FP32)
            k_cumdecay[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] = pypto.matmul(attn_block, \
                k_beta_block * g_block.exp().reshape([view_bn, view_c, l, 1]), pypto.DT_FP32) # view_bn, view_c, l, d


### new recurrent start ###
def recurrent_loop(inputs, outputs):
    pypto.set_vec_tile_shapes(16, 16, 16)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    middle_query, middle_key, middle_value, decay_mask, middle_k_cumdecay, gate, init_state, tril_mask = inputs
    core_attn_out, last_recurrent_state = outputs

    bn, c, l, d = middle_query.shape
    b, s, n, d = core_attn_out.shape

    view_bn = n
    view_s = l
    recurrent_state = pypto.tensor([bn, d, d], pypto.DT_FP32, "recurrent_state workspace")
    tmp_core_attn_out = pypto.tensor([bn, c, l, d], pypto.DT_FP32, "tmp_core_attn_out workspace")

    for _ in pypto.loop(0, 1, 1, name="", idx_name=""):
        recurrent_state[0:, 0:, 0:] = init_state.reshape([bn, d, d])

    for i in pypto.loop(0, c, 1, name="LOOP_RECURRENT", idx_name="r_idx", submit_before_loop=True):
        for bn_idx in pypto.loop(0, bn, view_bn, name="LOOP_BN", idx_name="bn_idx"):
            # view data
            _qi = middle_query[bn_idx:(bn_idx + view_bn), i:i + 1, :, :] # [view_bn, 1, l_size, d_size]
            _ki = middle_key[bn_idx:(bn_idx + view_bn), i:i + 1, :, :] # [view_bn, 1, l_size, d_size]
            _vi = middle_value[bn_idx:(bn_idx + view_bn), i:i + 1, :, :] # [view_bn, 1, l_size, d_size]
            _decay_mask = decay_mask[bn_idx:(bn_idx + view_bn), i:i + 1, :, :] # [view_bn, 1, l_size, l_size]
            _tril = tril_mask[bn_idx:(bn_idx + view_bn), i:i + 1, :, :] # [view_bn, 1, l_size, l_size]
            _k_cumdecay = middle_k_cumdecay[bn_idx:(bn_idx + view_bn), i:i + 1, :, :] # [view_bn, 1, l_size, d_size]
            _state = recurrent_state[bn_idx:(bn_idx + view_bn), :, :] # [view_bn, d_size, d_size]
            _gate = gate[bn_idx:(bn_idx + view_bn), i:i + 1, :] # [view_bn, 1, l_size]

            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            attn = pypto.matmul(_qi, _ki.transpose(2, 3), pypto.DT_FP32) * _decay_mask * _tril
            v_prime = pypto.matmul(_k_cumdecay, _state.reshape([view_bn, 1, d, d]), pypto.DT_FP32)
            v_new = _vi - v_prime # [view_bn, 1, l_size, d_size]
            attn_inter = pypto.matmul(_qi * _gate.reshape([view_bn, 1, l, 1]).exp(), \
            _state.reshape([view_bn, 1, d, d]), pypto.DT_FP32)
            _core_attn_out = attn_inter + pypto.matmul(attn, v_new, pypto.DT_FP32)
            pypto.set_vec_tile_shapes(16, 16, 16)
            _last_gate = pypto.expand_clone(_gate.reshape([view_bn, l])[:, l - 1:l], \
            [view_bn, l]).reshape([view_bn, 1, l]) # [view_bn, 1, l]
            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

            _tmp_matmul_result = \
            pypto.matmul((_ki * (_last_gate - _gate).reshape([view_bn, 1, l, 1]).exp()).transpose(2, 3), \
            v_new, pypto.DT_FP32) # [view_bn, 1, d_size, d_size]
            _last_gate_2 = pypto.expand_clone(_gate.reshape([view_bn, l])[:, l - 1:l], \
            [view_bn, d]).reshape([view_bn, d, 1])
            _updated_state = _state * _last_gate_2.exp() + _tmp_matmul_result.reshape([view_bn, d, d])

            # assemble
            tmp_core_attn_out[bn_idx:, i:i+1, :, :] = _core_attn_out
            recurrent_state[bn_idx:bn_idx+view_bn, :, :] = _updated_state

    for _ in pypto.loop(0, 1, 1, name="foo", idx_name="foo", submit_before_loop=True):
        last_recurrent_state[:] = recurrent_state.reshape([b, n, d, d])

    view_b = min(2, b)
    view_n = min(2, n)
    for _ in pypto.loop(0, 1, 1, name="foo", idx_name="foo", submit_before_loop=True):
        core_attn_out[:] = tmp_core_attn_out.reshape([b, n, c*l, d]).transpose(1, 2)


def gated_delta_rule_process(inputs, outputs):
    query, key, value, beta, gate, states, mask1, tril_mask, triu_mask, eye = inputs
    core_attn_out, final_state = outputs

    b, s, n, d = query.shape
    l = 64
    c = max(1, s // l)
    s = pypto.symbolic_scalar(key.shape[1])
    view_s = min(s, 64)

    pypto.set_vec_tile_shapes(1, 16, 16, 16)
    # defining workspace
    decay_mask = pypto.tensor([b * n, c, l, l], pypto.DT_FP32, "decay_mask_workspace")
    tmp_inverse = pypto.tensor([b * n, c, l, l], pypto.DT_FP32, "tmp_inverse_workspace")
    tmp_mask_mul = pypto.tensor([b * n, c, l, l], pypto.DT_FP32, "tmp_mask_mul_workspace")
    value_matmul = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "value_matmul_workspace")
    output_k_cumdecay = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "output_k_cumdecay_workspace")
    reshaped_query = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "reshaped_query_workspace")
    g = pypto.tensor([b*n, c, l], pypto.DT_FP32, "g_workspace")

    chunk_query = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "chunk_query")
    chunk_key = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "chunk_key")
    chunk_value = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "chunk_value")
    chunk_gate = pypto.tensor([b * n, c, l], pypto.DT_FP32, "chunk_gate")
    chunk_beta = pypto.tensor([b * n, c, l], pypto.DT_FP32, "chunk_beta")
    l2norm_query = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "l2norm_query")
    scaled_query = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "scaled_query")
    l2norm_key = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "l2norm_key")
    v_beta = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "v_beta_workspace")
    k_beta = pypto.tensor([b * n, c, l, d], pypto.DT_FP32, "k_beta_workspace")

    cast_and_reshape([query, key, value, gate, beta], [chunk_query, chunk_key, chunk_value, chunk_gate, chunk_beta])
    for _ in pypto.loop(0, 1, 1, name="foo", idx_name="foo", submit_before_loop=True):
        pass
    pypto_l2norm(chunk_query, l2norm_query)
    pypto_l2norm(chunk_key, l2norm_key)

    scale = 1 / query.shape[-1] ** 0.5

    view_bn = min(2, b*n)
    view_c = min(2, c)
    for bn_idx in pypto.loop(0, b*n, view_bn, name="LOOP_BN", idx_name="bn_idx"):
        for c_idx in pypto.loop(0, c, view_c, name="LOOP_C", idx_name="c_idx"):
            pypto.set_vec_tile_shapes(1, 16, 16, 16)
            value_block = chunk_value[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :]
            key_block = l2norm_key[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :]
            beta_block = chunk_beta[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :]
            query_block = l2norm_query[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :]

            v_beta_block = value_block * beta_block.reshape([view_bn, view_c, l, 1]) # b n s d
            k_beta_block = key_block * beta_block.reshape([view_bn, view_c, l, 1]) # b n s d
            scaled_query_block = query_block * scale

            v_beta[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] = v_beta_block
            k_beta[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] = k_beta_block
            scaled_query[bn_idx:bn_idx+view_bn, c_idx:c_idx+view_c, :, :] = scaled_query_block

    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass
    cal_cumsum(chunk_gate, triu_mask, g)
    cal_decay_mask(g, tril_mask, decay_mask)

    view_bn = min(2, b*n)
    view_c = min(8, c)
    view_m = min(8, l)
    view_n = min(8, l)
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])

    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass
    cal_pre_attn(k_beta, l2norm_key, decay_mask, mask1, tmp_mask_mul)
    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass
    cal_inverse(tmp_mask_mul, eye, tril_mask, tmp_inverse)
    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass
    cal_value_and_kcumdecay([tmp_inverse, v_beta, k_beta, g], [value_matmul, output_k_cumdecay])
    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass

    recurrent_loop([scaled_query, l2norm_key, value_matmul, decay_mask, output_k_cumdecay, g, states, tril_mask], \
    [core_attn_out, final_state])

    del decay_mask
    del tmp_mask_mul
    del tmp_inverse
    del value_matmul
    del output_k_cumdecay
    del reshaped_query
    del g

    del chunk_query
    del chunk_key
    del chunk_value
    del chunk_gate
    del chunk_beta
    del l2norm_query
    del l2norm_key
    del scaled_query
    del k_beta
    del v_beta


def pto_rms_norm(inputs, outputs):

    pypto.set_host_options(
        only_codegen=True
    )

    x = inputs[0]

    eps = 1e-6

    y = outputs[0]

    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype

    b = x.shape[0]
    s = x.shape[1]
    n = x.shape[2]
    d = x.shape[3]
    view_shape = [1, 1, n, d]
    tile_shape = [128, 128, 128, 128]

    b_loop = (b + view_shape[0] - 1) // view_shape[0]
    s_loop = (s + view_shape[1] - 1) // view_shape[1]

    def rms_inside_func():
        for b_idx in pypto.loop(b_loop, name="LOOP_RMS_NORM_L0", idx_name="b_idx"):
            def b_loop_func(b_idx):
                for s_idx in pypto.loop(s_loop, name="LOOP_RMS_NORM_L1", idx_name="s_idx"):
                    def s_loop_func(s_idx):
                        pypto.set_vec_tile_shapes(*tile_shape)
                        weight = pypto.full([x.shape[-1]], 0, pypto.DT_INT32)

                        tile_x = pypto.view(x, view_shape, [b_idx * view_shape[0], s_idx * view_shape[1], 0, 0], \
                        valid_shape=[(b - b_idx * view_shape[0]).min(view_shape[0]), \
                        (s - s_idx * view_shape[1]).min(view_shape[1]), n, d])

                        pypto.set_vec_tile_shapes(*tile_shape)

                        mean_coff = 1.0 / x.shape[-1]

                        # cast to calc_dtype
                        x_fp32 = pypto.cast(tile_x, calc_dtype)

                        weight_shape = [1]*len(x_fp32.shape)
                        weight_shape[-1] = weight.shape[0]
                        weight_4d = pypto.reshape(weight, weight_shape)
                        tile_weight_fp32 = pypto.cast(weight_4d, calc_dtype) # [1,1,1,d]

                        square = pypto.mul(x_fp32, x_fp32)
                        mean_res = pypto.mul(square, mean_coff)
                        reduce_asum = pypto.sum(mean_res, -1, keepdim=True) # [1,1,n,1]
                        reduce_sum = pypto.add(reduce_asum, eps)
                        reduce_sqrt = pypto.sqrt(reduce_sum)
                        res_div = pypto.div(x_fp32, reduce_sqrt) # [1,1,n,d]
                        res = pypto.mul(res_div, pypto.add(tile_weight_fp32, 1.0))

                        y_output = pypto.cast(res, input_dtype)
                        y[b_idx*view_shape[0]:, s_idx*view_shape[1]:, :, :] = y_output

                    s_loop_func(s_idx)
            b_loop_func(b_idx)
    rms_inside_func()
    assert isinstance(y, pypto.tensor)


@dataclass
class AttentionTileConfig:
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


def get_qwen_common_config():
    cube_tile = 128
    vector_tile = 128
    tile_cfg = AttentionTileConfig(
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile],
        [[cube_tile, cube_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [vector_tile, vector_tile])
    return tile_cfg


def pto_gated_linear(inputs, outputs):

    pypto.set_codegen_options(codegen_expression_fusion=True)

    delta_rule = inputs[0]
    gate_input = inputs[1]
    hidden_size = inputs[2]
    weight_shape = [hidden_size, gate_input.shape[2]]

    final_out = outputs[0]

    tile_cfg = get_qwen_common_config()
    n = delta_rule.shape[2]
    d = delta_rule.shape[3]
    dtype = delta_rule.dtype

    b_scalar = delta_rule.shape[0]
    s1_scalar = delta_rule.shape[1]

    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    def fun():
        for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx", submit_before_loop=True):
            def b_fun(b_idx):
                for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx", submit_before_loop=True):
                    def s1_fun(s1_idx):
                        for _ in pypto.loop(1, name="LOOP_gate_linear", idx_name="_", submit_before_loop=True):
                            def gate_linear_fun():
                                gate = pypto.reshape(gate_input, delta_rule.shape)
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                weight = pypto.full(weight_shape, 1.0, pypto.DT_FP32)
                                delta_rule_bs = pypto.view(delta_rule, [1, 1, n, d], [b_idx, s1_idx, 0, 0])
                                gate_bs = pypto.view(gate, [1, 1, n, d], [b_idx, s1_idx, 0, 0])

                                weight_bs = pypto.view(weight, [hidden_size, n * d], [0, 0])
                                weight_bs = pypto.reshape(weight_bs, [1, hidden_size, n * d])

                                pypto.set_vec_tile_shapes(1, 16, v1_tile[0], v1_tile[1])
                                delta_rule_bs_fp32 = pypto.cast(delta_rule_bs, pypto.DT_FP32)
                                gate_bs_fp32 = pypto.cast(gate_bs, pypto.DT_FP32)
                                weight_fp32 = pypto.cast(weight_bs, pypto.DT_FP32)

                                gate_bs_sub_2 = pypto.reshape(gate_bs_fp32, [n, d])
                                pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                                gate_bs_sub_3 = pypto.mul(gate_bs_sub_2, pypto.sigmoid(gate_bs_sub_2))
                                gate_bs_out = pypto.reshape(gate_bs_sub_3, [1, 1, n, d])

                                # gate
                                pypto.set_vec_tile_shapes(1, 16, v1_tile[0], v1_tile[1])
                                out_gate = pypto.mul(delta_rule_bs_fp32, gate_bs_out)
                                out_gate = pypto.reshape(out_gate, [1, 1, n * d])

                                # linear
                                pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                                out_linear = pypto.matmul(out_gate, weight_fp32, pypto.DT_FP32, \
                                a_trans=False, b_trans=True)
                                out_linear = pypto.cast(out_linear, dtype)
                                pypto.assemble(out_linear, [b_idx, s1_idx, 0], final_out)

                            gate_linear_fun()

                    s1_fun(s1_idx)

            b_fun(b_idx)

    fun()


def compute_all(
    hidden_states, qkvz_weight, ba_weight, a_log, dt_bias, conv_weight, init_state, mask1, tril_mask, triu_mask, eye,
    hidden_state_out, final_state
):

    b_size, s_size, d_size = hidden_states.shape
    n_size = a_log.shape[-1]

    v_h = n_size
    k_h = qkvz_weight.shape[1] // 2 // d_size - v_h

    mixed_qkv = pypto.tensor([b_size, d_size * (2 * k_h + v_h), s_size], pypto.DT_FP32, "MIXED_QKV_WORKSPACE")
    b = pypto.tensor([b_size, s_size, v_h], pypto.DT_FP32, "B_WORKSPACE")
    a = pypto.tensor([b_size, s_size, v_h], pypto.DT_FP32, "A_WORKSPACE")

    query = pypto.tensor([b_size, s_size, v_h, d_size], pypto.DT_FP32, "query_workspace")
    key = pypto.tensor([b_size, s_size, v_h, d_size], pypto.DT_FP32, "key_workspace")
    value = pypto.tensor([b_size, s_size, v_h, d_size], pypto.DT_FP32, "value_workspace")
    g = pypto.tensor([b_size, s_size, v_h], pypto.DT_FP32, "g_workspace")
    beta = pypto.tensor([b_size, s_size, v_h], pypto.DT_FP32, "beta_workspace")

    z = pypto.tensor([b_size, s_size, n_size * d_size], pypto.DT_FP32, "z_workspace")
    core_attn_out = pypto.tensor([b_size, s_size, n_size, d_size], pypto.DT_FP32, "core_attn_out_workspace")
    core_attn_out_2nd = pypto.tensor([b_size, s_size, n_size, d_size], pypto.DT_FP32, "core_attn_out_workspace")

    pto_core_compute_qkvzba(
        hidden_states, qkvz_weight, ba_weight, # inputs
        mixed_qkv, z, b, a, # outputs
        k_h, v_h, d_size # dims
    )

    ab_preproc(a, b, a_log, dt_bias, g, beta)

    mixed_qkv_after_conv = pypto.tensor([b_size, d_size * (2 * k_h + v_h), s_size], pypto.DT_FP32, \
    "MIXED_QKV_AFTER_CONV_WORKSPACE")
    compute_conv1d(mixed_qkv, conv_weight, mixed_qkv_after_conv)

    compute_interleave(mixed_qkv_after_conv, query, key, value, k_h, v_h, d_size)

    gated_delta_rule_process([query, key, value, beta, g, init_state, mask1, tril_mask, triu_mask, eye], \
    [core_attn_out, final_state])

    pto_rms_norm([core_attn_out], [core_attn_out_2nd])
    pto_gated_linear([core_attn_out_2nd, z, hidden_states.shape[-1]], [hidden_state_out])


@pypto.jit
def pypto_gated_delta_net(hidden_states, qkvz_weight, ba_weight, a_log,
                          dt_bias, conv_weight, init_state, mask1,
                          tril_mask, triu_mask, eye, hidden_state_out,
                          final_state):
    pypto.set_host_options(only_codegen=True)
    compute_all(hidden_states, qkvz_weight, ba_weight, a_log, dt_bias,
                conv_weight, init_state, mask1, tril_mask, triu_mask,
                eye, hidden_state_out, final_state)


def test_all():
    device = 3
    torch.npu.set_device(device)
    device = f'npu:{device}'

    b_size = 1
    s_size = 64
    d_size = 128
    k_h = 2
    v_h = 4
    d = 128

    l_size = 64
    c_size = (s_size + l_size - 1) // l_size

    c_d = (2 * k_h + v_h) * d_size # Conv dim
    k_size = 4 # Kernel size

    torch.manual_seed(0)

    hidden_states = torch.rand([b_size, s_size, d_size]).float()
    qkvz_weight = torch.rand([1, d * (k_h + k_h + v_h + v_h), d_size]).float()
    ba_weight = torch.rand([1, 2 * v_h, d_size]).float()
    a_log = torch.rand([1, 1, v_h]).float()
    dt_bias = torch.rand([1, 1, v_h]).float()
    conv_weight = torch.rand([c_d, c_d // c_d, k_size]).float()
    init_state = torch.zeros([b_size, v_h, d_size, d_size]).float()

    # helper data
    mask1 = torch.tril(-torch.ones([b_size * v_h, c_size, l_size, l_size], dtype=torch.float), diagonal=-1)
    tril_mask = torch.ones([b_size * v_h, c_size, l_size, l_size]).float().tril() # lower triangular
    triu_mask = torch.ones([l_size, l_size]).float().triu().reshape([1, l_size, l_size]) # Upper triangular
    eye = torch.eye(l_size).unsqueeze(0).unsqueeze(0).expand([b_size * v_h, c_size, l_size, \
    l_size]).contiguous().float()

    inputs_cpu = [hidden_states, qkvz_weight, ba_weight, a_log, dt_bias, conv_weight, init_state]
    inputs_npu = [tsr.to(device) for tsr in inputs_cpu] + [tsr.to(device) for tsr in [mask1, tril_mask, triu_mask, eye]]

    hidden_state_out = torch.zeros([b_size, s_size, d_size]).float()
    final_state = torch.zeros([b_size, v_h, d_size, d_size]).float()
    outputs_cpu = [hidden_state_out, final_state]
    outputs_npu = [tsr.to(device) for tsr in outputs_cpu]

    pto_inputs = [pypto.from_torch(x, "IN") for x in inputs_npu]
    pto_outputs = [pypto.from_torch(x, "OUT") for x in outputs_npu]

    torch_all(inputs_cpu, outputs_cpu)
    pypto_gated_delta_net(*pto_inputs, *pto_outputs)

    assert torch.allclose(outputs_cpu[0], outputs_npu[0].cpu(), 0.001, 0.001)
    assert torch.allclose(outputs_cpu[1], outputs_npu[1].cpu(), 0.001, 0.001)


if __name__ == "__main__":
    test_all()
