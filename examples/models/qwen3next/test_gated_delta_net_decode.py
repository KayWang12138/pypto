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
import pypto
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np


def torch_rms_norm(x, gamma, eps):
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


def torch_recurrent_gated_delta_rule(q, k, v, g, beta, state, start_list, accept_len_list, norm_weight, 
                                        gate_data, linear_weight):
    t_bs, num_head, head_dim = v.shape
    b_size, _, _, _ = state.shape
    h_size, _ = linear_weight.shape
    eps = 1e-6
    state_output = torch.zeros(t_bs, h_size, dtype=torch.float32)
    final_out = torch.zeros(b_size, num_head, head_dim, head_dim, dtype=torch.float32)

    for b in range(b_size):
        length = accept_len_list[b]
        start_idx = start_list[b]

        curr_state = state[b]

        for t in range(length):
            q_i = q[start_idx+t]
            k_i = k[start_idx+t]
            g_i = g[start_idx+t]
            v_i = v[start_idx+t]
            beta_i = beta[start_idx+t]

            scale = 1 / q.shape[-1]**0.5

            q_norm = torch.nn.functional.normalize(q_i, p=2, dim=-1) * scale
            k_norm = torch.nn.functional.normalize(k_i, p=2, dim=-1)

            curr_state = curr_state * g_i.unsqueeze(-1).exp()
            kv_mem = curr_state * k_norm.unsqueeze(-1)
            kv_mem_reduced = kv_mem.sum(-2)
            delta = v_i - kv_mem_reduced
            delta_beta = delta * beta_i # [num_head, head_dim]
            # [num_head, head_dim, 1] * [num_head, 1, head_dim] -> [num_head, head_dim, head_dim]
            k_delta = k_norm.unsqueeze(-1) * delta_beta.unsqueeze(-2) 
            curr_state = k_delta + curr_state
            
            core_out = curr_state * q_norm.unsqueeze(-1)
            core_out = core_out.sum(dim=-2)

            # zero-centered rms_norm
            core_norm_out = torch_rms_norm(core_out, norm_weight + 1, eps)

            # gate & linear
            gate_i = gate_data[start_idx+t]
            gate_silu = gate_i * torch.sigmoid(gate_i)
            core_gate_out = (core_norm_out * gate_silu).reshape([1, num_head * head_dim])

            linear_weight_fp32 = linear_weight.float()

            core_linear_out = torch.matmul(core_gate_out, linear_weight_fp32.transpose(0, 1))

            state_output[start_idx+t, :] = core_linear_out

            final_out[b] = curr_state
        
    return state_output, final_out


def torch_all_preproc(inputs, outputs):
    hidden_states = inputs[0]
    qkvz_weight = inputs[1]
    ba_weight = inputs[2]
    a_log = inputs[3]
    dt_bias = inputs[4]
    conv_weight = inputs[5]
    conv_state_in = inputs[6]
    init_state = inputs[7]
    start_list = inputs[8]
    accept_len_list = inputs[9]
    norm_weight = inputs[10]
    linear_weight = inputs[11]

    num_k_heads = 8
    num_v_heads = 16
    batch, seq_len, hidden_size = hidden_states.shape
    _, _, head_k_dim, _ = init_state.shape
    head_v_dim = head_k_dim
    conv_dim = 2*num_k_heads*head_k_dim + num_v_heads*head_v_dim
    conv_kernel_size = 4
    value_dim = num_v_heads * head_v_dim
    key_dim = num_k_heads * head_k_dim

    # Calling in_proj_qkvz() and in_proj_ba() in HF
    mixed_qkvz = hidden_states @ qkvz_weight.transpose(-1, -2)
    mixed_ba = hidden_states @ ba_weight.transpose(-1, -2)

    # Entering fix_query_key_value_ordering() in HF
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        num_k_heads,
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads,
    ) # b_size s_size k_h 2*d+2*d*v_h/k_h
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

    # b_size num_head KH*head_dim
    query, key, value, z = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value, z)) 
    mixed_qkv = torch.cat((query, key, value), dim=-1) # b_size num_head KH*head_dim+KH*head_dim+VH*head_dim
    mixed_qkv = mixed_qkv.transpose(1, 2) # b_size KH*head_dim+KH*head_dim+VH*head_dim num_head

    # entering torch_causal_conv1d_update()
    hidden_states = mixed_qkv
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state_in.shape[-1]
    hidden_states_new = torch.cat([conv_state_in, mixed_qkv], dim=-1).to(conv_weight.dtype)
    conv_state_out = hidden_states_new[:, :, -state_len:]
    out = torch.nn.functional.conv1d(
        input=hidden_states_new,
        weight=conv_weight,
        stride=(1,),
        padding=0,
        groups=hidden_size
        )
    out = torch.nn.functional.silu(out[:, :, -seq_len:])
    out = out.to(mixed_qkv.dtype)
    #exiting torch_causal_conv1d_update()
    mixed_qkv_after_conv = out

    mixed_qkv_after_trans = mixed_qkv_after_conv.transpose(1, 2)

    query, key, value = torch.split(
        mixed_qkv_after_trans,
        [
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

    query = query.reshape([query.shape[0], query.shape[2], query.shape[3]])
    key = key.reshape([key.shape[0], key.shape[2], key.shape[3]])
    value = value.reshape([value.shape[0], value.shape[2], value.shape[3]])
    g = g.reshape([g.shape[0], g.shape[2], 1])
    beta = beta.reshape([beta.shape[0], beta.shape[2], 1])
    z = z.reshape([z.shape[0], key.shape[1], key.shape[2]])

    attn_out, final_state = torch_recurrent_gated_delta_rule(query, key, value, g, beta, init_state, start_list, 
                                accept_len_list, norm_weight, z, linear_weight)

    outputs[0] = attn_out
    outputs[1] = final_state
    outputs[2] = conv_state_out


def pto_core_compute_qkvzba(
    hidden_states, # b_size s_size head_dim
    qkvz_weight, # 1, head_dim d*...
    ba_weight, # head_dim ...
    # output
    mixed_qkv, z, # b_size s_size Hd
    b, a, # b_size s_size h_size
    k_h, v_h, d
):
    
    b_size, s_size, head_dim = hidden_states.shape

    s_step = 1
    d_step = 1

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    # Calling in_proj_qkvz() and in_proj_ba() in HF
    projected_states_qkvz = pypto.tensor([b_size, s_size, d * (2 * k_h + 2 * v_h)], pypto.DT_FP32)
    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP00", idx_name="LOOP00IDX"):
        for d_idx in pypto.loop(0, qkvz_weight.shape[1], d_step, name="LOOP01", idx_name="LOOP01IDX"):
            pypto.set_vec_tile_shapes(8, 1, 8)
            hidden_states_view = hidden_states[:, s_idx:s_idx + s_step, :]
            qkvz_weight_view = qkvz_weight[:, d_idx:d_idx + d_step, :]
            projected_view = pypto.matmul(hidden_states_view, qkvz_weight_view, pypto.DT_FP32, b_trans=True)
            projected_states_qkvz[:, s_idx:s_idx + s_step, d_idx:d_idx + d_step] = projected_view

    projected_states_ba = pypto.tensor([b_size, s_size, 2 * v_h], pypto.DT_FP32)
    for s_idx in pypto.loop(0, s_size, 1, name="LOOP02", idx_name="LOOP02IDX"):
        for d_idx in pypto.loop(0, 2 * v_h, d_step, name="LOOP03", idx_name="LOOP03IDX"):
            pypto.set_vec_tile_shapes(8, 1, 8)
            hidden_states_view = hidden_states[:, s_idx:s_idx + 1, :]
            ba_weight_view = ba_weight[:, d_idx:d_idx + d_step, :]
            projected_view = pypto.matmul(hidden_states_view, ba_weight_view, pypto.DT_FP32, b_trans=True)
            projected_states_ba[:, s_idx:s_idx + s_step, d_idx:d_idx + d_step] = projected_view

    # qkvz
    middle_states_qkvz = pypto.tensor([b_size, s_size, k_h, (2 * d + 2 * d * v_h // k_h)], pypto.DT_FP32)
    middle_z = pypto.tensor([b_size, s_size, k_h, d * v_h // k_h], pypto.DT_FP32)
    for _ in pypto.loop(0, 1, 1, name="foo_reshape", idx_name="foo_idx"):
        pypto.set_vec_tile_shapes(16, 16, 16)
        middle_states_qkvz[:, :, :, :] = projected_states_qkvz.reshape([b_size, s_size, k_h, \
        (2 * d + 2 * d * v_h // k_h)])

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP04", idx_name="LOOP04IDX"):
        pypto.set_vec_tile_shapes(1, 1, 1, 16)
        # [b_size, s_step, k_h*(2*d+v_h//k_h*d)] -> [b_size, s_step, k_h*2*d]
        q_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, :d].reshape([b_size, s_step, k_h * d]) 
        # [b_size, s_step, k_h*(2*d+v_h//k_h*d)] -> [b_size, s_step, k_h*2*d]
        k_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, d:2 * d].reshape([b_size, s_step, k_h * d]) 
        # [b_size, s_step, k_h*(2*d+v_h//k_h*d)] -> [b_size, s_step, k_h*2*d]
        v_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, \
        2 * d: 2 * d + v_h // k_h * d].reshape([b_size, s_step, v_h * d]) 
        z_view = middle_states_qkvz[:, s_idx:s_idx + s_step, :, 2 * d + v_h // k_h * d:]
        pypto.set_vec_tile_shapes(1, 1, 16)
        qkv_view = pypto.concat([q_view, k_view, v_view], 2).reshape([b_size, s_step, 2 * d * k_h + v_h * d])
        qkv_transposed = pypto.transpose(qkv_view, 1, 2)
        mixed_qkv[:, :, s_idx:s_idx + s_step] = qkv_transposed
        middle_z[:, s_idx:s_idx + s_step, :, :] = z_view
    
    for _ in pypto.loop(0, 1, 1, name="foo_reshape", idx_name="foo_idx"):
        pypto.set_vec_tile_shapes(1, 16, 16, 16)
        z[:, :, :] = middle_z.reshape([b_size, v_h, d])

    middle_ba = pypto.tensor([b_size, s_size, k_h, 2 * v_h // k_h], pypto.DT_FP32)
    for _ in pypto.loop(0, 1, 1, name="foo_reshape", idx_name="foo_idx"):
        pypto.set_vec_tile_shapes(16, 16, 16)
        middle_ba[:, :, :, :] = projected_states_ba.reshape([b_size, s_size, k_h, 2 * v_h // k_h])

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP05", idx_name="LOOP05IDX"):
        pypto.set_vec_tile_shapes(1, 16, 16, 16)
        b_view = middle_ba[:, s_idx:s_idx + s_step, :, :v_h // k_h].reshape([b_size, s_step, v_h])
        a_view = middle_ba[:, s_idx:s_idx + s_step, :, v_h // k_h:].reshape([b_size, s_step, v_h])

        pypto.set_vec_tile_shapes(1, 16, 16)
        b[:, s_idx:s_idx + s_step, :] = b_view
        a[:, s_idx:s_idx + s_step, :] = a_view

    
def ab_preproc(a, b, a_log, dt_bias, g, beta):

    s_size = a.shape[1]
    s_step = 1
    thresh = 10

    for s_idx in pypto.loop(0, s_size, s_step, name="B_SIGMOID", idx_name="B_SIGMOID"):
        pypto.set_vec_tile_shapes(2, 1, 16)
        b_view = b[:, s_idx:s_idx + s_step, :]
        b_sigmoid = pypto.sigmoid(b_view)
        beta[:, s_idx:s_idx + s_step, :] = b_sigmoid
    
    for s_idx in pypto.loop(0, s_size, s_step, name="AB_PREPROC_1_S", idx_name="AB_PREPROC_1_S"):
        pypto.set_vec_tile_shapes(2, 1, 16)
        a_view = a[:, s_idx:s_idx + s_step, :]

        a_fp32 = pypto.cast(a_view, pypto.DT_FP32)
        a_biased = pypto.add(a_fp32, dt_bias)
        a_exp = pypto.exp(a_biased)
        a_exp_add = pypto.add(a_exp, 1) # ISSUE ALL INF
        a_softplus = pypto.log(a_exp_add) # ISSUE ALL INF

        a_view_large = a_view > pypto.full(a_view.shape, thresh, pypto.DT_FP32)
        # ...
        a_softplus_clean = pypto.where(a_view_large, a_biased, a_softplus)

        a_exp = pypto.exp(a_log)
        a_exp_neg = pypto.mul(a_exp, -1.)

        g_view = pypto.mul(a_softplus_clean, a_exp_neg)

        g[:, s_idx:s_idx + s_step, :] = g_view


class Conv1d():
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, groups=1):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def __call__(self,
        x:pypto.tensor,
        weights:pypto.tensor,
        left_pad:pypto.tensor,
        out_tensor: pypto.tensor,
        bias:pypto.tensor = None
    ) -> pypto.tensor:
        batch_size, conv_dim, input_length = x.shape
        d_type = x.dtype
        kernel_size = weights.shape[2]

        padded_x = pypto.tensor([x.shape[0], x.shape[1], left_pad.shape[2] + x.shape[2]], pypto.DT_FP32, "PADDED_X")
        input_length += left_pad.shape[2]
        pypto.set_vec_tile_shapes(8, 8, 8)
        step = 1
        for i in pypto.loop(0, left_pad.shape[2], step, name="PAD_1", idx_name="PAD_1"):
            left_pad_view = left_pad[:, :, i:i + step]
            padded_x[:, :, i:i + step] = left_pad_view

        for i in pypto.loop(left_pad.shape[2], left_pad.shape[2] + x.shape[2], step, name="PAD_2", idx_name="PAD_2"):
            x_view = x[:, :, i - left_pad.shape[2]:i - left_pad.shape[2] + step]
            padded_x[:, :, i:i + step] = x_view

        output_length = (input_length - kernel_size) // self.stride + 1
        in_group_size = self.in_channel // self.groups # 1
        out_group_size = self.out_channel // self.groups # 1
        for group_idx in pypto.loop(self.groups, name="group_loop", idx_name="group_idx"):
            for conv_idx in pypto.loop(output_length, name="conv_loop", idx_name="conv_idx"):
                pypto.set_vec_tile_shapes(8, 8, 8)
                weight_window = pypto.view(weights, [out_group_size, in_group_size, self.kernel_size],
                                            [group_idx * out_group_size, 0, 0])
                start_idx = conv_idx * self.stride
   
                x_window = pypto.view(padded_x, [batch_size, in_group_size, self.kernel_size], 
                            [0, group_idx * in_group_size, start_idx])
                pypto.set_vec_tile_shapes(8, 8, 8, 8)
                conv_out = pypto.sum(pypto.mul(pypto.unsqueeze(x_window, 1), pypto.unsqueeze(weight_window, 0)), -1, 
                                keepdim=True)
                conv_out = pypto.reshape(pypto.sum(conv_out, -2, keepdim=True), [batch_size, out_group_size, 1])
                pypto.assemble(conv_out, [0, group_idx * out_group_size, conv_idx], out_tensor)
        
        if bias is not None:
            for k in pypto.loop(1):
                out_tensor[:] = out_tensor + pypto.reshape(bias, [1, bias.shape[0], 1])

    
def compute_conv1d(mixed_qkv, weight, conv_state_in, conv_res, conv_state_out):
    conv_dim = weight.shape[0]
    conv_kernel_size = weight.shape[-1]
    num_batch = mixed_qkv.shape[0]
    seq_len = mixed_qkv.shape[-1]

    s_step = 1
    for s_idx in pypto.loop(seq_len, conv_kernel_size, s_step, name="CONCAT_1", idx_name="CONCAT_1"):
        pypto.set_vec_tile_shapes(8, 8, 8)
        conv_state_in_view = conv_state_in[:, :, s_idx:s_idx + s_step]
        conv_idx = s_idx - seq_len
        conv_state_out[:, :, conv_idx:conv_idx + s_step] = conv_state_in_view
    
    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass
    for s_idx in pypto.loop(max(0, seq_len - conv_kernel_size), seq_len, s_step, name="CONCAT_2", idx_name="CONCAT_2"):
        pypto.set_vec_tile_shapes(8, 8, 8)
        qkv_view = mixed_qkv[:, :, s_idx:s_idx + s_step]
        conv_idx = s_idx - seq_len + conv_kernel_size
        conv_state_out[:, :, conv_idx:conv_idx + s_step] = qkv_view

    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass

    conv1d = Conv1d(
        in_channel=conv_dim,
        out_channel=conv_dim,
        kernel_size=conv_kernel_size,
        stride=1,
        padding=0,
        groups=conv_dim
    )
    conv_result = pypto.tensor([num_batch, conv_dim, 2], pypto.DT_FP32)
    conv1d(mixed_qkv, weight, conv_state_in, conv_result) # b_size c_d s_size+k_size-1

    for _ in pypto.loop(0, 1, 1, name="WAIT", idx_name="WAIT", submit_before_loop=True):
        pass

    s_step = 1
    for s_idx in pypto.loop(0, 1, 1, name="SILU", idx_name="SILU"):
        pypto.set_vec_tile_shapes(8, 8, 8)
        conv_view = conv_result[:, :, s_idx + 1:s_idx + 2]
        sigm_view = pypto.sigmoid(conv_view)
        out_view = pypto.mul(conv_view, sigm_view)
        conv_res[:, :, 0:1] = out_view


def compute_interleave(mixed_qkv, query, key, value, k_h, v_h, d):

    b_size = mixed_qkv.shape[0]
    s_size = mixed_qkv.shape[2]

    s_step = 1
    replication = v_h // k_h
    
    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP06", idx_name="LOOP06IDX"):
        for h_idx in pypto.loop(0, v_h, 1, name="LOOP07", idx_name="LOOP07IDX"):
            pypto.set_vec_tile_shapes(8, 8, 8)
            query_view = mixed_qkv[:, h_idx // replication * d:h_idx // replication * d + d, s_idx:s_idx + s_step]
            key_view = mixed_qkv[:, d * k_h + h_idx // replication * d:d * k_h + h_idx // replication * d + d, \
            s_idx:s_idx + s_step]

            query_t = pypto.transpose(query_view, 1, 2)
            key_t = pypto.transpose(key_view, 1, 2)

            query_reshaped = pypto.reshape(query_t, [b_size, 1, d])
            key_reshaped = pypto.reshape(key_t, [b_size, 1, d])
            pypto.set_vec_tile_shapes(8, 8, 8, 8)

            query[:, h_idx:, :] = query_reshaped
            key[:, h_idx:, :] = key_reshaped

    for s_idx in pypto.loop(0, s_size, s_step, name="LOOP08", idx_name="LOOP08IDX"):
        for h_idx in pypto.loop(0, v_h * d, d, name="LOOP09", idx_name="LOOP09IDX"):
            pypto.set_vec_tile_shapes(8, 8, 8)
            value_view = mixed_qkv[:, d * 2 * k_h + h_idx:d * 2 * k_h + h_idx + d, s_idx:s_idx + s_step]

            value_t = pypto.transpose(value_view, 1, 2)

            value_reshaped = pypto.reshape(value_t, [b_size, 1, d])
            pypto.set_vec_tile_shapes(8, 8, 8, 8)

            value[:, h_idx // d:, :] = value_reshaped
    

def key_state(q, k, v, g, beta, state, start_list, accept_length_list, norm_weight, gate_data, linear_weight, 
        core_attn_out, final_out):
    eps = 1e-6
    t_bs, num_head, head_dim = v.shape
    b_size, _, _, _ = state.shape

    tmp_state = pypto.tensor([t_bs + 4, num_head, head_dim, head_dim], pypto.DT_FP32, "tmp_state")

    scale = 1 / q.shape[-1]**0.5

    for b in pypto.loop(0, b_size, submit_before_loop=True):
        start_idx = start_list[b]
        tmp_state[start_idx: start_idx+1, :, :, :] = state[b:b+1, :, :, :]
        del start_idx

    for b in pypto.loop(0, b_size, submit_before_loop=True):
        length = accept_length_list[b]
        start_idx = start_list[b]
        for t in pypto.loop(0, length, submit_before_loop=True):
            t_idx = t + start_idx
            q_i = q[t_idx:t_idx + 1, :, :]
            k_i = k[t_idx:t_idx + 1, :, :]
            v_i = v[t_idx:t_idx + 1, :, :]
            g_i = g[t_idx:t_idx + 1, :, :]
            beta_i = beta[t_idx:t_idx + 1, :, :]

            state_i = tmp_state[t_idx:t_idx + 1, :, :, :]

            # do l2norm(q)
            pypto.set_vec_tile_shapes(1, 1, 128)
            q_sqrt = pypto.sqrt((q_i * q_i).sum(-1, keepdim=True) + 1e-6)
            pypto.set_vec_tile_shapes(1, 1, 128)
            q_i = (q_i / q_sqrt) * scale

            # do l2norm(k)
            pypto.set_vec_tile_shapes(1, 1, 128)
            k_sqrt = pypto.sqrt((k_i * k_i).sum(-1, keepdim=True) + 1e-6)
            pypto.set_vec_tile_shapes(1, 1, 128)
            k_i = k_i / k_sqrt

            pypto.set_vec_tile_shapes(1, 128, 128)
            g_i2 = pypto.expand_clone(g_i, [1, num_head, head_dim]).exp()
            g_i3 = g_i2.unsqueeze(-1)
            pypto.set_vec_tile_shapes(1, 1, 128, 128)
            state_i2 = state_i * g_i3
            # compute kvmem
            kv_mem = state_i2 * k_i.unsqueeze(-1)
            # [1, num_head, head_dim, head_dim] -> [1, num_head, 1, head_dim] -> [1, num_head, head_dim]
            kv_mem_reduced = kv_mem.sum(2, keepdim=True).reshape([1, num_head, head_dim]) 
            # compute delta
            pypto.set_vec_tile_shapes(1, 1, 128)
            delta = v_i - kv_mem_reduced
            delta_beta = delta * beta_i
            # update state
            pypto.set_vec_tile_shapes(1, 1, 128, 128)
            k_delta = pypto.expand_clone(k_i.unsqueeze(-1), 
            [1, num_head, head_dim, head_dim]) * delta_beta.unsqueeze(-2)
            pypto.set_vec_tile_shapes(1, 1, 128)
            state_i3 = state_i2 + k_delta
            
            # get core_attn ouptut
            pypto.set_vec_tile_shapes(1, 1, 128, 128)
            core_out = state_i3 * q_i.unsqueeze(-1)
            core_out_reduced = core_out.sum(-2, keepdim=True).reshape([num_head, head_dim])

            # zero-centered rms_norm
            pypto.set_vec_tile_shapes(128)
            mean_coff = 1.0 / core_out_reduced.shape[-1]
            norm_weight_3d = pypto.reshape(norm_weight, [1, head_dim])
            pypto.set_vec_tile_shapes(128, 128)
            norm_weight_fp32 = pypto.cast(norm_weight_3d, pypto.DT_FP32)
            square = pypto.mul(core_out_reduced, core_out_reduced)
            mean_res = pypto.mul(square, mean_coff)
            reduce_asum = pypto.sum(mean_res, keepdim=True)
            reduce_sum = pypto.add(reduce_asum, eps)
            reduce_sqrt = pypto.sqrt(reduce_sum)
            res_div = pypto.div(core_out_reduced, reduce_sqrt)
            core_norm_out = pypto.mul(res_div, pypto.add(norm_weight_fp32, 1.0))

            # gate
            pypto.set_vec_tile_shapes(1, 128, 128)
            gate_i = gate_data[t_idx:t_idx + 1, :, :]
            gate_i = pypto.reshape(gate_i, [num_head, head_dim])
            pypto.set_vec_tile_shapes(128, 128)
            gate_silu = gate_i * pypto.sigmoid(gate_i)
            core_gate_out = pypto.mul(core_norm_out, gate_silu).reshape([1, num_head * head_dim])

            # linear
            pypto.set_vec_tile_shapes(128, 128)
            linear_weight_fp32 = pypto.cast(linear_weight, pypto.DT_FP32)
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            core_linear_out = pypto.matmul(core_gate_out, linear_weight_fp32, pypto.DT_FP32, 
                                a_trans=False, b_trans=True)
            
            core_attn_out[t_idx:t_idx + 1, :] = core_linear_out
            if pypto.cond(t < length - 1):
                tmp_state[t_idx + 1:t_idx + 2, :, :, :] = state_i3
            if pypto.cond(t == length - 1):
                final_out[b:b + 1, :, :, :] = state_i3
    
    del tmp_state
    del length
    del start_idx


def compute_all_preproc(
    inputs, # inputs
    outputs # outputs
):
    hidden_states, qkvz_weight, ba_weight, a_log, dt_bias, conv_weight, conv_state_in, init_state, \
    start_list, accept_length_list, norm_weight, linear_weight = inputs
    core_attn_out, final_state, conv_state_out = outputs
    
    b_size, num_head, head_dim, _ = init_state.shape
    d = 128
    s_size = 1
    t_bs = b_size
    v_h = num_head
    k_h = qkvz_weight.shape[1] // 2 // head_dim - v_h
    
    mixed_qkv = pypto.tensor([b_size, head_dim * (2 * k_h + v_h), s_size], pypto.DT_FP32, "")
    b = pypto.tensor([b_size, s_size, v_h], pypto.DT_FP32, "")
    a = pypto.tensor([b_size, s_size, v_h], pypto.DT_FP32, "")
    z = pypto.tensor([t_bs, v_h, head_dim], pypto.DT_FP32, "")

    
    query = pypto.tensor([t_bs, v_h, head_dim], pypto.DT_FP32, "query_workspace")
    key = pypto.tensor([t_bs, v_h, head_dim], pypto.DT_FP32, "key_workspace")
    value = pypto.tensor([t_bs, v_h, head_dim], pypto.DT_FP32, "value_workspace")
    g = pypto.tensor([b_size, v_h, 1], pypto.DT_FP32, "g_workspace")
    beta = pypto.tensor([b_size, v_h, 1], pypto.DT_FP32, "beta_workspace")

    pto_core_compute_qkvzba(
        hidden_states, qkvz_weight, ba_weight, # inputs
        mixed_qkv, z, b, a, # outputs
        k_h, v_h, head_dim # dims
    )
    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass

    ab_preproc(a, b, a_log, dt_bias, g, beta)

    for _ in pypto.loop(0, 1, 1, name="", idx_name="", submit_before_loop=True):
        pass

    mixed_qkv_after_conv = pypto.tensor([b_size, head_dim * (2 * k_h + v_h), s_size], pypto.DT_FP32, "")
    compute_conv1d(mixed_qkv, conv_weight, conv_state_in, mixed_qkv_after_conv, conv_state_out)

    compute_interleave(mixed_qkv_after_conv, query, key, value, k_h, v_h, head_dim)

    key_state(query, key, value, g, beta, init_state, start_list, accept_length_list, norm_weight, z, 
                linear_weight, core_attn_out, final_state)


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True},
    pass_options={"cycle_upper_bound": 256, "cycle_lower_bound":128}
)
def pto_all_preproc(inputs, outputs):
    compute_all_preproc(inputs, outputs)


def test_all_preproc():
    device = 3
    torch.npu.set_device(device)
    device = f'npu:{device}'

    b_size = 2
    s_size = 1
    head_dim = 128
    k_h = 8
    v_h = 16
    num_head = v_h
    h_size = 2048
    t_bs = b_size

    c_d = (2 * k_h + v_h) * head_dim # Conv dim
    k_size = 4 # Kernel size

    torch.manual_seed(0)

    hidden_states = torch.rand([b_size, s_size, h_size]).float()
    qkvz_weight = torch.rand([1, head_dim * (k_h + k_h + v_h + v_h), h_size]).float()
    ba_weight = torch.rand([1, 2 * v_h, h_size]).float()
    a_log = torch.rand([1, 1, v_h]).float()
    dt_bias = torch.rand([1, 1, v_h]).float()
    conv_weight = torch.rand([c_d, c_d // c_d, k_size]).float()
    conv_state_in = torch.rand([b_size, c_d, k_size]).float()
    state_data = torch.randn([b_size, v_h, head_dim, head_dim], dtype=torch.float)
    # gdr after inputs
    norm_weight = torch.full((head_dim,), 0, dtype=torch.int32)
    linear_weight = torch.randn([h_size, num_head * head_dim], dtype=torch.bfloat16)

     # index lists
    accept_length_list = torch.randint(1, 2, size=[b_size], dtype=torch.int32) # should be all 1
    start_list_1 = torch.cumsum(accept_length_list, -1)
    start_list = torch.zeros_like(start_list_1)
    start_list[1:] = start_list_1[:-1]
    start_list = start_list.to(torch.int32)

    core_attn_result = torch.ones([t_bs, h_size], dtype=torch.float)
    final_state_result = torch.ones([b_size, v_h, head_dim, head_dim], dtype=torch.float)
    conv_state_out = torch.ones([b_size, c_d, k_size]).float()

    inputs_cpu = [hidden_states, qkvz_weight, ba_weight, a_log, dt_bias, conv_weight, conv_state_in, 
                state_data, start_list, accept_length_list, norm_weight, linear_weight]
    inputs_npu = [tsr.to(device) for tsr in inputs_cpu]

    outputs_cpu = [core_attn_result, final_state_result, conv_state_out]
    outputs_npu = [tsr.to(device) for tsr in outputs_cpu]

    pto_inputs = [pypto.from_torch(x, "IN") for x in inputs_npu]
    pto_outputs = [pypto.from_torch(x, "OUT") for x in outputs_npu]

    torch_all_preproc(inputs_cpu, outputs_cpu)
    pto_all_preproc(pto_inputs, pto_outputs)

    npu_output = outputs_npu[0].cpu()
    cpu_output = outputs_cpu[0].cpu()

    diff_pos = np.where(np.array(abs(npu_output - cpu_output) / abs(cpu_output) >= 0.001))

    assert not torch.any(outputs_npu[0].cpu() == 0.)
    assert not torch.any(outputs_npu[1].cpu() == 0.)
    assert not torch.any(outputs_npu[2].cpu() == 0.)

    torch.testing.assert_close(outputs_cpu[0], outputs_npu[0].cpu(), atol=0.001, rtol=0.01)
    torch.testing.assert_close(outputs_cpu[1], outputs_npu[1].cpu(), atol=0.001, rtol=0.001)
    torch.testing.assert_close(outputs_cpu[2], outputs_npu[2].cpu(), atol=0.001, rtol=0.001)


if __name__ == "__main__":
    test_all_preproc()

