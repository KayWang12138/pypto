# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto
import torch
import torch_npu
import json
import random
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass

# from libs.generate_inputs import generate_inputs
# from libs.tools import convert_torch_to_numpy, get_torch_dtype, get_broadcast_offsets, generate_broadcast_info
# from libs.precision_compare import precision_compare_triple_data

@dataclass
class AssembleParams:
   batch_size: int
   seq_len: int 
   hidden_size: int
   tile_b: int
   tile_s: int 
   dtype: str
   cube_tile_shapes: list
   vector_tile_shapes_2d: list
   vector_tile_shapes: list

# def generate_case():
#     # 参数列表（保持原始顺序）
#     batch_size_list = [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024]
#     seq_len_list = [2, 4, 5, 6, 7, 10, 16, 32, 64, 127]
#     tile_b_list = [2, 4, 5, 7, 10, 16, 31, 128, 256, 511]
#     tile_s_list = [1, 2, 4, 8, 15, 13, 16, 30, 32, 64]
#     dtype_list = ["float32", "bfloat16", "float16"]
#     hidden_size_list = [64, 128]
#     cube_tile_shapes = [[128, 128], [128, 128], [128, 128]]
#     vector_tile_shapes_2d = [4, 128]
#     vector_tile_shapes = [4, 4, 256]

#     case_list = []
#     case_idx = 0

#     # 遍历 batch_size 和 seq_len 的所有组合
#     for i, batch_size in enumerate(batch_size_list):
#         for j, seq_len in enumerate(seq_len_list):
#             # tile_b 与 batch_size 索引 i 对应
#             tile_b = tile_b_list[i]

#             # tile_s 与 seq_len 索引 j 对应
#             tile_s = tile_s_list[j]

#             # 其他参数随机取值
#             dtype = random.choice(dtype_list)
#             hidden_size = random.choice(hidden_size_list)

#             # 构造 case
#             case = {
#                 "case_idx": case_idx,
#                 "case_name": f"test_view_assemble_{case_idx:03d}",
#                 "op_type": "test_assemble_3d",
#                 "batch_size": batch_size,
#                 "seq_len": seq_len,
#                 "hidden_size": hidden_size,
#                 "tile_b": tile_b,
#                 "tile_s": tile_s,
#                 "dtype": dtype,
#                 "cube_tile_shapes": cube_tile_shapes,
#                 "vector_tile_shapes_2d": vector_tile_shapes_2d,
#                 "vector_tile_shapes": vector_tile_shapes
#             }

#             case_list.append(case)
#             case_idx += 1

#     # 写入 JSON 文件
#     output_path = "view_assemble_test.json"
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(case_list, f, indent=4)

#     print(f"✅ 已生成 {len(case_list)} 个测试用例（按索引对应 tile_b/tile_s），保存至 {output_path}")

# # 调用函数
# generate_case()


# @pypto.jit
# def vec_cube_assemble_op(input_a_tensor, out_tensor):
#     pypto.set_vec_tile_shapes(2, 8)
#     offsets = [0, 0]
#     pypto.assemble(input_a_tensor, offsets, out_tensor)


@pypto.jit
def cube_vector_pypto(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: AssembleParams):
    tile_b = params.tile_b #4
    tile_s = params.tile_s #4
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) //  tile_b
    s_loop = (seq_len + tile_s - 1) // tile_s 
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx", unroll_list=[1, 2, 4]):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            # input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            # input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_a_view = pypto.view(input_tensor_a, [tile_b, tile_s, input_tensor_a.shape[-1]], [tile_b*b_idx, tile_s*s_idx, 0], 
                     valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b), pypto.min(tile_s, input_tensor_a.shape[1] -s_idx*tile_s), input_tensor_a.shape[-1]])
            input_b_view = pypto.view(input_tensor_b, [tile_b, tile_s, input_tensor_b.shape[-1]], [tile_b*b_idx, tile_s*s_idx, 0], 
                     valid_shape=[pypto.min(tile_b, input_tensor_b.shape[0]-b_idx*tile_b), pypto.min(tile_s, input_tensor_b.shape[1] -s_idx*tile_s), input_tensor_b.shape[-1]])
            
            print(input_a_view.shape, input_tensor_c.shape) # [tile_b, tile_s, hidden_size]
            # input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])
            # input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])
            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]], 
                          valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b) * pypto.min(tile_s, input_tensor_a.shape[1] -s_idx*tile_s), input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]],
                          valid_shape=[pypto.min(tile_b, input_tensor_b.shape[0]-b_idx*tile_b) * pypto.min(tile_s, input_tensor_b.shape[1] -s_idx*tile_s), input_b_view.shape[-1]])   
            matmul_out = pypto.matmul(input_b_view_2d, input_tensor_c, out_dtype=dtype)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            add_out = pypto.add(input_a_view_2d, input_b_view_2d)
            
            #[tile_b, params.hidden_size + params.hidden_size]->[64, 128]
            out_tmp = pypto.tensor([tile_b*tile_s, matmul_out.shape[-1] + add_out.shape[-1]], dtype=dtype)
            pypto.assemble(matmul_out, [0, 0], out_tmp)
            pypto.assemble(add_out, [0, matmul_out.shape[-1]], out_tmp)

            #[2*params.hidden_size, params.hidden_size] -> [128, 64]
            tmp_tensor_1 = pypto.full([out_tmp.shape[-1], output_tensor.shape[-1]] , 0.5, dtype=out_tmp.dtype)
            matmul_out_1 = pypto.matmul(out_tmp, tmp_tensor_1, out_dtype=out_tmp.dtype)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            # mamtul_out_1_3d = pypto.reshape(matmul_out_1, [tile_b, tile_s, matmul_out_1.shape[-1]])
            mamtul_out_1_3d = pypto.reshape(matmul_out_1, [tile_b, tile_s, matmul_out_1.shape[-1]], 
                         valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b), pypto.min(tile_s, input_tensor_a.shape[1] -s_idx*tile_s), matmul_out_1.shape[-1]])  

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = mamtul_out_1_3d       

@pypto.jit
def vector_cube_pypto(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: AssembleParams):
    tile_b = params.tile_b #4
    tile_s = params.tile_s #4
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) /  tile_b
    s_loop = (seq_len + tile_s - 1) / tile_s 
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx", unroll_list=[1, 2, 4]):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            
            print(input_a_view.shape, input_tensor_c.shape) # [tile_b, tile_s, hidden_size]
            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])
            matmul_out = pypto.matmul(input_b_view_2d, input_tensor_c, out_dtype=dtype)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            mul_out = pypto.mul(input_a_view_2d, input_b_view_2d)
            
            #[tile_b, params.hidden_size + params.hidden_size]->[64, 128]
            out_tmp = pypto.tensor([tile_b*tile_s, matmul_out.shape[-1] + mul_out.shape[-1]], dtype=dtype)
            pypto.assemble(mul_out, [0, 0], out_tmp)
            pypto.assemble(matmul_out, [0, mul_out.shape[-1]], out_tmp)

            #[2*params.hidden_size, params.hidden_size] -> [128, 64]
            tmp_tensor_1 = pypto.full([out_tmp.shape[-1], output_tensor.shape[-1]] , 1.5, dtype=out_tmp.dtype)
            matmul_out_1 = pypto.matmul(out_tmp, tmp_tensor_1, out_dtype=out_tmp.dtype)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            mamtul_out_1_3d = pypto.reshape(matmul_out_1, [tile_b, tile_s, matmul_out_1.shape[-1]])

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = mamtul_out_1_3d

@pypto.jit
def vec_vec_pypto(input_tensor_a, input_tensor_b, output_tensor, params: AssembleParams):
    tile_b = params.tile_b  #
    tile_s = params.tile_s
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) / tile_b
    s_loop = (seq_len + tile_s -1) / tile_s
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx", unroll_list=[1, 2, 4]):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            #input_a_view = pypto.view(input_tensor_a, [tile_b, input_tensor_a.shape[-1]], [tile_b*b_idx, 0], valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b), input_tensor_a.shape[-1]])
            #input_b_view = pypto.view(input_tensor_b, [tile_b, input_tensor_b.shape[-1]], [tile_b*b_idx, 0], valid_shape=[pypto.min(tile_b, input_tensor_b.shape[0]-b_idx*tile_b), input_tensor_b.shape[-1]])
            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]

            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            add_out = pypto.add(input_a_view_2d, input_b_view_2d)  #[tile_b*tile_s, hidden_size]
            mul_out = pypto.mul(input_a_view_2d, input_b_view_2d)  #[tile_b*tile_s, hidden_size]
            assemble_tmp = pypto.tensor([tile_b*tile_s, add_out.shape[-1] + mul_out.shape[-1]], dtype=dtype)
            pypto.assemble(add_out, [0, 0], assemble_tmp)
            pypto.assemble(mul_out, [0, add_out.shape[-1]], assemble_tmp)

            # view 出一块跟前面shape不一样大小: [tile_b*tile_s, hidden_size*0.5] + [tile_b*tile_s, hidden_size*1.5]
            tmp_view_1 = pypto.view(assemble_tmp, [tile_b*tile_s, input_a_view_2d.shape[-1] // 2], [0, 0])
            tmp_view_2 = pypto.view(assemble_tmp, [tile_b*tile_s, assemble_tmp.shape[-1] - input_a_view_2d.shape[-1] // 2], [0, input_a_view_2d.shape[-1] // 2])
            print(f"tmp_view_1.shape:{tmp_view_1.shape}")
            print(f"tmp_view_2.shape:{tmp_view_2.shape}")
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            mul_out1 = pypto.mul(tmp_view_1, 1.537)
            mul_out2 = pypto.mul(tmp_view_2, 2.538)
        
            #[tile_b, 2*hidden_size]
            concat_out_2 = pypto.concat([mul_out1, mul_out2], dim=-1)

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            concat_out_2_3d = pypto.reshape(concat_out_2, [tile_b, tile_s, concat_out_2.shape[-1]])
            print(f"concat_out_2_3d.shape:{concat_out_2_3d.shape}")

            #[tile_b, hiddensize]， output_tensor.shape[-1] * 2   ||  123行返回一半的写法需要继续看下
            #output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = concat_out_2_3d[:, :, 0: concat_out_2_3d.shape[-1] // 2]
            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = concat_out_2_3d
        

@pypto.jit
def cube_cube_pypto(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: AssembleParams):
    tile_b = params.tile_b  # 32
    tile_s = params.tile_s
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) / tile_b
    s_loop = (seq_len + tile_s -1) / tile_s
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop , name="Loop_B", idx_name="b_idx" , unroll_list=[1, 2, 4]):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            
            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])
            
            matmul_out_1 = pypto.matmul(input_a_view_2d, input_tensor_c, out_dtype=dtype)
            matmul_out_2 = pypto.matmul(input_b_view_2d, input_tensor_c, out_dtype=dtype)
            # assemble
            assemble_tmp = pypto.tensor([tile_b*tile_s, matmul_out_1.shape[-1] + matmul_out_2.shape[-1]], dtype=dtype)
            pypto.assemble(matmul_out_1, [0, 0], assemble_tmp)
            pypto.assemble(matmul_out_2, [0, matmul_out_1.shape[-1]], assemble_tmp)
            print(f"assemble_tmp.shape:{assemble_tmp.shape}")

            # view出来不对称tensor做add运算
            tmp_view_1 = pypto.view(assemble_tmp, [tile_b*tile_s, input_a_view_2d.shape[-1] *3 // 2], [0, 0])
            tmp_view_2 = pypto.view(assemble_tmp, [tile_b*tile_s, assemble_tmp.shape[-1] - input_a_view_2d.shape[-1] *3 // 2], [0, input_a_view_2d.shape[-1] *3 // 2])
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            add_out_1 = pypto.add(tmp_view_1, 1)
            add_out_2 = pypto.add(tmp_view_2, 2)
            concat_out_2 = pypto.concat([add_out_1, add_out_2], dim=-1)

            sum_out = pypto.sum(concat_out_2, dim=-1, keepdim=True)
            div_out = pypto.div(input_a_view_2d, sum_out)
            sub_out = pypto.sub(div_out, input_b_view_2d)

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            sub_out_3d = pypto.reshape(sub_out, [tile_b, tile_s, sub_out.shape[-1]])

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s : (s_idx+1)*tile_s, :] = sub_out_3d
         

def cube_cube_torch(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: AssembleParams):
    input_tensor_a_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_b_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])
    
    matmul_out_1 = torch.matmul(input_tensor_a_2d, input_tensor_c)
    matmul_out_2 = torch.matmul(input_tensor_b_2d, input_tensor_c)
    concat_out= torch.concat([matmul_out_1, matmul_out_2], dim=-1)

    tmp_view_1 = concat_out[: , 0: input_tensor_a_2d.shape[-1] * 3 // 2]
    tmp_view_2 = concat_out[: , input_tensor_a_2d.shape[-1] * 3 // 2: input_tensor_a_2d.shape[-1] * 2]
    add_out_1 = torch.add(tmp_view_1, 1)
    add_out_2 = torch.add(tmp_view_2, 2)
    concat_out_2 = torch.concat([add_out_1, add_out_2], dim=-1)

    sum_out = torch.sum(concat_out_2, dim=-1, keepdim=True)
    div_out = torch.div(input_tensor_a_2d, sum_out)
    sub_out = torch.sub(div_out, input_tensor_b_2d)
    output_tensor =  sub_out.reshape(output_tensor.shape)
    return output_tensor


def vec_vec_torch(input_tensor_a, input_tensor_b, output_tensor, params: AssembleParams):
    input_tensor_a_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_b_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])

    add_out = torch.add(input_tensor_a_2d, input_tensor_b_2d)
    mul_out = torch.mul(input_tensor_a_2d, input_tensor_b_2d)
    concat_out = torch.concat([add_out, mul_out], dim=-1)
    tmp_view_1 = concat_out[:, 0: input_tensor_a_2d.shape[-1] // 2]
    tmp_view_2 = concat_out[:, input_tensor_a_2d.shape[-1] // 2: input_tensor_a_2d.shape[-1] * 2]
    mul_out_1 = torch.mul(tmp_view_1, 1.537)
    mul_out_2 = torch.mul(tmp_view_2, 2.538)
    concat_out_2 = torch.concat([mul_out_1, mul_out_2], dim=-1)
    #output_tensor = concat_out_2[:, 0: concat_out_2.shape[-1] // 2]
    output_tensor = concat_out_2.reshape(output_tensor.shape)
    return output_tensor


def cube_vector_torch(input_tensor_a, input_tensor_b, input_tensor_c, params: AssembleParams):
    # -1表示自动推断该维度的大小
    input_tensor_a_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_b_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])

    matmul_out = torch.matmul(input_tensor_b_2d, input_tensor_c)     
    add_out = torch.add(input_tensor_a_2d, input_tensor_b_2d)   
    out_tmp = torch.concat([matmul_out, add_out], dim=-1)            
    out_tmp = out_tmp.to(input_tensor_a.device)
    full_out = torch.full([out_tmp.shape[-1], params.hidden_size], 0.5, dtype=torch.float32, device=out_tmp.device) 
    output_tensor = torch.matmul(out_tmp, full_out)   
    output_tensor = output_tensor.reshape(input_tensor_a.shape)  
    return output_tensor

def vector_cube_torch(input_tensor_a, input_tensor_b, input_tensor_c, params: AssembleParams):
    # -1表示自动推断该维度的大小
    input_tensor_a_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_b_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])

    matmul_out = torch.matmul(input_tensor_b_2d, input_tensor_c)     
    mul_out = torch.mul(input_tensor_a_2d, input_tensor_b_2d)   
    out_tmp = torch.concat([mul_out, matmul_out], dim=-1)            
    out_tmp = out_tmp.to(input_tensor_a.device)
    full_out = torch.full([out_tmp.shape[-1], params.hidden_size], 1.5, dtype=torch.float32, device=out_tmp.device) 
    output_tensor = torch.matmul(out_tmp, full_out)   
    output_tensor = output_tensor.reshape(input_tensor_a.shape)  
    return output_tensor

@pypto.jit
def ddr_cube_pypto(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: AssembleParams):
    tile_b = params.tile_b
    tile_s = params.tile_s
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) / tile_b
    s_loop = (seq_len + tile_s -1) / tile_s
    dtype = input_tensor_b.dtype
    
    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx"):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)   
            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]

            input_a_view_2d = pypto.reshape(input_a_view, [tile_b * tile_s, input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b * tile_s, input_b_view.shape[-1]])
            
            matmul_out =  pypto.matmul(input_b_view_2d, input_tensor_c, out_dtype=dtype)

            out_tmp = pypto.tensor([tile_b*tile_s, matmul_out.shape[-1] + input_a_view_2d.shape[-1]], dtype=dtype)
            pypto.assemble(input_a_view_2d, [0, 0], out_tmp)
            pypto.assemble(matmul_out, [0, input_a_view_2d.shape[-1]], out_tmp)
            print(f"input_a_view_2d.shape:{input_a_view_2d.shape}")
            print(f"matmul_out.shape:{matmul_out.shape}")
            print(f"out_tmp.shape:{out_tmp.shape}")

            # 二分法定位失败，下面没问题，是上面assemble失败了
            tmp_tensor_1 = pypto.full([out_tmp.shape[-1], output_tensor.shape[-1]] , 0.5, dtype=out_tmp.dtype)
            matmul_out_1 = pypto.matmul(out_tmp, tmp_tensor_1, out_dtype=out_tmp.dtype)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            exp_out = pypto.exp(matmul_out_1)

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            exp_out_3d = pypto.reshape(exp_out, [tile_b, tile_s, exp_out.shape[-1]])

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = exp_out_3d

def ddr_cube_torch(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: AssembleParams):
    input_tensor_a_2d =  input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_b_2d =  input_tensor_b.reshape(-1, input_tensor_b.shape[-1])
    matmul_out = torch.matmul(input_tensor_b_2d, input_tensor_c)
    out_tmp = torch.concat([input_tensor_a_2d, matmul_out], dim=-1)
    out_tmp = out_tmp.to(input_tensor_a.device)         
    full_out = torch.full([out_tmp.shape[-1], params.hidden_size], 0.5, dtype=torch.float32, device=out_tmp.device) 
    matmul_out_2 = torch.matmul(out_tmp, full_out)   
    exp_out = torch.exp(matmul_out_2)
    output_tensor = exp_out.reshape(input_tensor_a.shape)
    return output_tensor
    

@pypto.jit
def ddr_vector_pypto(input_tensor_a, input_tensor_b, output_tensor, params: AssembleParams):
    tile_b = params.tile_b
    tile_s = params.tile_s
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) / tile_b
    s_loop = (seq_len + tile_s -1) / tile_s
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx" , unroll_list=[1, 2, 4]):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)

            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx + 1)*tile_b, s_idx*tile_s : (s_idx + 1)*tile_s, :]
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx + 1)*tile_b, s_idx*tile_s : (s_idx + 1)*tile_s, :]

            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            mul_out = pypto.mul(input_a_view_2d, input_b_view_2d)
            
            out_tmp = pypto.tensor([tile_b*tile_s, mul_out.shape[-1] + input_a_view_2d.shape[-1]], dtype=dtype)
            pypto.assemble(input_a_view_2d, [0, 0], out_tmp)
            pypto.assemble(mul_out, [0, input_a_view_2d.shape[-1]], out_tmp)

            tmp_tensor_1 = pypto.full([out_tmp.shape[-1], output_tensor.shape[-1]] , 2.0, dtype=out_tmp.dtype)
            matmul_out_1 = pypto.matmul(out_tmp, tmp_tensor_1, out_dtype=out_tmp.dtype)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            sigmoid_out = pypto.sigmoid(matmul_out_1)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes)
            sigmoid_out_3d = pypto.reshape(sigmoid_out, [tile_b, tile_s, sigmoid_out.shape[-1]])

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s : (s_idx+1)*tile_s, :] = sigmoid_out_3d

def ddr_vector_torch(input_tensor_a, input_tensor_b, output_tensor, params: AssembleParams):
    input_tensor_a_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_b_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])
    mul_out = torch.mul(input_tensor_a_2d, input_tensor_b_2d)
    out_tmp = torch.concat([input_tensor_a_2d, mul_out], dim=-1)           
    out_tmp = out_tmp.to(input_tensor_a.device)   
    full_out = torch.full([out_tmp.shape[-1], params.hidden_size], 2.0, dtype=torch.float32, device=out_tmp.device) 
    matmul_out_1 = torch.matmul(out_tmp, full_out) 
    sigmoid_out = torch.sigmoid(matmul_out_1)  
    output_tensor = sigmoid_out.reshape(input_tensor_a.shape)
    return output_tensor

def test_assemble(device_id):
    torch_npu.npu.set_device(device_id)

    params = AssembleParams(
        batch_size=1024,
        seq_len=127,
        hidden_size=64,
        tile_b=511,
        tile_s=64,
        dtype="float32",
        cube_tile_shapes=[[128, 128], [128, 128], [128, 128]],
        vector_tile_shapes_2d=[4, 128],
        vector_tile_shapes=[4, 4, 256]
    )

    #[32,32,64] 、[32,32,64] 、 [64, 64] 、 [32,32, 64]
    tensor_a = torch.randn(params.batch_size, params.seq_len, params.hidden_size, dtype=torch.float32).to("npu")
    tensor_b = torch.randn(params.batch_size, params.seq_len, params.hidden_size, dtype=torch.float32).to("npu")
    tensor_c = torch.randn(params.hidden_size, params.hidden_size, dtype=torch.float32).to("npu")
    tensor_out = torch.zeros(params.batch_size, params.seq_len, params.hidden_size, dtype=torch.float32).to("npu")
    tensor_out_1 = torch.zeros(params.batch_size, params.seq_len, params.hidden_size * 2, dtype=torch.float32).to("npu")
    
    tensor_a_pto = pypto.from_torch(tensor_a, dynamic_axis=[0, 1], name="tensor_a")
    tensor_b_pto = pypto.from_torch(tensor_b, dynamic_axis=[0, 1], name="tensor_b")
    tensor_c_pto = pypto.from_torch(tensor_c, dynamic_axis=None, name="tensor_c")
    tensor_out_pto = pypto.from_torch(tensor_out, dynamic_axis=[0, 1], name="tensor_out")
    tensor_out_pto_1 = pypto.from_torch(tensor_out_1, dynamic_axis=[0, 1], name="tensor_out_1")

    
    output_pytorch = cube_vector_torch(tensor_a, tensor_b, tensor_c, params)
    output_golden = cube_vector_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_c.to(torch.float32).cpu(), params)
    #output_pytorch = vector_cube_torch(tensor_a, tensor_b, tensor_c, params)
    #output_golden = vector_cube_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_c.to(torch.float32).cpu(), params)
    #output_pytorch = vec_vec_torch(tensor_a, tensor_b , tensor_out_1 , params)
    #output_golden = vec_vec_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu() , tensor_out_1.to(torch.float32).cpu() , params)
    #output_pytorch = cube_cube_torch(tensor_a, tensor_b, tensor_c , tensor_out , params)
    #output_golden = cube_cube_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_c.to(torch.float32).cpu() , tensor_out.to(torch.float32).cpu() , params)
    
    #output_pytorch = ddr_cube_torch(tensor_a, tensor_b, tensor_c , tensor_out, params)
    #output_golden = ddr_cube_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_c.to(torch.float32).cpu() , tensor_out_1.to(torch.float32).cpu() , params)
    #output_pytorch = ddr_vector_torch(tensor_a, tensor_b , tensor_out , params)
    #output_golden = ddr_vector_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_out.to(torch.float32).cpu() , params)
    torch_npu.npu.synchronize()
    print(output_pytorch)
    print(output_golden)

    
    cube_vector_pypto(tensor_a_pto, tensor_b_pto, tensor_c_pto, tensor_out_pto, params)
    #vector_cube_pypto(tensor_a_pto, tensor_b_pto, tensor_c_pto, tensor_out_pto, params)
    #vec_vec_pypto(tensor_a_pto, tensor_b_pto, tensor_out_pto_1, params)
    #cube_cube_pypto(tensor_a_pto, tensor_b_pto, tensor_c_pto, tensor_out_pto, params)
    #ddr_cube_pypto(tensor_a_pto, tensor_b_pto, tensor_c_pto, tensor_out_pto, params)
    #ddr_vector_pypto(tensor_a_pto, tensor_b_pto, tensor_out_pto, params)
    torch_npu.npu.synchronize()

    results_list = []
    # print(tensor_out)
    # precision_result = precision_compare_triple_data(convert_torch_to_numpy(tensor_out), convert_torch_to_numpy(output_pytorch),
    #                                                convert_torch_to_numpy(output_golden), dtype="fp32")
    # print(precision_result)
    # results_list.append(precision_result)
    return results_list



if __name__ == "__main__":
    #pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    test_assemble(0)