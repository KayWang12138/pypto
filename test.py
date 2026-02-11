# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
import pypto 
import torch 
import torch_npu
from libs.generate_inputs import generate_inputs
from libs.tools import convert_torch_to_numpy, get_torch_dtype, get_broadcast_offsets, generate_broadcast_info
from libs.precision_compare import precision_compare_triple_data

@dataclass
class CompositeParams:
   batch_size: int
   seq_len: int 
   hidden_size: int
   tile_b: int
   tile_s: int
   input_datarange: list
   dtype: str
   cube_tile_shapes: list
   vector_tile_shapes_2d: list
   vector_tile_shapes_3d: list

verify_options = {
    "enable_pass_verify": True,
    # "pass_verify_save_tensor": True,
}
@pypto.jit(verify_options=verify_options)
def special_view_opcomposite_pto(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: CompositeParams):
    tile_b = params.tile_b #4
    tile_s = params.tile_s #4
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) /  tile_b
    s_loop = (seq_len + tile_s - 1) / tile_s 
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx"):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_3d)
            
            #scopeid = 1
            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            # [16, 64]
            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])          
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)
            # [8, 32] [8,32]  [8, 32] [8, 32]    
            input_tensor_a_1 = pypto.view(input_a_view_2d, [input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] // 2], [0, 0])
            input_tensor_a_2 = pypto.view(input_a_view_2d, [input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] - input_a_view_2d.shape[-1] // 2], [0, input_a_view_2d.shape[-1] // 2 ] )
            input_tensor_a_3 = pypto.view(input_a_view_2d, [input_a_view_2d.shape[0] - input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] // 2], [input_a_view_2d.shape[0] // 2 , 0])
            input_tensor_a_4 = pypto.view(input_a_view_2d, [input_a_view_2d.shape[0] - input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] - input_a_view_2d.shape[-1] // 2], [input_a_view_2d.shape[0] // 2 , input_a_view_2d.shape[-1] // 2 ])
            
            #[8, 64] [8, 64]
            tensor_a_tmp_dim1_1 = pypto.concat([input_tensor_a_1, input_tensor_a_2], dim=-1)
            tensor_a_tmp_dim1_2 = pypto.concat([input_tensor_a_3, input_tensor_a_4], dim=-1)
            
            # 相乘之后就有精度问题，去掉mul精度正确
            tmp_dim1_a_mul_1 = pypto.mul(tensor_a_tmp_dim1_1, 3.0)
            tmp_dim1_a_mul_2 = pypto.mul(tensor_a_tmp_dim1_2, 5.0)
            
            # [16, 64]
            concat_tmp_a_1 = pypto.concat([tmp_dim1_a_mul_1, tmp_dim1_a_mul_2], dim=0)
            # [8, 64] [8, 64]
            # concat_tmp_a_view_1 = concat_tmp_a_1[:concat_tmp_a_1.shape[0] // 2, :]
            # concat_tmp_a_view_2 = concat_tmp_a_1[concat_tmp_a_1.shape[0] // 2: concat_tmp_a_1.shape[0], :]
            
            #scopeid = 2
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_3d)
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d) 
            input_tensor_b_1 = pypto.view(input_b_view_2d, [input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] // 2], [0, 0])
            input_tensor_b_2 = pypto.view(input_b_view_2d, [input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] - input_b_view_2d.shape[-1] // 2], [0, input_b_view_2d.shape[-1] // 2 ] )
            input_tensor_b_3 = pypto.view(input_b_view_2d, [input_b_view_2d.shape[0] - input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] // 2], [input_b_view_2d.shape[0] // 2 , 0])
            input_tensor_b_4 = pypto.view(input_b_view_2d, [input_b_view_2d.shape[0] - input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] - input_b_view_2d.shape[-1] // 2], [input_b_view_2d.shape[0] // 2 , input_b_view_2d.shape[-1] // 2 ])
            
            #[8, 64] [8, 64]
            tensor_b_tmp_dim1_1 = pypto.concat([input_tensor_b_1, input_tensor_b_2], dim=-1)
            tensor_b_tmp_dim1_2 = pypto.concat([input_tensor_b_3, input_tensor_b_4], dim=-1)
            
            # 相乘之后就有精度问题，去掉mul精度正确
            tmp_dim1_b_mul_1 = pypto.mul(tensor_b_tmp_dim1_1, 3.0)
            tmp_dim1_b_mul_2 = pypto.mul(tensor_b_tmp_dim1_2, 5.0)
            # [16, 64]
            concat_b_tmp_1 = pypto.concat([tmp_dim1_b_mul_1, tmp_dim1_b_mul_2], dim=0)
 
            # [8, 64] [8, 64]
            # concat_tmp_b_view_1 = concat_b_tmp_1[:concat_b_tmp_1.shape[0] // 2, :]
            # concat_tmp_b_view_2 = concat_b_tmp_1[concat_b_tmp_1.shape[0] // 2: concat_b_tmp_1.shape[0], :]

            #matmul，设置scopid =3     [8,64] [8,64] [8,64] [8,64] 
            # matmul_tmp_a_1 = pypto.matmul(concat_tmp_a_view_1, input_tensor_c, out_dtype=dtype)
            # matmul_tmp_a_2 = pypto.matmul(concat_tmp_a_view_2, input_tensor_c, out_dtype=dtype)
            # matmul_tmp_b_1 = pypto.matmul(concat_tmp_b_view_1, input_tensor_c, out_dtype=dtype)
            # matmul_tmp_b_2 = pypto.matmul(concat_tmp_b_view_2, input_tensor_c, out_dtype=dtype)
            
            # 设置scopid = 4     [8, 128]  [8, 128] [16, 128]
            # concat_f_a = pypto.concat([concat_tmp_a_view_1, concat_tmp_a_view_2], dim=-1)
            # concat_f_b = pypto.concat([concat_tmp_b_view_1, concat_tmp_b_view_2], dim=-1)
            # concat_final = pypto.concat([concat_f_a, concat_f_b], dim=0)
            concat_final = pypto.concat([concat_tmp_a_1, concat_b_tmp_1], dim=-1)
            print(f"concat_final.shape:{concat_final.shape}")

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_3d)
            concat_final_3d = pypto.reshape(concat_final, [tile_b, tile_s, concat_final.shape[-1]])

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = concat_final_3d
            

def special_view_opcomposite_torch(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: CompositeParams):
    input_a_view_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_tensor_a_1 = input_a_view_2d[:input_a_view_2d.shape[0] // 2, :input_a_view_2d.shape[-1] // 2]
    input_tensor_a_2 = input_a_view_2d[:input_a_view_2d.shape[0] // 2, input_a_view_2d.shape[-1] // 2 : input_a_view_2d.shape[-1]]
    input_tensor_a_3 = input_a_view_2d[input_a_view_2d.shape[0] // 2 : input_a_view_2d.shape[0], :input_a_view_2d.shape[-1] // 2]
    input_tensor_a_4 = input_a_view_2d[input_a_view_2d.shape[0] // 2 : input_a_view_2d.shape[0], input_a_view_2d.shape[-1] // 2 : input_a_view_2d.shape[-1]]

    tensor_a_tmp_dim1_1 = torch.concat([input_tensor_a_1, input_tensor_a_2], dim=-1)
    tensor_a_tmp_dim1_2 = torch.concat([input_tensor_a_3, input_tensor_a_4], dim=-1)
    
    #相乘之后就有精度问题，去掉mul精度正确
    tmp_dim1_a_mul_1 = torch.mul(tensor_a_tmp_dim1_1, 3.0)
    tmp_dim1_a_mul_2 = torch.mul(tensor_a_tmp_dim1_2, 5.0)
    concat_tmp_a_1 = torch.concat([tmp_dim1_a_mul_1, tmp_dim1_a_mul_2], dim=0)

    # concat_tmp_a_view_1 = concat_tmp_a_1[:concat_tmp_a_1.shape[0] // 2, :]
    # concat_tmp_a_view_2 = concat_tmp_a_1[concat_tmp_a_1.shape[0] // 2: concat_tmp_a_1.shape[0], :]

    input_b_view_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1])
    input_tensor_b_1 = input_b_view_2d[:input_b_view_2d.shape[0] // 2, : input_b_view_2d.shape[-1] // 2]
    input_tensor_b_2 = input_b_view_2d[:input_b_view_2d.shape[0] // 2, input_b_view_2d.shape[-1] // 2 : input_b_view_2d.shape[-1]]
    input_tensor_b_3 = input_b_view_2d[input_b_view_2d.shape[0] // 2 : input_b_view_2d.shape[0], : input_b_view_2d.shape[-1] // 2]
    input_tensor_b_4 = input_b_view_2d[input_b_view_2d.shape[0] // 2 : input_b_view_2d.shape[0], input_b_view_2d.shape[-1] // 2 : input_b_view_2d.shape[-1]]

    tensor_b_tmp_dim1_1 = torch.concat([input_tensor_b_1, input_tensor_b_2], dim=-1)
    tensor_b_tmp_dim1_2 = torch.concat([input_tensor_b_3, input_tensor_b_4], dim=-1)
    
    #相乘之后就有精度问题，去掉mul精度正确
    tmp_dim1_b_mul_1 = torch.mul(tensor_b_tmp_dim1_1, 3.0)
    tmp_dim1_b_mul_2 = torch.mul(tensor_b_tmp_dim1_2, 5.0)
    concat_tmp_b_1 = torch.concat([tmp_dim1_b_mul_1, tmp_dim1_b_mul_2], dim=0)

    # concat_tmp_b_view_1 = concat_tmp_b_1[:concat_tmp_b_1.shape[0] // 2, :]
    # concat_tmp_b_view_2 = concat_tmp_b_1[concat_tmp_b_1.shape[0] // 2: concat_tmp_b_1.shape[0], :]
    
    # matmul_tmp_a_1 = torch.matmul(concat_tmp_a_view_1, input_tensor_c)
    # matmul_tmp_a_2 = torch.matmul(concat_tmp_a_view_2, input_tensor_c)
    # matmul_tmp_b_1 = torch.matmul(concat_tmp_b_view_1, input_tensor_c)
    # matmul_tmp_b_2 = torch.matmul(concat_tmp_b_view_2, input_tensor_c)

    # concat_f_a = torch.concat([concat_tmp_a_view_1, concat_tmp_a_view_2], dim=-1)
    # concat_f_b = torch.concat([concat_tmp_b_view_1, concat_tmp_b_view_2], dim=-1)
    # concat_final = torch.concat([concat_f_a, concat_f_b], dim=0)

    concat_final = torch.concat([concat_tmp_a_1, concat_tmp_b_1], dim=-1)
    print(f"torch_concat_final.shape:{concat_final.shape}")
    output_tensor = concat_final.reshape(output_tensor.shape)
    return output_tensor

            
@pypto.jit
def vec_vec_composite_pto(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: CompositeParams):
    tile_b = params.tile_b #4
    tile_s = params.tile_s #4
    batch_size, seq_len = input_tensor_a.shape[:2]
    b_loop = (batch_size + tile_b -1) /  tile_b
    s_loop = (seq_len + tile_s - 1) / tile_s 
    dtype = input_tensor_b.dtype

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx"):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_3d)
            input_a_view = input_tensor_a[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            input_b_view = input_tensor_b[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :]
            
            #pypto.set_pass_options(sg_set_scope=1)
            input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]])
            input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]])

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_2d)          
            input_tensor_a_view_0 = pypto.view(input_a_view_2d, [input_a_view_2d.shape[0], input_a_view_2d.shape[-1] // 3], [0, 0])
            input_tensor_a_view_1 = pypto.view(input_a_view_2d, [input_a_view_2d.shape[0], input_a_view_2d.shape[-1] - input_a_view_2d.shape[-1] // 3], [0, input_a_view_2d.shape[-1] // 3 ] )
            
            input_tensor_b_view_0 = pypto.view(input_b_view_2d, [input_b_view_2d.shape[0], input_b_view_2d.shape[-1] // 3], [0, 0] )
            input_tensor_b_view_1 = pypto.view(input_b_view_2d, [input_b_view_2d.shape[0], input_b_view_2d.shape[-1] - input_b_view_2d.shape[-1] // 3], [0, input_b_view_2d.shape[-1] // 3 ] )
            #pypto.set_pass_options(sg_set_scope=-1)

            #tensor_a view 1/3块 和tensor_b view 1/3块 相加、tensor_a view 2/3块 和tensor_b view 2/3块 相乘
            #pypto.set_pass_options(sg_set_scope=2)
            temp_ab_view_0_add = pypto.add(input_tensor_a_view_0, input_tensor_b_view_0)
            temp_ab_view_1_mul = pypto.mul(input_tensor_a_view_1, input_tensor_b_view_1)
            #pypto.set_pass_options(sg_set_scope=-1)

            out_tmp_0 = pypto.tensor([tile_b*tile_s, temp_ab_view_0_add.shape[-1] + temp_ab_view_1_mul.shape[-1]], dtype=dtype)
            pypto.assemble(temp_ab_view_0_add, [0, 0] , out_tmp_0)
            pypto.assemble(temp_ab_view_1_mul, [0, temp_ab_view_0_add.shape[-1]] , out_tmp_0)
            print(f"out_tmp_0.shape:{out_tmp_0.shape}")
            
            #The aicore execution is abnormal报错问题：出在pypto.exp(out_tmp_0)，加上之后就有问题，去掉就精度正确
            #out_tmp_exp = pypto.exp(out_tmp_0)
            pypto.set_cube_tile_shapes(*params.cube_tile_shapes)
            tmp_matmul_out_0 = pypto.matmul(out_tmp_0, input_tensor_c, out_dtype=dtype)
            tmp_cos_out = pypto.cos(tmp_matmul_out_0)

            #pypto.set_pass_options(sg_set_scope=3)
            input_a_view_2d_sqrt = pypto.sqrt(pypto.abs(input_a_view_2d))
            input_b_view_2d_rsqrt = pypto.rsqrt(pypto.abs(input_b_view_2d))
            tmp_div_out = pypto.div(input_a_view_2d_sqrt, input_b_view_2d_rsqrt)
            tmp_sin_out = pypto.sin(tmp_div_out)
            #pypto.set_pass_options(sg_set_scope=-1)

            #pypto.set_pass_options(sg_set_scope=4)
            tmp_sub_out = pypto.sub(tmp_cos_out, tmp_sin_out)
            tmp_softmax_out = pypto.softmax(tmp_sub_out, dim=-1)
            tmp_sigmod_out = pypto.sigmoid(tmp_softmax_out)
            #pypto.set_pass_options(sg_set_scope=-1)

            matmul_out_1 = pypto.matmul(tmp_sigmod_out, input_tensor_c, out_dtype=dtype)
            print(f"matmul_out_1.shape:{matmul_out_1.shape}")

            pypto.set_vec_tile_shapes(*params.vector_tile_shapes_3d)
            mamtul_out_1_3d = pypto.reshape(matmul_out_1, [tile_b, tile_s, matmul_out_1.shape[-1]])

            output_tensor[b_idx*tile_b: (b_idx+1)*tile_b, s_idx*tile_s: (s_idx+1)*tile_s, :] = mamtul_out_1_3d



def vec_vec_composite_torch(input_tensor_a, input_tensor_b, input_tensor_c, output_tensor, params: CompositeParams):
    input_a_view_2d = input_tensor_a.reshape(-1, input_tensor_a.shape[-1])
    input_b_view_2d = input_tensor_b.reshape(-1, input_tensor_b.shape[-1]) 

    input_tensor_a_view_0 = input_a_view_2d[:, 0: input_a_view_2d.shape[-1] // 3] 
    input_tensor_a_view_1 = input_a_view_2d[:, input_a_view_2d.shape[-1] // 3 : input_a_view_2d.shape[-1]] 
    input_tensor_b_view_0 = input_b_view_2d[:, 0: input_b_view_2d.shape[-1] // 3] 
    input_tensor_b_view_1 = input_b_view_2d[:, input_b_view_2d.shape[-1] // 3 : input_b_view_2d.shape[-1]] 

    temp_ab_view_0_add = torch.add(input_tensor_a_view_0, input_tensor_b_view_0)
    temp_ab_view_1_mul = torch.mul(input_tensor_a_view_1, input_tensor_b_view_1)   
    out_tmp_0 = torch.concat([temp_ab_view_0_add, temp_ab_view_1_mul], dim=-1)
    #print(f"out_tmp_0:{out_tmp_0}")  

    #pypto也注释掉torch.exp(out_tmp_0)，定位The aicore execution is abnormal报错问题
    #out_tmp_exp = torch.exp(out_tmp_0)
    tmp_matmul_out_0 = torch.matmul(out_tmp_0, input_tensor_c)
    tmp_cos_out = torch.cos(tmp_matmul_out_0) 

    input_a_view_2d_sqrt = torch.sqrt(torch.abs(input_a_view_2d))
    input_b_view_2d_rsqrt = torch.rsqrt(torch.abs(input_b_view_2d)) 

    tmp_div_out = torch.div(input_a_view_2d_sqrt, input_b_view_2d_rsqrt)
    tmp_sin_out = torch.sin(tmp_div_out)  

    tmp_sub_out = torch.sub(tmp_cos_out, tmp_sin_out)
    tmp_softmax_out = torch.softmax(tmp_sub_out, dim=-1)
    tmp_sigmod_out = torch.sigmoid(tmp_softmax_out)
    #print(f"tmp_sigmod_out:{tmp_sigmod_out}")

    matmul_out_1 = torch.matmul(tmp_sigmod_out, input_tensor_c)
    output_tensor = matmul_out_1.reshape(input_tensor_a.shape)
    
    return output_tensor


def composite_graph(device_id):
    torch_npu.npu.set_device(device_id)
    params = CompositeParams(
        batch_size=32,
        seq_len=32,
        hidden_size=64,
        tile_b=4,
        tile_s=4,
        input_datarange=["0_0.01_normal_normal", "-1_1_normal_normal"],
        dtype="float32",
        cube_tile_shapes=[[128, 128], [128, 128], [128, 128]],
        vector_tile_shapes_2d=[4, 64],
        vector_tile_shapes_3d=[4, 4, 128]
    )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if isinstance(params.dtype, str):
        dtype = dtype_map.get(params.dtype.lower())
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")


    tensor_a = generate_inputs([params.batch_size, params.seq_len, params.hidden_size], params.dtype, params.input_datarange[0]).npu()
    tensor_b = generate_inputs([params.batch_size, params.seq_len, params.hidden_size], params.dtype, params.input_datarange[0]).npu()
    tensor_c = generate_inputs([params.hidden_size, params.hidden_size], params.dtype, params.input_datarange[0]).npu()
    tensor_out = torch.zeros(params.batch_size, params.seq_len, params.hidden_size, dtype=dtype).npu()
    tensor_out_1 = torch.zeros(params.batch_size, params.seq_len, params.hidden_size * 2, dtype=dtype).npu()

    tensor_a_pto = pypto.from_torch(tensor_a, dynamic_axis=[0, 1], name="tensor_a")
    tensor_b_pto = pypto.from_torch(tensor_b, dynamic_axis=[0, 1], name="tensor_b")
    tensor_c_pto = pypto.from_torch(tensor_c, dynamic_axis=None, name="tensor_c")
    tensor_out_pto = pypto.from_torch(tensor_out, dynamic_axis=[0, 1], name="tensor_out")
    tensor_out_pto_1 = pypto.from_torch(tensor_out_1, dynamic_axis=[0, 1], name="tensor_out_1")

    #output_pytorch = vec_vec_composite_torch(tensor_a, tensor_b, tensor_c, tensor_out, params)
    #output_golden = vec_vec_composite_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_c.to(torch.float32).cpu(), tensor_out.to(torch.float32).cpu(), params)
    output_pytorch = special_view_opcomposite_torch(tensor_a, tensor_b, tensor_c, tensor_out_1, params)
    output_golden = special_view_opcomposite_torch(tensor_a.to(torch.float32).cpu(), tensor_b.to(torch.float32).cpu(), tensor_c.to(torch.float32).cpu(), tensor_out_1.to(torch.float32).cpu(), params)
    torch_npu.npu.synchronize()
    print(output_pytorch)
    print(output_golden)
    print(f"output_golden.shape:{output_golden.shape}")

    #vec_vec_composite_pto(tensor_a_pto, tensor_b_pto, tensor_c_pto, tensor_out_pto, params)
    special_view_opcomposite_pto(tensor_a_pto, tensor_b_pto, tensor_c_pto, tensor_out_pto_1, params)
    torch_npu.npu.synchronize()
    print(tensor_out_1)
    print(f"tensor_out_1.shape:{tensor_out_1.shape}")

    results_list = []
    precision_result = precision_compare_triple_data(convert_torch_to_numpy(tensor_out_1), convert_torch_to_numpy(output_pytorch),
                                                   convert_torch_to_numpy(output_golden), dtype="fp32")
    print(precision_result)
    results_list.append(precision_result)
    return results_list


if __name__ == "__main__":
    #pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
    composite_graph(13)





