import pypto
import torch
import torch_npu
from dataclasses import dataclass
import numpy as np
import logging

from flatbuffers.packer import int64
from pypto.symbolic_scalar import SymInt
from typing import List, Optional

# -------------------------- 1. 手动填入你的表格用例数据（无需读文件） --------------------------
CASE_DATA = {
    "case_name": "Gather_L0__105860_108_dense",
    "input_shape": "[114688,512],[105860,108]",
    "input_dtype": "int32,int32",
    "input_format": "ND,ND",
    "input_datarange": "0_0.001_normal_normal,[0,512]",
    "output_shape": "[105860,108]",
    "output_dtype": "int32",
    "output_format": "ND",
    "axis": "1",
    "view_shape": "[7514,512]",
    "tile_shape": "[1,512]",
    "tile_block": "9758",
    "total_size": "0.303931594"
}

# -------------------------- 2. 数据类+参数转换（无修改，适配手动填参） --------------------------
@dataclass
class GatherParams:
    input_shape: list
    input_dtype: list
    input_format: str
    input_datarange: str
    output_shape: list
    output_dtype: str
    output_format: str
    view_shape: list
    tile_shape: list
    axis: int

def convert_json_to_params(case_json):
    params = GatherParams(
        input_shape=eval(case_json["input_shape"]),
        input_dtype=case_json["input_dtype"].split(","),
        input_format=case_json["input_format"],
        input_datarange=case_json["input_datarange"].split(","),
        output_shape=eval(case_json["output_shape"]),
        output_dtype=case_json["output_dtype"],
        output_format=case_json["output_format"],
        view_shape=eval(case_json["view_shape"]),
        tile_shape=eval(case_json["tile_shape"]),
        axis=int(case_json["axis"])
    )
    return params

# -------------------------- 3. PYPTO分维度Gather实现（原逻辑完整保留） --------------------------
@pypto.jit
def gather_custom_2dim(input_tensor,index_tensor, output_tensor, params: GatherParams):
    pypto.set_vec_tile_shapes(*params.tile_shape)

    b1, s1 = input_tensor.shape
    b, s = index_tensor.shape
    tile_b = params.view_shape[0]
    tile_s = params.view_shape[1]
    axis = params.axis

    b_loop = (b + tile_b - 1) // tile_b
    s_loop = (s + tile_s - 1) // tile_s

    for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_LO_bIdx", idx_name="b_idx"):
        for s_idx in pypto.loop(0, s_loop, 1, name="LOOP_L1_sIdx", idx_name="s_idx"):
            b_offset = b_idx * tile_b
            s_offset = s_idx * tile_s

            input_offsets = [[b_offset, b_offset + tile_b], [s_offset, s_offset + tile_s]]

            input_view = input_tensor[input_offsets[0][0]:input_offsets[0][1], input_offsets[1][0]:input_offsets[1][1]]
            index_view = index_tensor[input_offsets[0][0]:input_offsets[0][1], input_offsets[1][0]:input_offsets[1][1]]
            output_c_view = pypto.gather(input_view, axis, index_view)
            output_tensor[b_offset:b_offset + tile_b, s_offset:s_offset + tile_s] = output_c_view

@pypto.jit
def gather_custom_3dim(input_tensor,index_tensor, output_tensor, params: GatherParams):
    pypto.set_vec_tile_shapes(*params.tile_shape)

    b1, s1, h1 = input_tensor.shape
    b, s, h = index_tensor.shape
    tile_b = params.view_shape[0]
    tile_s = params.view_shape[1]
    tile_h = params.view_shape[2]
    b_loop = (b + tile_b - 1) // tile_b
    s_loop = (s + tile_s - 1) // tile_s
    h_loop = (h + tile_h - 1) // tile_h
    axis = params.axis

    for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_LO_bIdx", idx_name="b_idx"):
        for s_idx in pypto.loop(0, s_loop, 1, name="LOOP_L1_sIdx", idx_name="s_idx"):
            for h_idx in pypto.loop(0, h_loop, 1, name="LOOP_L2_hIdx", idx_name="h_idx"):
                b_offset = b_idx * tile_b
                s_offset = s_idx * tile_s
                h_offset = h_idx * tile_h

                input_offsets = [[b_offset, b_offset + tile_b], [s_offset, s_offset + tile_s], [h_offset, h_offset + tile_h]]

                input_view = input_tensor[input_offsets[0][0]:input_offsets[0][1], input_offsets[1][0]:input_offsets[1][1], input_offsets[2][0]:input_offsets[2][1]]
                index_view = index_tensor[input_offsets[0][0]:input_offsets[0][1], input_offsets[1][0]:input_offsets[1][1], input_offsets[2][0]:input_offsets[2][1]]
                output_c_view = pypto.gather(input_view, axis, index_view)
                output_tensor[b_offset:b_offset + tile_b, s_offset:s_offset + tile_s, h_offset:h_offset + tile_h] = output_c_view

@pypto.jit
def gather_custom_4dim(input_tensor,index_tensor, output_tensor, params: GatherParams):
    pypto.set_vec_tile_shapes(*params.tile_shape)

    b1, s1, n1, m1 = input_tensor.shape
    b, s, n, m = index_tensor.shape
    tile_b = pypto.symbolic_scalar(params.view_shape[0])
    tile_s = pypto.symbolic_scalar(params.view_shape[1])
    tile_n = pypto.symbolic_scalar(params.view_shape[2])
    tile_m = pypto.symbolic_scalar(params.view_shape[3])
    b_loop = (b + tile_b - 1) // tile_b
    s_loop = (s + tile_s - 1) // tile_s
    n_loop = (n + tile_n - 1) // tile_n
    m_loop = (m + tile_m - 1) // tile_m
    axis = params.axis

    for b_idx in pypto.loop(0, b_loop, 1, name="LOOP_LO_bIdx", idx_name="b_idx"):
        for s_idx in pypto.loop(0, s_loop, 1, name="LOOP_L1_sIdx", idx_name="s_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L2_nIdx", idx_name="n_idx"):
                for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L3_mIdx", idx_name="m_idx"):
                    b_offset = b_idx * tile_b
                    s_offset = s_idx * tile_s
                    n_offset = n_idx * tile_n
                    m_offset = m_idx * tile_m

                    input_offsets = [[b_offset, b_offset + tile_b], [s_offset, s_offset + tile_s], [n_offset, n_offset + tile_n], [m_offset, m_offset + tile_m]]

                    input_view = input_tensor[input_offsets[0][0]:input_offsets[0][1], input_offsets[1][0]:input_offsets[1][1], input_offsets[2][0]:input_offsets[2][1],
                                 input_offsets[3][0]:input_offsets[3][1]]
                    index_view = index_tensor[input_offsets[0][0]:input_offsets[0][1], input_offsets[1][0]:input_offsets[1][1], input_offsets[2][0]:input_offsets[2][1],
                                 input_offsets[3][0]:input_offsets[3][1]]
                    output_c_view = pypto.gather(input_view, axis, index_view)
                    output_tensor[b_offset:b_offset + tile_b, s_offset:s_offset + tile_s, n_offset:n_offset + tile_n, m_offset:m_offset + tile_m] = output_c_view

# -------------------------- 4. PyTorch原生Gather实现（原逻辑保留） --------------------------
def gather_pytorch(inputs, params: GatherParams):
    output_tensor = torch.gather(inputs[0], params.axis, inputs[1])
    return output_tensor

def gather_golden(inputs, params: GatherParams):
    input_tensor_0_cpu = inputs[0].cpu()
    input_tensor_1_cpu = inputs[1].cpu()
    if input_tensor_0_cpu.dtype in [torch.float16, torch.bfloat16]:
        input_tensor_0_cpu = input_tensor_0_cpu.to(torch.float32)
    if input_tensor_1_cpu.dtype in [torch.int32]:
        input_tensor_1_cpu = input_tensor_1_cpu.to(torch.int64)

    output_tensor_golden = torch.gather(input_tensor_0_cpu, params.axis, input_tensor_1_cpu)
    return output_tensor_golden

# -------------------------- 5. 核心：移除libs依赖，原生实现所有工具函数 --------------------------
def get_torch_dtype(dtype_str: str):
    """替代libs.tools.get_torch_dtype：字符串转torch dtype"""
    dtype_map = {
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return dtype_map.get(dtype_str, torch.float32)

def convert_torch_to_numpy(tensor: torch.Tensor):
    """替代libs.tools.convert_torch_to_numpy：torch tensor转numpy array"""
    return tensor.detach().cpu().numpy()

def precision_compare_binary_consistency(pred: np.ndarray, target: np.ndarray):
    """替代libs.precision_compare：整型二进制一致性校验"""
    assert pred.shape == target.shape, f"shape mismatch: pred {pred.shape} vs target {target.shape}"
    match = (pred == target).all()
    return (match, pred.shape, np.sum(pred != target), np.mean(pred != target))

def precision_compare_float(pred: np.ndarray, target: np.ndarray, dtype_str: str, compute_count=1):
    """替代libs.precision_compare：浮点型精度校验"""
    assert pred.shape == target.shape, f"shape mismatch: pred {pred.shape} vs target {target.shape}"
    abs_error = np.abs(pred - target)
    rel_error = abs_error / (np.abs(target) + 1e-8)
    return (np.max(abs_error), np.mean(abs_error), np.max(rel_error), np.mean(rel_error))

def precision_compare_triple_data(pred: np.ndarray, torch_out: np.ndarray, golden: np.ndarray, dtype_str: str):
    """替代libs.precision_compare：三方数据精度校验"""
    res1 = precision_compare_float(pred, torch_out, dtype_str)
    res2 = precision_compare_float(pred, golden, dtype_str)
    res3 = precision_compare_float(torch_out, golden, dtype_str)
    return (res1, res2, res3, pred.shape)

# -------------------------- 6. 主执行函数+输入构造（完全移除libs.generate依赖） --------------------------
def Gather(case_json, device_id=3, logger=None):
    torch_npu.npu.set_device(device_id)
    params = convert_json_to_params(case_json)
    
    # ========== 原生构造输入张量【核心改造，无任何libs依赖】 ==========
    dtype_input = get_torch_dtype(params.input_dtype[0])
    dtype_index = get_torch_dtype(params.input_dtype[1])
    input_shape_0 = params.input_shape[0]
    input_shape_1 = params.input_shape[1]
    gather_axis = params.axis

    # 构造输入tensor：根据dtype区分整型/浮点型
    if dtype_input in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
        # 整型输入：正态分布归一化到0~0.001区间（匹配datarange:0_0.001_normal_normal）
        input_tensor = torch.randn(input_shape_0, dtype=torch.float32) * 0.0001
        input_tensor = input_tensor.to(dtype_input)
    else:
        # 浮点型输入：标准正态分布+归一化
        input_tensor = torch.randn(input_shape_0, dtype=dtype_input) * 0.0001

    # 构造索引tensor：关键！索引值范围必须是 [0, input_shape[axis]) 避免越界，匹配datarange:[0,512]
    index_min = 0
    index_max = params.input_shape[0][gather_axis]  # 512，和你的用例一致
    index_tensor = torch.randint(low=index_min, high=index_max, size=input_shape_1, dtype=dtype_index)

    # 张量移到NPU
    input_tensor = input_tensor.npu()
    index_tensor = index_tensor.npu()
    output_tensor = torch.zeros(params.output_shape, dtype=get_torch_dtype(params.output_dtype)).npu()
    inputs = [input_tensor, index_tensor]

    # PyTorch原生结果
    torch_out = gather_pytorch(inputs, params)
    torch_out_np = convert_torch_to_numpy(torch_out)
    torch_npu.npu.synchronize()

    # 金标准结果
    golden_out = gather_golden(inputs, params)
    golden_out_np = convert_torch_to_numpy(golden_out)
    torch_npu.npu.synchronize()

    # PYPTO自定义算子执行
    assert len(params.input_shape[0]) in [2, 3, 4], "Currently Only Support Input Shape Dim in [2, 3, 4]"
    if len(params.input_shape[0]) == 2:
        pypto_a_inputs = pypto.from_torch(input_tensor, name="IN1",dynamic_axis=[0,1])
        pypto_b_inputs = pypto.from_torch(index_tensor, name="IN2", dynamic_axis=[0, 1])
        pypto_outputs = pypto.from_torch(output_tensor, name="OUT",dynamic_axis=[0,1])
        gather_custom_2dim(pypto_a_inputs,pypto_b_inputs, pypto_outputs, params)
    elif len(params.input_shape[0]) == 3:
        pypto_a_inputs = pypto.from_torch(input_tensor, name="IN1", dynamic_axis=[0, 1,2])
        pypto_b_inputs = pypto.from_torch(index_tensor, name="IN2", dynamic_axis=[0, 1,2])
        pypto_outputs = pypto.from_torch(output_tensor, name="OUT", dynamic_axis=[0, 1,2])
        gather_custom_3dim(pypto_a_inputs,pypto_b_inputs, pypto_outputs, params)
    else:
        pypto_a_inputs = pypto.from_torch(input_tensor, name="IN1", dynamic_axis=[0, 1, 2,3])
        pypto_b_inputs = pypto.from_torch(index_tensor, name="IN2", dynamic_axis=[0, 1, 2,3])
        pypto_outputs = pypto.from_torch(output_tensor, name="OUT", dynamic_axis=[0, 1, 2,3])
        gather_custom_4dim(pypto_a_inputs,pypto_b_inputs, pypto_outputs, params)
    
    torch_npu.npu.synchronize()
    pypto_out_np = convert_torch_to_numpy(output_tensor)
    print(f"Output Tensor Size: {output_tensor.size()}")

    # 精度校验
    compute_count = 1
    if params.output_dtype.startswith("int"):
        precision_result = precision_compare_binary_consistency(pypto_out_np, torch_out_np)
        precision_result_triple = (precision_result[0], None, None, None)
    else:
        precision_result = precision_compare_float(pypto_out_np, torch_out_np, params.output_dtype, compute_count)
        precision_result_triple = precision_compare_triple_data(pypto_out_np, torch_out_np, golden_out_np, params.output_dtype)

    return precision_result, precision_result_triple

# -------------------------- 7. 执行入口 --------------------------
if __name__ == "__main__":
    # 直接传入手动填写的用例数据，无需读取文件
    res, res_triple = Gather(CASE_DATA)
    print("="*50)
    print("精度校验结果：", res)
    print("三方精度校验结果：", res_triple)
    print("="*50)