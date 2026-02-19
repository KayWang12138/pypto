import torch
import torch.nn as nn
import pypto

import numpy as np
from numpy.testing import assert_allclose
from torch.onnx import register_custom_op_symbolic

import os
import argparse
from typing import Optional

NPU_DEVICE_ID = 1
DOMAIN = "ai.onnx.contrib"
OP_TYPE__ADD = "AddPyptoCustomOp"
DOMAIN_OP_TYPE__ADD = f"{DOMAIN}::{OP_TYPE__ADD}"
SHAPE = (32, 32, 1, 64)
TILE_SHAPES = (1, 16, 1, 64)

@pypto.export.pypto_op_kernel(kernel_name="add_kernel", tile_shapes=TILE_SHAPES, support_dynamic_aligned=True, version=1,
                incl_src=True, incl_binary=True, incl_ir=True)
def add_kernel_py(t0, t1, t2):
    print("Goes through add_kernel")

    tensor_shape = t0.shape
    pypto.set_vec_tile_shapes(*TILE_SHAPES)

    b = pypto.symbolic_scalar(tensor_shape[0])
    n1, n2, dim = tensor_shape[1:]
    tile_b = pypto.symbolic_scalar(1)
    b_loop = b / tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = t0[b_offset:b_offset_end, :n1, :n2, :dim]
        t1_sub = t1[b_offset:b_offset_end, :n1, :n2, :dim]
        t3_sub = t0_sub + t1_sub
        t2[b_offset:b_offset_end, :, :, :] = t3_sub

# 像这样使用jit，以避免jit被传递到@pypto_op装饰器函数中
# add_kernel = pypto.jit(add_kernel_py) # --- can't reallt set runtime_options (e.g. run_mode=1) this way

def create_add_kernel(run_mode: int):
    @pypto.jit(runtime_options={"run_mode": run_mode})
    def add_kernel(t0, t1, t2):
        return add_kernel_py(t0, t1, t2)
    return add_kernel

@torch.library.custom_op("pypto::add_pypto", mutates_args=())
def add_pypto(x0: torch.Tensor, x1: torch.Tensor, run_mode: int = 0) -> torch.Tensor:
    print("Goes through pypto npu kernel")
    output_data = torch.zeros_like(x0, device=x0.device)

    pto_inputs = [
        pypto.from_torch(x0, "IN_0"),
        pypto.from_torch(x1, "IN_1"),
    ]
    pto_outputs = [pypto.from_torch(output_data, "OUT_0")]

    create_add_kernel(run_mode)(*pto_inputs, *pto_outputs)
    #pypto.runtime._device_synchronize()

    print("Returned output_data")
    return output_data

@torch.library.register_fake("pypto::add_pypto")
def add_pypto_fake(x0, x1, run_mode = 0):
    print("Goes through fake kernel")
    assert x0.shape == x1.shape
    return torch.empty_like(x0)

@pypto.export.pypto_op_infer_shape(pypto_op_kernel=add_kernel_py)
def add_pypto_infer_shape(x0_shape, x1_shape):
    return x0_shape

@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=add_kernel_py)
def add_pypto_calc_workspace(x0_shape, x1_shape):
    return 42

@pypto.export.pypto_op_onnx_symbolic(pypto_op_kernel=add_kernel_py)
def add_pypto_onnx_symbolic(g, x0, x1, run_mode=0, op_context={}):
    # 这些将是onnx内部的属性
    node = g.op(DOMAIN_OP_TYPE__ADD, x0, x1, run_mode, **op_context)
    node.setType(x0.type())
    return node

register_custom_op_symbolic(
    "pypto::add_pypto",
    add_pypto_onnx_symbolic,
    opset_version=12,
)

class CustomModel(nn.Module):
    def forward(self, x0, x1, run_mode=0):
        return torch.ops.pypto.add_pypto(x0, x1, run_mode=run_mode)

def export_demo(path: str, force_cpu: bool = False, force_sim: bool = False):
    pypto.set_host_options(only_codegen=True)
    pypto.set_codegen_options(support_dynamic_aligned=True)

    model = CustomModel()
    device = "cpu"

    if not force_sim and torch.npu.is_available():
        run_mode = 0
        torch.npu.set_device(NPU_DEVICE_ID)
        if not force_cpu:
            device = f"npu:{NPU_DEVICE_ID}"
            model = model.to(device)
    else:
        run_mode = 1

    print(f"DEVICE = {device}, RUN_MODE = {run_mode}")

    input_data0 = torch.rand(SHAPE, dtype=torch.float32, device=device)
    input_data1 = torch.rand(SHAPE, dtype=torch.float32, device=device)

    y = model(input_data0, input_data1, run_mode=run_mode)
    y_torch = torch.add(input_data0, input_data1)

    if run_mode == 0:
        assert_allclose(
            np.array(y.cpu()),
            np.array(y_torch.cpu()),
            rtol=3e-3,
            atol=3e-3
        )
        print("Assert Passed")

    model = model.cpu()
    input_data0 = input_data0.cpu()
    input_data1 = input_data1.cpu()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pypto.export.export_to_onnx(
        model,
        (input_data0, input_data1),
        path,
        input_names=["x0", "x1"],
        output_names=["y"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="onnx model path")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-sim", action="store_true")

    args = parser.parse_args()

    export_demo(
        path=args.path,
        force_cpu=args.force_cpu,
        force_sim=args.force_sim,
    )
