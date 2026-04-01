import torch
import torch.nn as nn

import torchair
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor as torchair_tensor

import numpy as np
from numpy.testing import assert_allclose

import argparse
import math
import os
from typing import Callable

import pypto

OP_TYPE__ADD = "AddPyptoCustomOp"
# 4D inputs/output (elementwise add); aligned with onnx/export_add_kernel.py
SHAPE = (32, 32, 1, 64)
TILE_SHAPES = (1, 4, 1, 64)


@pypto.export.pypto_op_kernel(
    kernel_name="add_kernel",
    vec_tile_shapes=TILE_SHAPES,
    support_dynamic_aligned=True,
    version=1,
    incl_src=True,
    incl_binary=True,
    incl_ir=True,
)
def add_kernel_body(input0, input1):
    print("Goes through add_kernel")
    pypto.set_vec_tile_shapes(*TILE_SHAPES)
    return input0 + input1


def create_add_kernel(run_mode: int):
    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def add_kernel(
        input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
        input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
        output: pypto.Tensor(SHAPE, pypto.DT_FP32),
    ):
        output.move(add_kernel_body(input0, input1))
    return add_kernel


@torch.library.custom_op("pypto::add_pypto", mutates_args=())
def add_pypto(input0: torch.Tensor, input1: torch.Tensor, run_mode: int = 0) -> torch.Tensor:
    print("Goes through pypto npu kernel")
    out_shape = add_pypto_infer_shape(tuple(input0.shape), tuple(input1.shape))
    output = torch.zeros(out_shape, dtype=input0.dtype, device=input0.device)
    create_add_kernel(run_mode)(input0, input1, output)
    pypto.runtime._device_synchronize()
    print("Returned output")
    return output


@torch.library.register_fake("pypto::add_pypto")
def add_pypto_fake(input0, input1, run_mode = 0):
    print("Goes through fake kernel")
    assert input0.shape == input1.shape
    out_shape = add_pypto_infer_shape(tuple(input0.shape), tuple(input1.shape))
    return torch.empty(out_shape, dtype=input0.dtype, device=input0.device)


@pypto.export.pypto_op_infer_shape(pypto_op_kernel=add_kernel_body)
def add_pypto_infer_shape(
    input0_shape: tuple[int, int, int, int],
    input1_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return input0_shape


@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=add_kernel_body)
def add_pypto_calc_workspace(
    input0_shape: tuple[int, int, int, int],
    input1_shape: tuple[int, int, int, int],
) -> int:
    n = math.prod(input0_shape)
    return n * 4 * 2


@pypto.export.pypto_op_infer_dtype(pypto_op_kernel=add_kernel_body)
def add_pypto_infer_dtype(
    input0_dtype: torch.dtype,
    input1_dtype: torch.dtype,
) -> torch.dtype:
    return input0_dtype


@register_fx_node_ge_converter(torch.ops.pypto.add_pypto.default)
@pypto.export.pypto_op_torchair_fx_node_ge_converter(pypto_op_kernel=add_kernel_body)
def converter_add_pypto(x: torchair_tensor, y: torchair_tensor, z: torchair_tensor = None,
        meta_outputs: any = None, pypto_op_kernel_export: Callable[..., dict]=None):
    op_context = pypto_op_kernel_export(x, y, op_type=OP_TYPE__ADD)
    return torchair.ge.custom_op(
        op_type=OP_TYPE__ADD,
        inputs={
            "x": x,
            "y": y,
        },
        outputs=["z"],
        attrs=op_context,
    )


class CustomModel(nn.Module):
    def forward(self, input0, input1, run_mode=0):
        return torch.ops.pypto.add_pypto(input0, input1, run_mode=run_mode)

def get_device_id():
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set TILE_FWK_DEVICE_ID variable before running this demo:")
        print("\texport TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ["TILE_FWK_DEVICE_ID"])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None

def export_demo(path: str, force_cpu: bool = False, force_sim: bool = False):
    pypto.set_codegen_options(support_dynamic_aligned=True)

    model = CustomModel()
    device = "cpu"

    if not force_sim and torch.npu.is_available():
        run_mode = 0
        device_id = get_device_id()
        torch.npu.set_device(device_id)
        if not force_cpu:
            device = f"npu:{device_id}"
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

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pypto.export.export_to_torchair(
        model,
        (input_data0, input_data1),
        path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="torch air model path")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-sim", action="store_true")

    args = parser.parse_args()

    export_demo(
        path=args.path,
        force_cpu=args.force_cpu,
        force_sim=args.force_sim,
    )
