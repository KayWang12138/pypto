import torch
import torch.nn as nn

import torchair
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor as torchair_tensor

import numpy as np
from numpy.testing import assert_allclose

import os
import argparse
from typing import Callable

import pypto

OP_TYPE__ADD = "AddPyptoCustomOp"
SHAPE = (32, 32, 1, 64)
TILE_SHAPES = (1, 16, 1, 64)

@pypto.export.pypto_op_kernel(kernel_name="add_kernel", tile_shapes=TILE_SHAPES, support_dynamic_aligned=True, version=1,
                incl_src=True, incl_binary=True, incl_ir=True)
def add_kernel_body(t0, t1):
    print("Goes through add_kernel")
    pypto.set_vec_tile_shapes(*TILE_SHAPES)
    return t0 + t1

def create_add_kernel(run_mode: int):
    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def add_kernel(
        t0: pypto.Tensor(SHAPE, pypto.DT_FP32),
        t1: pypto.Tensor(SHAPE, pypto.DT_FP32),
        out: pypto.Tensor(SHAPE, pypto.DT_FP32),
    ):
        out.move(add_kernel_body(t0, t1))
    return add_kernel

@torch.library.custom_op("pypto::add_pypto", mutates_args=())
def add_pypto(x0: torch.Tensor, x1: torch.Tensor, run_mode: int = 0) -> torch.Tensor:
    print("Goes through pypto npu kernel")
    output_data = torch.zeros_like(x0)
    create_add_kernel(run_mode)(x0, x1, output_data)
    pypto.runtime._device_synchronize()
    print("Returned output_data")
    return output_data

@torch.library.register_fake("pypto::add_pypto")
def add_pypto_fake(x0, x1, run_mode = 0):
    print("Goes through fake kernel")
    assert x0.shape == x1.shape
    return torch.empty_like(x0)

@pypto.export.pypto_op_infer_shape(pypto_op_kernel=add_kernel_body)
def add_pypto_infer_shape(x0_shape: tuple[int, int], x1_shape: tuple[int, int]) -> tuple[int, int]:
    return (x0_shape[0], x1_shape[1])

@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=add_kernel_body)
def add_pypto_calc_workspace(x0_shape: tuple[int, int], x1_shape: tuple[int, int]) -> int:
    return 42

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
    def forward(self, x0, x1, run_mode=0):
        return torch.ops.pypto.add_pypto(x0, x1, run_mode=run_mode)

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
