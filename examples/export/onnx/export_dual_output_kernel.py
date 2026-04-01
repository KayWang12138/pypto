"""ONNX export demo: one pypto JIT kernel with two output tensors."""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from numpy.testing import assert_allclose
from torch.onnx import register_custom_op_symbolic

import pypto

DOMAIN = "ai.onnx.contrib"
OP_TYPE__DUAL = "DualPyptoCustomOp"
DOMAIN_OP_TYPE__DUAL = f"{DOMAIN}::{OP_TYPE__DUAL}"
SHAPE = (32, 32, 1, 64)
TILE_SHAPES = (1, 4, 1, 64)


@pypto.export.pypto_op_kernel(
    kernel_name="dual_kernel",
    vec_tile_shapes=TILE_SHAPES,
    support_dynamic_aligned=True,
    version=1,
    incl_src=True,
    incl_binary=True,
    incl_ir=False,
)
def dual_kernel_body(input0, input1):
    print("Goes through dual_kernel")
    pypto.set_vec_tile_shapes(*TILE_SHAPES)
    return pypto.add(input0, input1), pypto.sub(input0, input1)


def create_dual_kernel(run_mode: int):
    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def dual_kernel(
        input0: pypto.Tensor(SHAPE, pypto.DT_FP16),
        input1: pypto.Tensor(SHAPE, pypto.DT_FP16),
        output0: pypto.Tensor(SHAPE, pypto.DT_FP16),
        output1: pypto.Tensor(SHAPE, pypto.DT_FP16),
    ):
        x, y = dual_kernel_body(input0, input1)
        output0.move(x)
        output1.move(y)

    return dual_kernel


@pypto.export.pypto_op_infer_shape(pypto_op_kernel=dual_kernel_body)
def dual_pypto_infer_shape(
    input0_shape: tuple[int, int, int, int],
    input1_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    return (input0_shape, input0_shape)


@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=dual_kernel_body)
def dual_pypto_calc_workspace(
    input0_shape: tuple[int, int, int, int],
    input1_shape: tuple[int, int, int, int],
) -> int:
    # Two elementwise ops (add/sub); allow a few FP16 tile-sized temporaries.
    n = math.prod(input0_shape)
    return n * 2 * 3


@pypto.export.pypto_op_infer_dtype(pypto_op_kernel=dual_kernel_body)
def dual_pypto_infer_dtype(
    input0_dtype: torch.dtype, input1_dtype: torch.dtype
) -> tuple[torch.dtype, torch.dtype]:
    return (input0_dtype, input0_dtype)


@torch.library.custom_op("pypto::dual_pypto", mutates_args=())
def dual_pypto(
    input0: torch.Tensor, input1: torch.Tensor, run_mode: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    sh = dual_pypto_infer_shape(tuple(input0.shape), tuple(input1.shape))
    dt0, dt1 = dual_pypto_infer_dtype(input0.dtype, input1.dtype)
    out0 = torch.zeros(sh[0], dtype=dt0, device=input0.device)
    out1 = torch.zeros(sh[1], dtype=dt1, device=input1.device)
    create_dual_kernel(run_mode)(input0, input1, out0, out1)
    pypto.runtime._device_synchronize()
    return out0, out1


@torch.library.register_fake("pypto::dual_pypto")
def dual_pypto_fake(input0, input1, run_mode=0):
    sh = dual_pypto_infer_shape(tuple(input0.shape), tuple(input1.shape))
    dt0, dt1 = dual_pypto_infer_dtype(input0.dtype, input1.dtype)
    return (
        torch.empty(sh[0], dtype=dt0, device=input0.device),
        torch.empty(sh[1], dtype=dt1, device=input1.device),
    )


@pypto.export.pypto_op_onnx_symbolic(pypto_op_kernel=dual_kernel_body)
def dual_pypto_onnx_symbolic(
    g, input0, input1, run_mode=0, pypto_op_kernel_export=None
):
    op_context = pypto_op_kernel_export(input0, input1, op_type=OP_TYPE__DUAL)
    raw = g.op(DOMAIN_OP_TYPE__DUAL, input0, input1, run_mode, outputs=2, **op_context)
    if isinstance(raw, (list, tuple)):
        o0, o1 = raw[0], raw[1]
    else:
        raise RuntimeError("Expected g.op(..., outputs=2) to return a sequence of two values")
    o0.setType(input0.type())
    o1.setType(input0.type())
    return o0, o1


register_custom_op_symbolic(
    "pypto::dual_pypto",
    dual_pypto_onnx_symbolic,
    opset_version=12,
)


class CustomModel(nn.Module):
    def forward(self, input0, input1, run_mode=0):
        return torch.ops.pypto.dual_pypto(input0, input1, run_mode=run_mode)


def get_device_id():
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set TILE_FWK_DEVICE_ID variable before running this demo:")
        print("\texport TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(
            "ERROR: TILE_FWK_DEVICE_ID must be an integer, got:",
            os.environ["TILE_FWK_DEVICE_ID"],
        )
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

    input_data0 = torch.rand(SHAPE, dtype=torch.float16, device=device)
    input_data1 = torch.rand(SHAPE, dtype=torch.float16, device=device)

    y0, y1 = model(input_data0, input_data1, run_mode=run_mode)
    g0 = input_data0 + input_data1
    g1 = input_data0 - input_data1

    if run_mode == 0:
        assert_allclose(np.array(y0.cpu()), np.array(g0.cpu()), rtol=3e-3, atol=3e-3)
        assert_allclose(np.array(y1.cpu()), np.array(g1.cpu()), rtol=3e-3, atol=3e-3)
        print("Assert Passed")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pypto.export.export_to_onnx(
        model,
        (input_data0, input_data1),
        path,
        input_names=["x0", "x1"],
        output_names=["y0", "y1"],
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
