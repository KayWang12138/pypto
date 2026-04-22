import torch
import torch.nn as nn

import numpy as np
from numpy.testing import assert_allclose
from torch.onnx import register_custom_op_symbolic

import argparse
import math
import os

import pypto


DOMAIN = "ai.onnx.contrib"
OP_TYPE__ADD = "AddPyptoCustomOp"
OP_TYPE__SOFTMAX = "SoftmaxPyptoCustomOp"
DOMAIN_OP_TYPE__ADD = f"{DOMAIN}::{OP_TYPE__ADD}"
DOMAIN_OP_TYPE__SOFTMAX = f"{DOMAIN}::{OP_TYPE__SOFTMAX}"
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


@pypto.export.pypto_op_infer_shape(pypto_op_kernel=add_kernel_body)
def add_pypto_infer_shape(
    input0_shape: tuple[int, int, int, int],
    input1_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return input0_shape


@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=add_kernel_body)
def add_pypto_calc_workspace(
    input0_shape: tuple[int, int, int, int],
    input0_dtype_size: int,
    input1_shape: tuple[int, int, int, int],
    input1_dtype_size: int,
) -> int:
    n = math.prod(input0_shape)
    esz = max(input0_dtype_size, input1_dtype_size)
    return n * 2 * esz


@pypto.export.pypto_op_infer_dtype(pypto_op_kernel=add_kernel_body)
def add_pypto_infer_dtype(
    input0_dtype: torch.dtype,
    input1_dtype: torch.dtype,
) -> torch.dtype:
    return input0_dtype


@torch.library.custom_op("pypto::add_pypto", mutates_args=())
def add_pypto(input0: torch.Tensor, input1: torch.Tensor, run_mode: int = 0) -> torch.Tensor:
    print("Goes through pypto npu add kernel")
    out_shape = add_pypto_infer_shape(tuple(input0.shape), tuple(input1.shape))
    out_dtype = add_pypto_infer_dtype(input0.dtype, input1.dtype)
    output = torch.zeros(out_shape, dtype=out_dtype, device=input0.device)
    create_add_kernel(run_mode)(input0, input1, output)
    pypto.runtime._device_synchronize()
    print("Returned add_pypto output")
    return output


@torch.library.register_fake("pypto::add_pypto")
def add_pypto_fake(input0, input1, run_mode=0):
    print("Goes through fake add kernel")
    assert input0.shape == input1.shape
    out_shape = add_pypto_infer_shape(tuple(input0.shape), tuple(input1.shape))
    out_dtype = add_pypto_infer_dtype(input0.dtype, input1.dtype)
    return torch.empty(out_shape, dtype=out_dtype, device=input0.device)


@pypto.export.pypto_op_onnx_symbolic(pypto_op_kernel=add_kernel_body)
def add_pypto_onnx_symbolic(g, input0, input1, run_mode=0, pypto_op_kernel_export=None):
    op_context = pypto_op_kernel_export(input0, input1, op_type=OP_TYPE__ADD)
    node = g.op(DOMAIN_OP_TYPE__ADD, input0, input1, run_mode, **op_context)
    node.setType(input0.type())
    return node


register_custom_op_symbolic(
    "pypto::add_pypto",
    add_pypto_onnx_symbolic,
    opset_version=12,
)


@pypto.export.pypto_op_kernel(
    kernel_name="softmax_kernel",
    vec_tile_shapes=TILE_SHAPES,
    support_dynamic_aligned=True,
    version=1,
    incl_src=True,
    incl_binary=True,
    incl_ir=False,
)
def softmax_kernel_body(input_tensor, output_tensor):
    print("Goes through softmax_kernel")
    bs, seqlen, head, dim = input_tensor.shape
    tile_b = 1  # Process one batch at a time
    b_loop = bs // tile_b

    pypto.set_vec_tile_shapes(*TILE_SHAPES)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        input_view = input_tensor[b_offset:b_offset_end, :seqlen, :head, :dim]

        row_max = pypto.amax(input_view, dim=-1, keepdim=True)
        sub = input_view - row_max
        exp = pypto.exp(sub)
        esum = pypto.sum(exp, dim=-1, keepdim=True)
        softmax_out = exp / esum

        output_tensor[b_offset:, ...] = softmax_out


def create_softmax_kernel(run_mode: int):
    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def softmax_kernel(
        input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
        output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    ):
        softmax_kernel_body(input_tensor, output_tensor)

    return softmax_kernel


@torch.library.custom_op("pypto::softmax_pypto", mutates_args=())
def softmax_pypto(input_tensor: torch.Tensor, run_mode: int = 0) -> torch.Tensor:
    print("Goes through pypto npu softmax kernel")
    out_shape = softmax_pypto_infer_shape(tuple(input_tensor.shape))
    out_dtype = softmax_pypto_infer_dtype(input_tensor.dtype)
    output_tensor = torch.zeros(out_shape, dtype=out_dtype, device=input_tensor.device)
    create_softmax_kernel(run_mode)(input_tensor, output_tensor)
    pypto.runtime._device_synchronize()
    print("Returned softmax_pypto output")
    return output_tensor


@torch.library.register_fake("pypto::softmax_pypto")
def softmax_pypto_fake(input_tensor, run_mode=0):
    print("Goes through fake softmax kernel")
    out_shape = softmax_pypto_infer_shape(tuple(input_tensor.shape))
    out_dtype = softmax_pypto_infer_dtype(input_tensor.dtype)
    return torch.empty(out_shape, dtype=out_dtype, device=input_tensor.device)


@pypto.export.pypto_op_infer_shape(pypto_op_kernel=softmax_kernel_body)
def softmax_pypto_infer_shape(input_tensor_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Output shape matches input (softmax is shape-preserving)."""
    return input_tensor_shape


@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=softmax_kernel_body)
def softmax_pypto_calc_workspace(
    input_tensor_shape: tuple[int, ...],
    input_tensor_dtype_size: int,
) -> int:
    n = math.prod(input_tensor_shape)
    return n * 5 * input_tensor_dtype_size


@pypto.export.pypto_op_infer_dtype(pypto_op_kernel=softmax_kernel_body)
def softmax_pypto_infer_dtype(input_tensor_dtype: torch.dtype) -> torch.dtype:
    return input_tensor_dtype


@pypto.export.pypto_op_onnx_symbolic(pypto_op_kernel=softmax_kernel_body)
def softmax_pypto_onnx_symbolic(g, input_tensor, run_mode=0, pypto_op_kernel_export=None):
    op_context = pypto_op_kernel_export(input_tensor, op_type=OP_TYPE__SOFTMAX)
    node = g.op(DOMAIN_OP_TYPE__SOFTMAX, input_tensor, run_mode, **op_context)
    node.setType(input_tensor.type())
    return node


register_custom_op_symbolic(
    "pypto::softmax_pypto",
    softmax_pypto_onnx_symbolic,
    opset_version=12,
)


class CombinedModel(nn.Module):
    def forward(self, x, y, z, run_mode=0):
        t_add = torch.ops.pypto.add_pypto(x, y, run_mode=run_mode)
        t_mul = t_add * z
        t_soft = torch.ops.pypto.softmax_pypto(t_mul, run_mode=run_mode)
        out = torch.neg(t_soft)
        return out


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

    model = CombinedModel()
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

    x = torch.rand(SHAPE, dtype=torch.float32, device=device)
    y = torch.rand(SHAPE, dtype=torch.float32, device=device)
    z = torch.rand(SHAPE, dtype=torch.float32, device=device)

    y_model = model(x, y, z, run_mode=run_mode)
    y_torch = torch.neg(torch.softmax(torch.mul(torch.add(x, y), z), dim=-1))

    if run_mode == 0:
        assert_allclose(
            np.array(y_model.cpu()),
            np.array(y_torch.cpu()),
            rtol=3e-3,
            atol=3e-3,
        )
        print("Assert Passed")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pypto.export.export_to_onnx(
        model,
        (x, y, z),
        path,
        input_names=["x", "y", "z"],
        output_names=["out"],
        forward_kwargs={"run_mode": run_mode},
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

