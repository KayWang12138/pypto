import torch
import torch.nn as nn

import numpy as np
from numpy.testing import assert_allclose
from torch.onnx import register_custom_op_symbolic

import os
import argparse

import pypto

DOMAIN = "ai.onnx.contrib"
OP_TYPE__SOFTMAX = "SoftmaxPyptoCustomOp"
DOMAIN_OP_TYPE__SOFTMAX = f"{DOMAIN}::{OP_TYPE__SOFTMAX}"
SHAPE = (32, 32, 64)  # 3D: (batch, seqlen, dim)
TILE_SHAPES = (1, 4, 64)


@pypto.export.pypto_op_kernel(
    kernel_name="softmax_kernel",
    tile_shapes=TILE_SHAPES,
    support_dynamic_aligned=True,
    version=1,
    incl_src=True,
    incl_binary=True,
    incl_ir=False,
)
def softmax_kernel_body(input_tensor, output_tensor):
    print("Goes through softmax_kernel")
    bs, seqlen, dim = input_tensor.shape
    tile_b = 1  # Process one batch at a time
    b_loop = bs // tile_b

    pypto.set_vec_tile_shapes(*TILE_SHAPES)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        input_view = input_tensor[b_offset:b_offset_end, :seqlen, :dim]

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
    print("Goes through pypto npu kernel")
    out_shape = softmax_pypto_infer_shape(tuple(input_tensor.shape))
    output_tensor = torch.zeros(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    create_softmax_kernel(run_mode)(input_tensor, output_tensor)
    pypto.runtime._device_synchronize()
    print("Returned output_tensor")
    return output_tensor


@torch.library.register_fake("pypto::softmax_pypto")
def softmax_pypto_fake(input_tensor, run_mode = 0):
    print("Goes through fake kernel")
    out_shape = softmax_pypto_infer_shape(tuple(input_tensor.shape))
    return torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)


@pypto.export.pypto_op_infer_shape(pypto_op_kernel=softmax_kernel_body)
def softmax_pypto_infer_shape(input_tensor_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Output shape matches input (softmax is shape-preserving)."""
    return input_tensor_shape


@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=softmax_kernel_body)
def softmax_pypto_calc_workspace(input_tensor_shape: tuple[int, ...]) -> int:
    return 123


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


class CustomModel(nn.Module):
    def forward(self, input_tensor, run_mode=0):
        return torch.ops.pypto.softmax_pypto(input_tensor, run_mode=run_mode)


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

    x = torch.rand(SHAPE, dtype=torch.float32, device=device)

    y = model(x, run_mode=run_mode)
    y_torch = torch.softmax(x, dim=-1)

    if run_mode == 0:
        assert_allclose(
            np.array(y.cpu()),
            np.array(y_torch.cpu()),
            rtol=3e-3,
            atol=3e-3
        )
        print("Assert Passed")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pypto.export.export_to_onnx(
        model,
        (x,),
        path,
        input_names=["x"],
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
