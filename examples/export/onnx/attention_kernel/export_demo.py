import argparse
import os

import numpy as np
from numpy.testing import assert_allclose

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic

import pypto


DOMAIN = "ai.onnx.contrib"
OP_TYPE__ATTENTION = "AttentionPyptoCustomOp"
DOMAIN_OP_TYPE__ATTENTION = f"{DOMAIN}::{OP_TYPE__ATTENTION}"

# Shapes mirror the advanced attention example
BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LEN_Q = 16
SEQ_LEN_KV = 16
HEAD_DIM = 64

# Q, K, V and output all share this shape
ATTN_SHAPE = (BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM)

# Tile shapes for the attention kernel
VEC_TILE_SHAPES = (1, NUM_HEADS, 1, HEAD_DIM)
CUBE_TILE_SHAPES = ([64, 64], [64, 64], [64, 64])


def scaled_dot_product_attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference implementation of scaled dot-product attention using PyTorch."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


@pypto.export.pypto_op_kernel(
    kernel_name="attention_kernel",
    vec_tile_shapes=VEC_TILE_SHAPES,
    cube_tile_shapes=CUBE_TILE_SHAPES,
    support_dynamic_aligned=True,
    version=1,
    incl_src=True,
    incl_binary=True,
    incl_ir=False,
)
def attention_kernel_body(q, k, v):
    """PyPTO kernel body for scaled dot-product attention."""
    scale = 1.0 / (HEAD_DIM**0.5)
    pypto.set_cube_tile_shapes(*CUBE_TILE_SHAPES)
    pypto.set_vec_tile_shapes(*VEC_TILE_SHAPES)

    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=pypto.DT_BF16)
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    attn_output = pypto.matmul(attn_weights, v, out_dtype=pypto.DT_BF16)
    return attn_output


def create_attention_kernel(run_mode: int):
    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def attention_kernel(
        q: pypto.Tensor(ATTN_SHAPE, pypto.DT_BF16),
        k: pypto.Tensor(ATTN_SHAPE, pypto.DT_BF16),
        v: pypto.Tensor(ATTN_SHAPE, pypto.DT_BF16),
        output: pypto.Tensor(ATTN_SHAPE, pypto.DT_BF16),
    ):
        output.move(attention_kernel_body(q, k, v))

    return attention_kernel


@pypto.export.pypto_op_infer_shape(pypto_op_kernel=attention_kernel_body)
def attention_pypto_infer_shape(
    q_shape: tuple[int, int, int, int],
    k_shape: tuple[int, int, int, int],
    v_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return q_shape


@pypto.export.pypto_op_calc_workspace(pypto_op_kernel=attention_kernel_body)
def attention_pypto_calc_workspace(
    q_shape: tuple[int, int, int, int],
    q_dtype_size: int,
    k_shape: tuple[int, int, int, int],
    k_dtype_size: int,
    v_shape: tuple[int, int, int, int],
    v_dtype_size: int,
) -> int:
    b, h, lq, d = q_shape
    _, _, lkv, _ = k_shape
    n_scores = b * h * lq * lkv
    n_qkv = b * h * lq * d
    esz = max(q_dtype_size, k_dtype_size, v_dtype_size)
    # K^T, score matmul, softmax on scores, attn×V — rough multi-buffer bound (bytes).
    return (n_scores * 6 + n_qkv * 3) * esz


@pypto.export.pypto_op_infer_dtype(pypto_op_kernel=attention_kernel_body)
def attention_pypto_infer_dtype(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
) -> torch.dtype:
    # In this example, we fix the output dtype to BF16 regardless of inputs.
    return torch.bfloat16


@torch.library.custom_op("pypto::attention_pypto", mutates_args=())
def attention_pypto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    run_mode: int = 0,
) -> torch.Tensor:
    out_shape = attention_pypto_infer_shape(tuple(q.shape), tuple(k.shape), tuple(v.shape))
    out_dtype = attention_pypto_infer_dtype(q.dtype, k.dtype, v.dtype)
    output = torch.zeros(out_shape, dtype=out_dtype, device=q.device)
    create_attention_kernel(run_mode)(q, k, v, output)
    pypto.runtime._device_synchronize()
    return output


@torch.library.register_fake("pypto::attention_pypto")
def attention_pypto_fake(q, k, v, run_mode: int = 0):
    out_shape = attention_pypto_infer_shape(tuple(q.shape), tuple(k.shape), tuple(v.shape))
    out_dtype = attention_pypto_infer_dtype(q.dtype, k.dtype, v.dtype)
    return torch.empty(out_shape, dtype=out_dtype, device=q.device)


@pypto.export.pypto_op_onnx_symbolic(pypto_op_kernel=attention_kernel_body)
def attention_pypto_onnx_symbolic(g, q, k, v, run_mode=0, pypto_op_kernel_export=None):
    op_context = pypto_op_kernel_export(q, k, v, op_type=OP_TYPE__ATTENTION)
    node = g.op(DOMAIN_OP_TYPE__ATTENTION, q, k, v, run_mode, **op_context)
    node.setType(q.type())
    return node


register_custom_op_symbolic(
    "pypto::attention_pypto",
    attention_pypto_onnx_symbolic,
    opset_version=12,
)


class CustomAttentionModel(nn.Module):
    def forward(self, q, k, v, run_mode: int = 0):
        return torch.ops.pypto.attention_pypto(q, k, v, run_mode=run_mode)


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

    model = CustomAttentionModel()
    device = "cpu"

    if not force_sim and torch.npu.is_available():
        run_mode = 0
        device_id = get_device_id()
        if device_id is None:
            return
        torch.npu.set_device(device_id)
        if not force_cpu:
            device = f"npu:{device_id}"
            model = model.to(device)
    else:
        run_mode = 1

    print(f"DEVICE = {device}, RUN_MODE = {run_mode}")

    q = torch.rand(ATTN_SHAPE, dtype=torch.bfloat16, device=device)
    k = torch.rand(ATTN_SHAPE, dtype=torch.bfloat16, device=device)
    v = torch.rand(ATTN_SHAPE, dtype=torch.bfloat16, device=device)

    y = model(q, k, v, run_mode=run_mode)

    scale = 1.0 / (HEAD_DIM**0.5)
    y_torch = scaled_dot_product_attention_golden(q, k, v, scale)

    if run_mode == 0:
        assert_allclose(
            np.array(y.cpu().float().numpy()),
            np.array(y_torch.cpu().float().numpy()),
            rtol=3e-3,
            atol=3e-3,
        )
        print("Assert Passed")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pypto.export.export_to_onnx(
        model,
        (q, k, v),
        path,
        input_names=["q", "k", "v"],
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




