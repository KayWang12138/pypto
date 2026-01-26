import torch
import torch.nn as nn
import onnx
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch.onnx import register_custom_op_symbolic
import json
import os
import sys
import argparse

from tools.onnx.export import export_to_onnx
from tools.onnx.pypto_op import pypto_op, META_KEY_ZIP_FILE
from tools.onnx.extract import extract_zip_from_onnx, extract_pypto_meta_from_onnx

NPU_DEVICE_ID = 1
DOMAIN = "ai.onnx.contrib"
OP_TYPE = "AddPyptoCustomOp"
SHAPE = (32, 32, 1, 64)
TILE_SHAPES = (1, 16, 1, 64)

@pypto_op(kernel_name="add_kernel", tile_shapes=TILE_SHAPES, support_dynamic_aligned=True, version=1, incl_src=True)
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

def add_pypto_onnx_symbolic(g, x0, x1, run_mode=0):
    meta = getattr(add_kernel_py, "__pypto_meta__", {})
    meta_json = json.dumps(meta, sort_keys=True)

    # 这些将是onnx内部的属性
    return g.op(f"{DOMAIN}::{OP_TYPE}", x0, x1, run_mode,
                pypto_meta_s = meta_json,
                kernel_name_s = str(meta.get("kernel_name", "add_kernel")),
                version_i = int(meta.get("version", 1)),
                tile_shapes_s=json.dumps(list(meta.get("tile_shapes", []))),
                kernel_source_zip_s=meta.get(META_KEY_ZIP_FILE,""),
                )

register_custom_op_symbolic(
    "pypto::add_pypto",
    add_pypto_onnx_symbolic,
    opset_version=12,
)

class CustomModel(nn.Module):
    def forward(self, x0, x1, run_mode=0):
        return torch.ops.pypto.add_pypto(x0, x1, run_mode=run_mode)

def example_export(models_dir: str, force_cpu: bool = False, force_sim: bool = False):
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

    os.makedirs(models_dir, exist_ok=True)
    export_to_onnx(
        model,
        (input_data0, input_data1),
        f"{models_dir}/pypto_custom_add_zip.onnx",
        input_names=["x0", "x1"],
        output_names=["y"],
    )

def example_load(models_dir: str):
    m = onnx.load(f"{models_dir}/pypto_custom_add_zip.onnx")

    zip = extract_zip_from_onnx(
        onnx_model=m,
        domain=DOMAIN,
        op_type=OP_TYPE,
        b64_attr_names=("kernel_source_zip",),
        out_dir=f"{models_dir}/extracted_onnx_src",
    )

    pypto_meta = extract_pypto_meta_from_onnx(
        onnx_model=m,
        domain=DOMAIN,
        op_type=OP_TYPE,
    )

    print(f"zip:\t{zip}\npypto_meta:\t{pypto_meta}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please specify the mode (export/load) and models dir")
        sys.exit(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["export", "load"])
    parser.add_argument("models_dir", type=str)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-sim", action="store_true")

    args = parser.parse_args()

    if args.mode == "export":
        example_export(args.models_dir, force_cpu=args.force_cpu, force_sim=args.force_sim)
    elif args.mode == "load":
        example_load(args.models_dir)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
