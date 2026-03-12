import onnx
import pypto

from torchair.ge._ge_graph import GeGraph

import argparse
from pathlib import Path

DOMAIN = "ai.onnx.contrib"
OP_TYPE__ADD = "AddPyptoCustomOp"
DOMAIN_OP_TYPE__ADD = f"{DOMAIN}::{OP_TYPE__ADD}"
SHAPE = (32, 32, 1, 64)

def load_demo(path: str):
    path = Path(path)
    model_format = str(path.suffix[1:])

    print(f"\nLoading {model_format} model...")

    if model_format == "onnx":
        m = onnx.load(path)
        node = pypto.export.extract_node_from_onnx(
            onnx_model=m,
            domain=DOMAIN,
            op_type=OP_TYPE__ADD,
        )
    elif model_format == "air":
        with open(path, "rb") as f:
            ge = GeGraph(serialized_model_def=f.read())
            node = pypto.export.extract_node_from_ge_graph(
                ge_graph=ge,
                op_type=DOMAIN_OP_TYPE__ADD,
            )
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print("\n----------------\n")

    kernel_format = pypto.export.extract_kernel_format(node)
    pypto_meta = pypto.export.extract_pypto_meta(node)

    print(f"kernel_format:\t{kernel_format}")
    print(f"pypto_meta:\t{pypto_meta}")

    print("\n----------------\n")

    infer_shape_source = pypto.export.extract_infer_shape_source(node)
    calc_workspace_source = pypto.export.extract_calc_workspace_source(node)
    
    namespace = DOMAIN_OP_TYPE__ADD # optional
    x0_shape = SHAPE
    x1_shape = SHAPE

    print(f"infer_shape() source:\n{infer_shape_source}\n")
    infer_shape = pypto.export.register_infer_shape_fn(infer_shape_source, namespace=namespace)
    print(f"infer_shape() call:\t{infer_shape(x0_shape, x1_shape)}\n")

    print(f"calc_workspace() source:\n{calc_workspace_source}\n")
    calc_workspace = pypto.export.register_calc_workspace_fn(calc_workspace_source, namespace=namespace)
    print(f"calc_workspace() call:\t{calc_workspace(x0_shape, x1_shape)}")

    print("\n----------------\n")

    out_dir_prefix = path.parent / f"extracted_{model_format}"
    must_contain_zip = True

    cpp_sources_zip_meta = pypto.export.extract_cpp_sources(
        node=node,
        out_dir=f"{out_dir_prefix}_cpp_sources",
        strict=must_contain_zip,
    )
    print(f"cpp_zip:\t{cpp_sources_zip_meta}")

    if kernel_format == pypto.export.KERNEL_FORMAT__MULTI:
        must_contain_zip = False # some format may not be presented and that's fine

    if kernel_format == pypto.export.KERNEL_FORMAT__SOURCE or kernel_format == pypto.export.KERNEL_FORMAT__MULTI:
        kernel_source_zip_meta = pypto.export.extract_kernel_source(
            node=node,
            out_dir=f"{out_dir_prefix}_src",
            strict=must_contain_zip,
        )
        print(f"src_zip:\t{kernel_source_zip_meta}" if kernel_source_zip_meta is not None else "No kernel source or failed to unpack")

    if kernel_format == pypto.export.KERNEL_FORMAT__BINARY or kernel_format == pypto.export.KERNEL_FORMAT__MULTI:
        kernel_binary_zip_meta = pypto.export.extract_kernel_binary(
            node=node,
            out_dir=f"{out_dir_prefix}_binary",
            strict=must_contain_zip,
        )
        print(f"binary_zip:\t{kernel_binary_zip_meta}" if kernel_binary_zip_meta is not None else "No kernel binary or failed to unpack")

    if kernel_format == pypto.export.KERNEL_FORMAT__IR or kernel_format == pypto.export.KERNEL_FORMAT__MULTI:
        kernel_ir_zip_meta = pypto.export.extract_kernel_ir(
            node=node,
            out_dir=f"{out_dir_prefix}_ir",
            strict=must_contain_zip,
        )
        print(f"ir_zip:\t{kernel_ir_zip_meta}" if kernel_ir_zip_meta is not None else "No kernel IR or failed to unpack")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="onnx or torch air model path")

    args = parser.parse_args()

    load_demo(args.path)
