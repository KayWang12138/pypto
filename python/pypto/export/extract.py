import os
import json
from typing import Union

import onnx
import torchair

from . import pypto_op
from .zip import _unzip_b64_to_dir

### Common

def _try_call(fn, strict=True):
    def wrapper(*args, **kwargs):
        if strict:
            return fn(*args, **kwargs)
        else:
            try:
                return fn(*args, **kwargs)
            except:
                return None
    return wrapper

### ONNX

def extract_node_from_onnx(
    onnx_model: onnx.ModelProto,
    domain: str,
    op_type: str,
):
    for node in onnx_model.graph.node:
        if node.domain == domain and node.op_type == op_type:
            return node
    raise ValueError(f"No node found for {domain}::{op_type}")

def _extract_attr_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_name: str,
):
    for attr in onnx_node.attribute:
        if attr.name == attr_name:
            return attr
    raise KeyError(f"Could not find {attr_name} attribute in onnx node")

def _extract_string_attr_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_name: str,
):
    attr = _extract_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=attr_name,
    )

    if attr.type != onnx.AttributeProto.STRING:
        raise TypeError(f"{attr.name} is not a STRING attribute")

    value = attr.s.decode("utf-8", errors="strict").strip()
    if not value:
        raise ValueError(f"{attr.name} is empty")
    return value

def _extract_zip_from_onnx_node(
    onnx_node: onnx.NodeProto,
    b64_attr_name: str,
    out_dir: Union[str, None] = None,
):
    b64 = _extract_string_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=b64_attr_name,
    )

    op_out_dir = os.path.join(out_dir, onnx_node.op_type)
    zip_meta = _unzip_b64_to_dir(b64, op_out_dir)

    return zip_meta

def _extract_kernel_format_from_onnx_node(onnx_node: onnx.NodeProto):
    return _extract_string_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__KERNEL_FORMAT,
    )

def _extract_pypto_meta_from_onnx_node(onnx_node: onnx.NodeProto):
    meta_json = _extract_string_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__META_JSON,
    )
    return json.loads(meta_json)

def _extract_infer_shape_source_from_onnx_node(onnx_node: onnx.NodeProto):
    return _extract_string_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__INFER_SHAPE_SOURCE,
    )

def _extract_calc_workspace_source_from_onnx_node(onnx_node: onnx.NodeProto):
    return _extract_string_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__CALC_WORKSPACE_SOURCE,
    )

def _extract_kernel_source_from_onnx_node(onnx_node: onnx.NodeProto, out_dir: str, strict: bool = True):
    return _try_call(_extract_zip_from_onnx_node, strict=strict)(
        onnx_node=onnx_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_SOURCE_ZIP,
        out_dir=out_dir,
    )

def _extract_kernel_binary_from_onnx_node(onnx_node: onnx.NodeProto, out_dir: str, strict: bool = True):
    return _try_call(_extract_zip_from_onnx_node, strict=strict)(
        onnx_node=onnx_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_BINARY_ZIP,
        out_dir=out_dir,
    )

def _extract_kernel_ir_from_onnx_node(onnx_node: onnx.NodeProto, out_dir: str, strict: bool = True):
    return _try_call(_extract_zip_from_onnx_node, strict=strict)(
        onnx_node=onnx_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_IR_ZIP,
        out_dir=out_dir,
    )

### TorchAir

def extract_node_from_ge_graph(
    ge_graph: torchair.ge._ge_graph.GeGraph,
    op_type: str,
):
    for op in ge_graph._proto.op:
        if op.type == op_type:
            return op
    raise ValueError(f"No node found for {op_type}")

def _extract_string_attr_from_ge_node(
    ge_node,
    attr_name: str,
):
    attr = ge_node.attr[attr_name].s
    if attr is None:
        raise TypeError(f"{attr.name} is not a STRING attribute")
    return attr.decode()

def _extract_zip_from_ge_node(
    ge_node,
    b64_attr_name: str,
    out_dir: Union[str, None] = None,
):
    b64 = _extract_string_attr_from_ge_node(
        ge_node=ge_node,
        attr_name=b64_attr_name,
    )

    op_out_dir = os.path.join(out_dir, ge_node.type)
    zip_meta = _unzip_b64_to_dir(b64, op_out_dir)

    return zip_meta

def _extract_kernel_format_from_ge_node(ge_node):
    return _extract_string_attr_from_ge_node(
        ge_node=ge_node,
        attr_name=pypto_op._META_KEY__KERNEL_FORMAT
    )

def _extract_pypto_meta_from_ge_node(ge_node):
    meta_json = _extract_string_attr_from_ge_node(
        ge_node=ge_node,
        attr_name=pypto_op._META_KEY__META_JSON
    )
    return json.loads(meta_json)

def _extract_infer_shape_source_from_ge_node(ge_node):
    return _extract_string_attr_from_ge_node(
        ge_node=ge_node,
        attr_name=pypto_op._META_KEY__INFER_SHAPE_SOURCE,
    )

def _extract_calc_workspace_source_from_ge_node(ge_node):
    return _extract_string_attr_from_ge_node(
        ge_node=ge_node,
        attr_name=pypto_op._META_KEY__CALC_WORKSPACE_SOURCE,
    )

def _extract_kernel_source_from_ge_node(ge_node, out_dir: str, strict: bool = True):
    return _try_call(_extract_zip_from_ge_node, strict=strict)(
        ge_node=ge_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_SOURCE_ZIP,
        out_dir=out_dir,
    )

def _extract_kernel_binary_from_ge_node(ge_node, out_dir: str, strict: bool = True): 
    return _try_call(_extract_zip_from_ge_node, strict=strict)(
        ge_node=ge_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_BINARY_ZIP,
        out_dir=out_dir,
    )

def _extract_kernel_ir_from_ge_node(ge_node, out_dir: str, strict: bool = True):
    return _try_call(_extract_zip_from_ge_node, strict=strict)(
        ge_node=ge_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_IR_ZIP,
        out_dir=out_dir,
    )

### Common
# > can introduce Extractor class and use templates for similar functions

def extract_kernel_format_from_node(node):
    if isinstance(node, onnx.NodeProto):
        return _extract_kernel_format_from_onnx_node(node)
    else:
        return _extract_kernel_format_from_ge_node(node)

def extract_pypto_meta_from_node(node):
    if isinstance(node, onnx.NodeProto):
        return _extract_pypto_meta_from_onnx_node(node)
    else:
        return _extract_pypto_meta_from_ge_node(node)

def extract_infer_shape_source_from_node(node):
    if isinstance(node, onnx.NodeProto):
        return _extract_infer_shape_source_from_onnx_node(node)
    else:
        return _extract_infer_shape_source_from_ge_node(node)

def extract_calc_workspace_source_from_node(node):
    if isinstance(node, onnx.NodeProto):
        return _extract_calc_workspace_source_from_onnx_node(node)
    else:
        return _extract_calc_workspace_source_from_ge_node(node)

def extract_kernel_source_from_node(node, out_dir: str, strict: bool = True):
    if isinstance(node, onnx.NodeProto):
        return _extract_kernel_source_from_onnx_node(node, out_dir, strict)
    else:
        return _extract_kernel_source_from_ge_node(node, out_dir, strict)

def extract_kernel_binary_from_node(node, out_dir: str, strict: bool = True):
    if isinstance(node, onnx.NodeProto):
        return _extract_kernel_binary_from_onnx_node(node, out_dir, strict)
    else:
        return _extract_kernel_binary_from_ge_node(node, out_dir, strict)

def extract_kernel_ir_from_node(node, out_dir: str, strict: bool = True):
    if isinstance(node, onnx.NodeProto):
        return _extract_kernel_ir_from_onnx_node(node, out_dir, strict)
    else:
        return _extract_kernel_ir_from_ge_node(node, out_dir, strict)
