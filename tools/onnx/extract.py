import os
import json
import onnx

from typing import Union

import tools.onnx.pypto_op as pypto_op
from tools.onnx.zip import unzip_b64_to_dir

def extract_node_from_onnx(
    onnx_model: str,
    domain: str,
    op_type: str,
):
    for node in onnx_model.graph.node:
        if node.domain == domain and node.op_type == op_type:
            return node
    raise ValueError(f"No node found for {domain}::{op_type}")

def _extract_attr_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_names: Union[list[str], tuple[str]], #extract any of these
):
    attrs = {a.name: a for a in onnx_node.attribute}

    attr = None
    for name in attr_names:
        if name in attrs:
            attr = attrs[name]
            break
    if attr is None:
        raise KeyError(f"Could not find any of {attr_names} in {domain}::{op_type}")

    return attr

def _extract_string_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_name: str,
):
    attr = _extract_attr_from_onnx_node(
        onnx_node=onnx_node,
        attr_names=(attr_name,),
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
    b64 = _extract_string_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=b64_attr_name,
    )

    op_out_dir = os.path.join(out_dir, onnx_node.op_type)
    zip_meta = unzip_b64_to_dir(b64, op_out_dir)

    return zip_meta

def extract_kernel_format_from_onnx_node(onnx_node: onnx.NodeProto):
    return _extract_string_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__KERNEL_FORMAT,
    )

def extract_pypto_meta_from_onnx_node(onnx_node: onnx.NodeProto):
    meta_json = _extract_string_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__META_JSON,
    )
    return json.loads(meta_json)

def extract_infer_shape_source_from_onnx_node(onnx_node: onnx.NodeProto):
    return _extract_string_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__INFER_SHAPE_SOURCE,
    )

def extract_calc_workspace_source_from_onnx_node(onnx_node: onnx.NodeProto):
    return _extract_string_from_onnx_node(
        onnx_node=onnx_node,
        attr_name=pypto_op._META_KEY__CALC_WORKSPACE_SOURCE,
    )

def extract_kernel_source_from_onnx_node(onnx_node: onnx.NodeProto, out_dir: str):
    return _extract_zip_from_onnx_node(
        onnx_node=onnx_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_SOURCE_ZIP,
        out_dir=out_dir,
    )

def extract_kernel_binary_from_onnx_node(onnx_node: onnx.NodeProto, out_dir: str):
    return _extract_zip_from_onnx_node(
        onnx_node=onnx_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_BINARY_ZIP,
        out_dir=out_dir,
    )

def extract_kernel_ir_from_onnx_node(onnx_node: onnx.NodeProto, out_dir: str):
    return _extract_zip_from_onnx_node(
        onnx_node=onnx_node,
        b64_attr_name=pypto_op._META_KEY__KERNEL_IR_ZIP,
        out_dir=out_dir,
    )

