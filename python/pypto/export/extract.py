import os
import json
from typing import Union

import onnx
import torchair

from . import pypto_op
from .zip import _unzip_b64_to_dir

_NODE_TYPE__GE = "ge"
_NODE_TYPE__ONNX = "onnx"

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

def __generate_value_extractor(node_type, field_name, value_type):
    func_name = f"_extract_{field_name}_from_{node_type}_node"
    func_code = f"""
def {func_name}({node_type}_node):
    return _extract_{value_type}_attr_from_{node_type}_node(
        {node_type}_node={node_type}_node,
        attr_name=pypto_op._META_KEY__{field_name.upper()}
    )
"""
    exec(func_code, globals())

def __generate_string_value_extractor(node_type, field_name):
    return __generate_value_extractor(node_type, field_name, value_type="string")

def __generate_zip_extractor(node_type, field_name):
    func_name = f"_extract_{field_name}_from_{node_type}_node"
    func_code = f"""
def {func_name}({node_type}_node, out_dir: str, strict: bool = True):
    return _try_call(_extract_zip_from_{node_type}_node, strict=strict)(
        {node_type}_node={node_type}_node,
        b64_attr_name=pypto_op._META_KEY__{field_name.upper()}_ZIP,
        out_dir=out_dir,
    )
"""
    exec(func_code, globals())

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

__generate_string_value_extractor(_NODE_TYPE__ONNX, "kernel_format")
__generate_string_value_extractor(_NODE_TYPE__ONNX, "infer_shape_source")
__generate_string_value_extractor(_NODE_TYPE__ONNX, "calc_workspace_source")
__generate_string_value_extractor(_NODE_TYPE__ONNX, "meta_json")

__generate_zip_extractor(_NODE_TYPE__ONNX, "kernel_source")
__generate_zip_extractor(_NODE_TYPE__ONNX, "kernel_binary")
__generate_zip_extractor(_NODE_TYPE__ONNX, "kernel_ir")
__generate_zip_extractor(_NODE_TYPE__ONNX, "cpp_sources")

def _extract_pypto_meta_from_onnx_node(onnx_node):
    return json.loads(_extract_meta_json_from_onnx_node(onnx_node))

### Torch Air

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

__generate_string_value_extractor(_NODE_TYPE__GE, "kernel_format")
__generate_string_value_extractor(_NODE_TYPE__GE, "infer_shape_source")
__generate_string_value_extractor(_NODE_TYPE__GE, "calc_workspace_source")
__generate_string_value_extractor(_NODE_TYPE__GE, "meta_json")

__generate_zip_extractor(_NODE_TYPE__GE, "kernel_source")
__generate_zip_extractor(_NODE_TYPE__GE, "kernel_binary")
__generate_zip_extractor(_NODE_TYPE__GE, "kernel_ir")
__generate_zip_extractor(_NODE_TYPE__GE, "cpp_sources")

def _extract_pypto_meta_from_ge_node(ge_node):
    return json.loads(_extract_meta_json_from_ge_node(ge_node))

### Common
# > can introduce Extractor class to store state / cache

def __generate_common_simple_value_extractor(field_name):
    func_name = f"extract_{field_name}"
    func_code = f"""
def {func_name}(node):
    if isinstance(node, onnx.NodeProto):
        return _extract_{field_name}_from_{_NODE_TYPE__ONNX}_node(node)
    else:
        return _extract_{field_name}_from_{_NODE_TYPE__GE}_node(node)
"""
    exec(func_code, globals())

def __generate_common_zip_extractor(field_name):
    func_name = f"extract_{field_name}"
    func_code = f"""
def {func_name}(node, out_dir: str, strict: bool = True):
    if isinstance(node, onnx.NodeProto):
        return _extract_{field_name}_from_{_NODE_TYPE__ONNX}_node(node, out_dir, strict)
    else:
        return _extract_{field_name}_from_{_NODE_TYPE__GE}_node(node, out_dir, strict)
"""
    exec(func_code, globals())

__generate_common_simple_value_extractor("kernel_format")
__generate_common_simple_value_extractor("pypto_meta")
__generate_common_simple_value_extractor("infer_shape_source")
__generate_common_simple_value_extractor("calc_workspace_source")

__generate_common_zip_extractor("kernel_source")
__generate_common_zip_extractor("kernel_binary")
__generate_common_zip_extractor("kernel_ir")
__generate_common_zip_extractor("cpp_sources")
