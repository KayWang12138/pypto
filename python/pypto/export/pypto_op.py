import datetime
import inspect
import json
import os
from pathlib import Path
import random

import torchair
import pypto_ir

from .cpp import _generate_pybind_wrapper, _generate_op_kernel_info, _generate_op_compile, _generate_op_execute
from .helpers import _get_renamed_func_source
from .kernel_utils import _find_kernel_binary_path, _find_kernel_pto_path
from .zip import _zip_source_file_to_b64, _zip_kernel_dir_to_b64, _zip_pto_file_to_b64, _zip_cpp_sources_dir_to_b64

# FIXME temporary import path just for demos, TODO update after ir_converter is finalized and pushed by the developer
from .ir_converter.ir_converter_tile import convert_kernel_to_tile_ir
from .ir_converter.compile_and_preview import compile_and_preview

KERNEL_FORMAT__SOURCE = "source"
KERNEL_FORMAT__BINARY = "binary"
KERNEL_FORMAT__IR = "ir"
KERNEL_FORMAT__MULTI = "multi"

_FUNC_NAME__INFER_SHAPE = "infer_shape"
_FUNC_NAME__CALC_WORKSPACE = "calc_workspace"

# can refactor and add meta key registration interface with attributes like hidden, user, etc.
_META_KEY__KERNEL_NAME = "kernel_name"
_META_KEY__KERNEL_FORMAT = "kernel_format"
_META_KEY__KERNEL_SOURCE_ZIP = "kernel_source_zip"
_META_KEY__KERNEL_BINARY_ZIP = "kernel_binary_zip"
_META_KEY__KERNEL_IR_ZIP = "kernel_ir_zip"

_META_KEY__CPP_SOURCES_ZIP = "cpp_sources_zip"

_META_KEY__TILE_SHAPES = "tile_shapes"

_META_KEY__INFER_SHAPE_SOURCE = "infer_shape_source"
_META_KEY__INFER_SHAPE_SOURCE_CPP = "infer_shape_source_cpp"
_META_KEY__CALC_WORKSPACE_SOURCE = "calc_workspace_source"
_META_KEY__CALC_WORKSPACE_SOURCE_CPP = "calc_workspace_source_cpp"

_META_KEY__META_JSON = "meta_json"

_OPTIONS_KEY__INCL_BINARY = "incl_binary"
_OPTIONS_KEY__INCL_IR = "incl_ir"

def _is_hidden_meta(key: str) -> bool:
    return key in [
        _META_KEY__INFER_SHAPE_SOURCE_CPP, _META_KEY__CALC_WORKSPACE_SOURCE_CPP,
    ]

def _is_user_meta(key: str) -> bool:
    return key not in [
        _META_KEY__INFER_SHAPE_SOURCE, _META_KEY__INFER_SHAPE_SOURCE_CPP,
        _META_KEY__CALC_WORKSPACE_SOURCE, _META_KEY__CALC_WORKSPACE_SOURCE_CPP,
        _META_KEY__KERNEL_SOURCE_ZIP, _META_KEY__KERNEL_BINARY_ZIP, _META_KEY__KERNEL_IR_ZIP,
    ]

def _json_dumps_user_meta(meta: dict, *args, **kwargs) -> dict:
    user_meta = {
        k: v for k, v in meta.items() if _is_user_meta(k)
    }
    return json.dumps(user_meta, *args, **kwargs)

def pypto_op_kernel(*, kernel_name, tile_shapes=None, incl_src=False, incl_binary=False, incl_ir=False, **meta):
    def _derive_kernel_format():
        n_formats = int(incl_src) + int(incl_binary) + int(incl_ir)
        if n_formats > 1:
            return KERNEL_FORMAT__MULTI
        if incl_src:
            return KERNEL_FORMAT__SOURCE
        if incl_binary:
            return KERNEL_FORMAT__BINARY
        if incl_ir:
            return KERNEL_FORMAT__IR

    def decorator(fn):
        meta_local = dict(meta)
        meta_local[_META_KEY__KERNEL_NAME] = kernel_name
        meta_local[_META_KEY__KERNEL_FORMAT] = _derive_kernel_format()
        if tile_shapes is not None:
            meta_local[_META_KEY__TILE_SHAPES] = tile_shapes

        if incl_src:
            src_path, b64 = _zip_source_file_to_b64(fn)
            if src_path and b64:
                meta_local[_META_KEY__KERNEL_SOURCE_ZIP] = b64
        if incl_ir:
            if tile_shapes is None:
                raise ValueError("tile_shapes must be specified when incl_ir=True")

        fn.__pypto_meta__ = meta_local
        fn.__pypto_options__ = {
            _OPTIONS_KEY__INCL_BINARY : incl_binary,
            _OPTIONS_KEY__INCL_IR : incl_ir,
        }
        return fn
    return decorator

def pypto_op_infer_shape(*, pypto_op_kernel):
    def decorator(fn):
        # can pass func names as separate attributes instead of renaming if needed later
        pypto_op_kernel.__pypto_meta__[_META_KEY__INFER_SHAPE_SOURCE] = \
            _get_renamed_func_source(fn, _FUNC_NAME__INFER_SHAPE)
        pypto_op_kernel.__pypto_meta__[_META_KEY__INFER_SHAPE_SOURCE_CPP] = \
            _generate_pybind_wrapper(fn, _FUNC_NAME__INFER_SHAPE)
        return fn
    return decorator

def pypto_op_calc_workspace(*, pypto_op_kernel):
    def decorator(fn):
        # can pass func names as separate attributes instead of renaming if needed later
        pypto_op_kernel.__pypto_meta__[_META_KEY__CALC_WORKSPACE_SOURCE] = \
            _get_renamed_func_source(fn, _FUNC_NAME__CALC_WORKSPACE)
        pypto_op_kernel.__pypto_meta__[_META_KEY__CALC_WORKSPACE_SOURCE_CPP] = \
            _generate_pybind_wrapper(fn, _FUNC_NAME__CALC_WORKSPACE)
        return fn
    return decorator

def _create_pypto_op_kernel_export(*, pypto_op_kernel, dump_meta, extract_input_shapes_dtype):
    def _ir_export(*input_nodes, _kernel_name, _kernel_fn, _tile_shapes, _extract_input_shapes_dtype):
        def _convert_dtype(dtype):
            dtype_name = str(dtype)
            if not dtype_name.startswith("torch."):
                raise ValueError(f"unsupported dtype ::: {dtype}")
            ir_dtype_name = dtype_name.replace("torch.", "").replace("float", "FP").replace("int", "INT").replace("_", "").upper()
            return getattr(pypto_ir.DataType, ir_dtype_name)

        # TODO can add explicit input checks: type, shape matching
        input_shapes, dtype = _extract_input_shapes_dtype(*input_nodes)
        prog = convert_kernel_to_tile_ir(
            kernel_fn=_kernel_fn,
            program_name=_kernel_name,
            func_name=_kernel_name,
            vec_tile_shapes=_tile_shapes,
            cube_tile_shapes=_tile_shapes,
            input_shapes=input_shapes,
            dtype=_convert_dtype(dtype),
        )
        # TODO check if can return path to .pto in compile_and_preview step so we don't have to look it up later
        compile_and_preview(prog, _kernel_name, pypto_ir.ir.OptimizationStrategy.PTOAS, pypto_ir.backend.BackendType.PTO)
        return _find_kernel_pto_path(_kernel_name)

    # TODO refactor when requirements are finalized
    def _generate_cpp_sources(kernel_name, extra_sources: dict[str, str] = {}):
        def _create_cpp_sources_dir(kernel_name):
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            rand_int = random.randint(1000, 9999)
            path = Path(f"./output_cpp_sources/{kernel_name}_{now_str}_{rand_int}")
            os.makedirs(path, exist_ok=True)
            return path

        cpp_sources = {
            "op_kernel_info": _generate_op_kernel_info(),
            "op_compile": _generate_op_compile(),
            "op_execute": _generate_op_execute(),
            **extra_sources,
        }
        cpp_sources_dir = _create_cpp_sources_dir(kernel_name)
        for fname, content in cpp_sources.items():
            with open(cpp_sources_dir / f"{fname}.cpp", "w") as f:
                f.write(content)
        return cpp_sources_dir


    def pypto_op_kernel_export(*input_nodes):
        meta = getattr(pypto_op_kernel, "__pypto_meta__", {})
        kernel_name = str(meta.get(_META_KEY__KERNEL_NAME))

        cpp_sources_dir = _generate_cpp_sources(
            kernel_name,
            extra_sources = {
                "infer_shape": meta.get(_META_KEY__INFER_SHAPE_SOURCE_CPP),
                "calc_workspace": meta.get(_META_KEY__CALC_WORKSPACE_SOURCE_CPP),
            }
        )
        meta[_META_KEY__CPP_SOURCES_ZIP] = _zip_cpp_sources_dir_to_b64(cpp_sources_dir)
        print(f"cpp sources path ::: {cpp_sources_dir}")

        options = getattr(pypto_op_kernel, "__pypto_options__", {})
        incl_binary = options.get(_OPTIONS_KEY__INCL_BINARY, False)
        incl_ir = options.get(_OPTIONS_KEY__INCL_IR, False)

        if incl_binary:
            binary_path = _find_kernel_binary_path(kernel_name)
            print(f"kernel binary path ::: {binary_path}")
            meta[_META_KEY__KERNEL_BINARY_ZIP] = _zip_kernel_dir_to_b64(binary_path)

        if incl_ir:
            ir_path = _ir_export(
                *input_nodes,
                _kernel_name=kernel_name,
                _kernel_fn=pypto_op_kernel,
                _tile_shapes=meta.get(_META_KEY__TILE_SHAPES),
                _extract_input_shapes_dtype=extract_input_shapes_dtype,
            )
            print(f"kernel IR path ::: {ir_path}")
            meta[_META_KEY__KERNEL_IR_ZIP] = _zip_pto_file_to_b64(ir_path)

        op_context = dump_meta(meta)
        return op_context
    return pypto_op_kernel_export

def _with_pypto_op_kernel_export(pypto_op_kernel, dump_meta, extract_input_shapes_dtype=None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            pypto_op_kernel_export=_create_pypto_op_kernel_export(
                pypto_op_kernel=pypto_op_kernel,
                dump_meta=dump_meta,
                extract_input_shapes_dtype=extract_input_shapes_dtype,
            )
            return fn(*args, **kwargs, pypto_op_kernel_export=pypto_op_kernel_export)
        return wrapper
    return decorator

def pypto_op_onnx_symbolic(*, pypto_op_kernel):
    def decorator(fn):
        def _dump_meta(meta: dict):
            op_context = {}
            for k, v in meta.items():
                if _is_hidden_meta(k):
                    continue
                if type(v) in [list, tuple, dict]:
                    op_context[f"{k}_s"] = json.dumps(v)
                elif type(v) not in [int, float, str]:
                    op_context[f"{k}_s"] = str(v)
                else:
                    op_context[f"{k}_{type(v).__name__[0]}"] = v

            # skip dumping non-user meta to json (zips to optimize space usage, the rest for clarity)
            meta_json = _json_dumps_user_meta(meta, sort_keys=True)
            op_context[f"{_META_KEY__META_JSON}_s"] = meta_json
            return op_context
       
        def _extract_input_shapes_dtype(*input_nodes):
            if len(input_nodes) == 0:
                raise ValueError("input_nodes cannot be empty")
            shapes = [node.type().sizes() for node in input_nodes]
            dtype = input_nodes[0].type().dtype() # TODO check a scenario with multiple dtypes
            return (shapes, dtype)

        return _with_pypto_op_kernel_export(pypto_op_kernel, _dump_meta, _extract_input_shapes_dtype)(fn)
    return decorator

def pypto_op_torchair_fx_node_ge_converter(*, pypto_op_kernel):
    def decorator(fn):
        def _dump_meta(meta: dict):
            def _get_ge_attr_name(val):
                if type(val) in [list, tuple]:
                    elem_attr_name = _get_ge_attr_name(val[0]) if len(val) > 0 else "DataType"
                    return f"List{elem_attr_name}"
                elif type(val) in [int, float, str, bool]:
                    return type(val).__name__.capitalize()
                elif type(val) == dict:
                    return "Str"
                else:
                    return "DataType"

            def _recursive_tuple_to_list(val):
                if type(val) in [list, tuple]:
                    return [_recursive_tuple_to_list(elem) for elem in val]
                return val

            op_context = {}
            for k, v in meta.items():
                ge_attr = getattr(torchair.ge.attr, _get_ge_attr_name(v))
                if type(v) in [list, tuple]:
                    op_context[k] = ge_attr(_recursive_tuple_to_list(v))
                elif type(v) == dict:
                    op_context[k] = ge_attr(json.dumps(v))
                else:
                    op_context[k] = ge_attr(v)

            # skip dumping non-user meta to json (zips to optimize space usage, the rest for clarity)
            meta_json = _json_dumps_user_meta(meta, sort_keys=True)
            op_context[_META_KEY__META_JSON] = torchair.ge.attr.Str(meta_json)
            return op_context

        def _extract_input_shapes_dtype(*input_nodes):
            if len(input_nodes) == 0:
                raise ValueError("input_nodes cannot be empty")
            shapes = [t.meta.size() for t in input_nodes]
            dtype = input_nodes[0].meta.dtype # TODO check a scenario with multiple dtypes
            return (shapes, dtype)

        return _with_pypto_op_kernel_export(pypto_op_kernel, _dump_meta, _extract_input_shapes_dtype)(fn)
    return decorator
