import datetime
import inspect
import json
import os
from pathlib import Path
import random
import torchair
import pypto_ir

from . import cpp
from . import cpp_layout
from .helpers import _camel_case_to_snake_case, _get_renamed_func_source, _snake_case_to_camel_case
from .kernel_utils import _find_kernel_binary_path, _find_kernel_pto_path
from . import meta_schema
from .meta_schema import _is_hidden, _is_user_defined
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

_OPTIONS_KEY__INCL_BINARY = "incl_binary"
_OPTIONS_KEY__INCL_IR = "incl_ir"

_FRAMEWORK_TYPE__ONNX = "ONNX"

# TODO(TorchAir): replace with real GE ``FrameworkType`` token when defined.
_FRAMEWORK_TYPE__TORCHAIR_GE_STUB = "TORCHAIR_GE_STUB"

__all__ = (
    "KERNEL_FORMAT__SOURCE",
    "KERNEL_FORMAT__BINARY",
    "KERNEL_FORMAT__IR",
    "KERNEL_FORMAT__MULTI",
    "pypto_op_kernel",
    "pypto_op_infer_shape",
    "pypto_op_calc_workspace",
    "pypto_op_onnx_symbolic",
    "pypto_op_torchair_fx_node_ge_converter",
)


def _json_dumps_user_meta(meta: dict, *args, **kwargs) -> dict:
    """Serialize only user-defined meta keys to JSON."""
    user_meta = {k: v for k, v in meta.items() if _is_user_defined(k)}
    return json.dumps(user_meta, *args, **kwargs)


def pypto_op_kernel(*, kernel_name, tile_shapes=None, incl_src=False, incl_binary=False, incl_ir=False, **meta):
    """Decorator to attach kernel meta and optional source/binary/IR to a function."""
    def _derive_kernel_format():
        """Return kernel format constant from incl_* flags."""
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
        meta_local = dict(meta) # not hidden, user defined
        meta_local[meta_schema._META_KEY__KERNEL_NAME] = kernel_name
        meta_local[meta_schema._META_KEY__KERNEL_FORMAT] = _derive_kernel_format()
        if tile_shapes is not None:
            meta_local[meta_schema._META_KEY__TILE_SHAPES] = tile_shapes

        if incl_src:
            src_path, b64 = _zip_source_file_to_b64(fn)
            if src_path and b64:
                meta_local[meta_schema._META_KEY__KERNEL_SOURCE_ZIP] = b64
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
    """Decorator to register infer_shape source and C++ wrapper on a kernel."""
    def decorator(fn):
        # can pass func names as separate attributes instead of renaming if needed later
        pypto_op_kernel.__pypto_meta__[meta_schema._META_KEY__INFER_SHAPE_SOURCE] = \
            _get_renamed_func_source(fn, _FUNC_NAME__INFER_SHAPE)
        pypto_op_kernel.__pypto_meta__[meta_schema._META_KEY__INFER_SHAPE_SOURCE_CPP] = \
            cpp._generate_pybind_wrapper(fn, _snake_case_to_camel_case(_FUNC_NAME__INFER_SHAPE))
        pypto_op_kernel.__pypto_infer_shape_fn__ = fn
        return fn
    return decorator


def pypto_op_calc_workspace(*, pypto_op_kernel):
    """Decorator to register calc_workspace Python source and generated C++ wrapper on kernel meta.

    These meta keys are kept for graph tooling and future use; they are not bundled into
    ``cpp_sources_zip`` (no ``calc_workspace`` TU in the exported cpp tree).
    """
    def decorator(fn):
        # can pass func names as separate attributes instead of renaming if needed later
        pypto_op_kernel.__pypto_meta__[meta_schema._META_KEY__CALC_WORKSPACE_SOURCE] = \
            _get_renamed_func_source(fn, _FUNC_NAME__CALC_WORKSPACE)
        pypto_op_kernel.__pypto_meta__[meta_schema._META_KEY__CALC_WORKSPACE_SOURCE_CPP] = \
            cpp._generate_pybind_wrapper(fn, _snake_case_to_camel_case(_FUNC_NAME__CALC_WORKSPACE))
        return fn
    return decorator


def _create_pypto_op_kernel_export(
    *,
    pypto_op_kernel,
    dump_meta,
    extract_input_shapes_dtype,
    framework_type: str,
):
    """Build the kernel export callable that runs the pipeline and returns op_context."""
    def _export_ir(*input_nodes, _kernel_name, _kernel_fn, _tile_shapes, _extract_input_shapes_dtype):
        """Convert kernel to tile IR, compile to PTO, and return path to .pto file."""
        def _convert_dtype(dtype):
            """Map torch dtype string to pypto_ir.DataType."""
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
    def _generate_cpp_sources(
        kernel_name,
        op_type: str,
        framework_type: str,
    ):
        """Write GE OpDef TU, executor, domi plugin, and cpp_layout.json; return path."""
        def _create_cpp_sources_dir(kernel_name):
            """Create a timestamped output dir for C++ sources."""
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            rand_int = random.randint(0, 9999)
            path = Path(f"./output_cpp_sources/{kernel_name}_{now_str}_{rand_int:04d}")
            os.makedirs(path, exist_ok=True)
            return path

        def _write_recursively(dict_of_files, path):
            for fname, content in dict_of_files.items():
                if isinstance(content, dict):
                    os.makedirs(path / fname, exist_ok=True)
                    _write_recursively(content, path / fname)
                else:
                    with open(path / f"{fname}.cpp", "w") as f:
                        f.write(content)

        infer_shape_fn = getattr(pypto_op_kernel, "__pypto_infer_shape_fn__", None)
        stem = _camel_case_to_snake_case(op_type)
        cpp_sources = {
            "framework": {
                "onnx_plugin": {
                    f"{stem}_plugin": cpp._generate_op_custom_plugin_cpp(
                        op_type=op_type, framework_type=framework_type,
                    ),
                },
            },
            "op_host": {
                "src": {
                    f"{stem}_executor": cpp._generate_custom_executor_cpp(op_type),
                },
                f"{stem}_def": cpp._generate_op_custom_def_cpp(
                    infer_shape_fn, op_type=op_type
                ),
            },
        }
        cpp_sources_dir = _create_cpp_sources_dir(kernel_name)
        _write_recursively(cpp_sources, cpp_sources_dir)
        manifest = cpp_layout._build_cpp_layout_manifest(op_type=op_type)
        layout_path = cpp_sources_dir / cpp_layout._CPP_LAYOUT_FILENAME
        with open(layout_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return cpp_sources_dir

    def pypto_op_kernel_export(*input_nodes, op_type: str):
        """Run export pipeline (cpp, optional binary/IR zips) and return op_context from dump_meta.

        op_type
            C++ identifier used everywhere generated code named the op: plugin
            ``ParseParam*``, ``REGISTER_CUSTOM_OP`` / ``OriginOpType``, ``OpDef`` / ``OP_ADD``,
            sinkable executor class, and ``REG_AUTO_MAPPING_OP`` (must match your ONNX / GE op type
            string without domain, e.g. ``AddPyptoCustomOp``).
        """
        cpp._validate_op_type_identifier(op_type)
        meta = getattr(pypto_op_kernel, "__pypto_meta__", {})
        kernel_name = str(meta.get(meta_schema._META_KEY__KERNEL_NAME))

        # infer_shape C++ is embedded in op_host/<op_type_snake>_def.cpp (not a separate infer_shape.cpp).
        # calc_workspace stays on meta only (see pypto_op_calc_workspace); not emitted into cpp_sources_zip.
        cpp_sources_dir = _generate_cpp_sources(
            kernel_name,
            op_type=op_type,
            framework_type=framework_type,
        )
        meta[meta_schema._META_KEY__CPP_SOURCES_ZIP] = _zip_cpp_sources_dir_to_b64(cpp_sources_dir)
        print(f"cpp sources path ::: {cpp_sources_dir}")

        options = getattr(pypto_op_kernel, "__pypto_options__", {})
        incl_binary = options.get(_OPTIONS_KEY__INCL_BINARY, False)
        incl_ir = options.get(_OPTIONS_KEY__INCL_IR, False)

        if incl_binary:
            binary_path = _find_kernel_binary_path(kernel_name)
            print(f"kernel binary path ::: {binary_path}")
            meta[meta_schema._META_KEY__KERNEL_BINARY_ZIP] = _zip_kernel_dir_to_b64(binary_path)

        if incl_ir:
            ir_path = _export_ir(
                *input_nodes,
                _kernel_name=kernel_name,
                _kernel_fn=pypto_op_kernel,
                _tile_shapes=meta.get(meta_schema._META_KEY__TILE_SHAPES),
                _extract_input_shapes_dtype=extract_input_shapes_dtype,
            )
            print(f"kernel IR path ::: {ir_path}")
            meta[meta_schema._META_KEY__KERNEL_IR_ZIP] = _zip_pto_file_to_b64(ir_path)

        op_context = dump_meta(meta)
        return op_context
    return pypto_op_kernel_export


def _with_pypto_op_kernel_export(
    pypto_op_kernel,
    dump_meta,
    extract_input_shapes_dtype,
    framework_type: str,
):
    """Decorator that injects pypto_op_kernel_export into the wrapped function.

    *framework_type* is bound into generated domi plugin C++ (``FrameworkType``), e.g.
    ``_FRAMEWORK_TYPE__ONNX`` or ``_FRAMEWORK_TYPE__TORCHAIR_GE_STUB``.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            pypto_op_kernel_export = _create_pypto_op_kernel_export(
                pypto_op_kernel=pypto_op_kernel,
                dump_meta=dump_meta,
                extract_input_shapes_dtype=extract_input_shapes_dtype,
                framework_type=framework_type,
            )
            return fn(*args, **kwargs, pypto_op_kernel_export=pypto_op_kernel_export)
        return wrapper
    return decorator


def pypto_op_onnx_symbolic(*, pypto_op_kernel):
    """Decorator to register ONNX symbolic and meta dump for a kernel."""
    def decorator(fn):
        def _dump_meta(meta: dict):
            """Build ONNX op context from meta (hidden keys skipped, types encoded in attr names)."""
            op_context = {}
            for k, v in meta.items():
                if _is_hidden(k):
                    continue
                if type(v) in [list, tuple, dict]:
                    op_context[f"{k}_s"] = json.dumps(v)
                elif type(v) not in [int, float, str]:
                    op_context[f"{k}_s"] = str(v)
                else:
                    op_context[f"{k}_{type(v).__name__[0]}"] = v

            # skip dumping non-user meta to json (zips to optimize space usage, the rest for clarity)
            meta_json = _json_dumps_user_meta(meta, sort_keys=True)
            op_context[f"{meta_schema._META_KEY__META_JSON}_s"] = meta_json
            return op_context

        def _extract_input_shapes_dtype(*input_nodes):
            """Return (list of shapes, dtype) from ONNX input nodes."""
            if len(input_nodes) == 0:
                raise ValueError("input_nodes cannot be empty")
            shapes = [node.type().sizes() for node in input_nodes]
            dtype = input_nodes[0].type().dtype()  # TODO check a scenario with multiple dtypes
            return (shapes, dtype)

        return _with_pypto_op_kernel_export(
            pypto_op_kernel,
            _dump_meta,
            _extract_input_shapes_dtype,
            framework_type=_FRAMEWORK_TYPE__ONNX,
        )(fn)
    return decorator


def pypto_op_torchair_fx_node_ge_converter(*, pypto_op_kernel):
    """Decorator to register TorchAir GE converter and meta dump for a kernel."""
    def decorator(fn):
        def _dump_meta(meta: dict):
            """Build GE op context from meta (all keys, types mapped to torchair.ge.attr)."""
            def _get_ge_attr_name(val):
                """Return GE attribute type name for a Python value."""
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
                """Convert nested tuples to lists for GE list attributes."""
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
            op_context[meta_schema._META_KEY__META_JSON] = torchair.ge.attr.Str(meta_json)
            return op_context

        def _extract_input_shapes_dtype(*input_nodes):
            """Return (list of shapes, dtype) from TorchAir/GE input nodes."""
            if len(input_nodes) == 0:
                raise ValueError("input_nodes cannot be empty")
            shapes = [t.meta.size() for t in input_nodes]
            dtype = input_nodes[0].meta.dtype  # TODO check a scenario with multiple dtypes
            return (shapes, dtype)

        return _with_pypto_op_kernel_export(
            pypto_op_kernel,
            _dump_meta,
            _extract_input_shapes_dtype,
            framework_type=_FRAMEWORK_TYPE__TORCHAIR_GE_STUB,
        )(fn)
    return decorator
