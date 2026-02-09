import inspect
import json

from tools.onnx.kernel_utils import find_kernel_binary_path, find_kernel_pto_path
from tools.onnx.zip import zip_source_file_to_b64, zip_kernel_dir_to_b64, zip_pto_file_to_b64

KERNEL_FORMAT__SOURCE = "source"
KERNEL_FORMAT__BINARY = "binary"
KERNEL_FORMAT__IR = "ir"
KERNEL_FORMAT__MULTI = "multi"

_FUNC_NAME__INFER_SHAPE = "infer_shape"
_FUNC_NAME__CALC_WORKSPACE = "calc_workspace"

_META_KEY__KERNEL_NAME = "kernel_name"
_META_KEY__KERNEL_FORMAT = "kernel_format"
_META_KEY__KERNEL_SOURCE_ZIP = "kernel_source_zip"
_META_KEY__KERNEL_BINARY_ZIP = "kernel_binary_zip"
_META_KEY__KERNEL_IR_ZIP = "kernel_ir_zip"

_META_KEY__INFER_SHAPE_SOURCE = "infer_shape_source"
_META_KEY__CALC_WORKSPACE_SOURCE = "calc_workspace_source"

_META_KEY__META_JSON = "meta_json"

_OPTIONS_KEY__INCL_BINARY = "incl_binary"
_OPTIONS_KEY__INCL_IR = "incl_ir"

def _unwrap_decorated_func_source(source: str):
    return source[source.find("def "):] # a bit ugly, check if there're better options

def _unwrap_decorated_func_name(name: str):
    return name.split()[0]

def _get_renamed_func_source(func, new_func_name: str):
    source = _unwrap_decorated_func_source(inspect.getsource(func))
    orig_func_name = _unwrap_decorated_func_name(func.__name__)
    return source.replace(orig_func_name, new_func_name, 1)

# 通过@pypto_op装饰器输出pypto算子相关信息
def pypto_op_kernel(*, kernel_name, incl_src=False, incl_binary=False, incl_ir=False, **meta):
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
        if incl_src:
            src_path, b64 = zip_source_file_to_b64(fn)
            if src_path and b64:
                meta_local[_META_KEY__KERNEL_SOURCE_ZIP] = b64
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
        pypto_op_kernel.__pypto_meta__[_META_KEY__INFER_SHAPE_SOURCE] = _get_renamed_func_source(fn, _FUNC_NAME__INFER_SHAPE)
        return fn
    return decorator

def pypto_op_calc_workspace(*, pypto_op_kernel):
    def decorator(fn):
        # can pass func names as separate attributes instead of renaming if needed later
        pypto_op_kernel.__pypto_meta__[_META_KEY__CALC_WORKSPACE_SOURCE] = _get_renamed_func_source(fn, _FUNC_NAME__CALC_WORKSPACE)
        return fn
    return decorator

def pypto_op_onnx_symbolic(*, pypto_op_kernel):
    def decorator(fn):
        def _dump_meta(meta: dict):
            op_context = {}
            for k, v in meta.items():
                if type(v) in [list, tuple, dict]:
                    op_context[f"{k}_s"] = json.dumps(v)
                elif type(v) not in [int, float, str]:
                    op_context[f"{k}_s"] = str(v)
                else:
                    op_context[f"{k}_{type(v).__name__[0]}"] = v

            # skip dumping non-user meta to json (zips to optimize space usage, the rest for clarity)
            filtered_meta = {
                k: v for k, v in meta.items()
                if k not in [
                    _META_KEY__INFER_SHAPE_SOURCE, _META_KEY__CALC_WORKSPACE_SOURCE,
                    _META_KEY__KERNEL_SOURCE_ZIP, _META_KEY__KERNEL_BINARY_ZIP,
                ]
            }

            meta_json = json.dumps(filtered_meta, sort_keys=True)
            op_context[f"{_META_KEY__META_JSON}_s"] = meta_json
            return op_context
        
        def wrapper(*args, **kwargs):
            meta = getattr(pypto_op_kernel, "__pypto_meta__", {})
            kernel_name = str(meta.get(_META_KEY__KERNEL_NAME))

            options = getattr(pypto_op_kernel, "__pypto_options__", {})
            incl_binary = options.get(_OPTIONS_KEY__INCL_BINARY, False)
            incl_ir = options.get(_OPTIONS_KEY__INCL_IR, False)

            if incl_binary:
                binary_path = find_kernel_binary_path(kernel_name)
                print(f"kernel binary path ::: {binary_path}")
                meta[_META_KEY__KERNEL_BINARY_ZIP] = zip_kernel_dir_to_b64(binary_path)

            if incl_ir:
                ir_path = find_kernel_pto_path(kernel_name)
                print(f"kernel IR path ::: {ir_path}")
                meta[_META_KEY__KERNEL_IR_ZIP] = zip_pto_file_to_b64(ir_path)

            return fn(*args, **kwargs, op_context=_dump_meta(meta))
        return wrapper
    return decorator
