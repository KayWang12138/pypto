import inspect
import re

__all__: tuple[str, ...] = ()

# Consider moving elsewhere
_FUNC_NAME__CALC_WORKSPACE = "calc_workspace"
_FUNC_NAME__INFER_SHAPE = "infer_shape"
_FUNC_NAME__INFER_DTYPE = "infer_dtype"


def _unwrap_decorated_func_source(source: str) -> str:
    """Return the def ... body of a function, stripping decorator lines."""
    return source[source.find("def "):]  # a bit ugly, check if there're better options


def _unwrap_decorated_func_name(name: str) -> str:
    """Return the actual function name when __name__ may include decorator info."""
    return name.split()[0]


def _get_renamed_func_source(func, new_func_name: str) -> str:
    """Return function source with the function name replaced by new_func_name."""
    source = _unwrap_decorated_func_source(inspect.getsource(func))
    orig_func_name = _unwrap_decorated_func_name(func.__name__)
    return source.replace(orig_func_name, new_func_name, 1)


def _snake_case_to_camel_case(name: str) -> str:
    """Convert a snake_case string to a camelCase string."""
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _camel_case_to_snake_case(name: str) -> str:
    """Convert PascalCase or camelCase (e.g. ``Add``, ``AddPyptoCustomOp``) to snake_case for paths."""
    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    step2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1)
    return step2.lower()


def _torch_dtype_to_ge_dtype(dtype) -> str:
    """Map torch dtype-like values to GE dtype token string.

    Rules (by torch dtype name):

    - ``torch.float32`` -> ``ge::DT_FLOAT``
    - ``torch.float16`` -> ``ge::DT_FLOAT16``
    - ``torch.int{8,16,32,64}`` -> ``ge::DT_INT{8,16,32,64}``
    - ``torch.uint{8,16,32,64}`` -> ``ge::DT_UINT{8,16,32,64}``
    - ``torch.bool`` -> ``ge::DT_BOOL``
    - ``torch.bfloat16`` -> ``ge::DT_BF16``
    """
    dtype_name = str(dtype)
    if not dtype_name.startswith("torch."):
        raise ValueError(f"Unsupported dtype for GE mapping: {dtype!r}")
    base = dtype_name.replace("torch.", "")
    if base == "float32":
        return "ge::DT_FLOAT"
    if base == "float16":
        return "ge::DT_FLOAT16"
    if base == "bool":
        return "ge::DT_BOOL"
    if base == "bfloat16":
        return "ge::DT_BF16"
    import re
    m = re.fullmatch(r"(u?)int(8|16|32|64)", base)
    if m:
        signed = m.group(1) == ""
        bits = m.group(2)
        kind = "INT" if signed else "UINT"
        return f"ge::DT_{kind}{bits}"
    raise ValueError(f"Unsupported torch dtype for GE mapping: {dtype!r}")


def _torch_dtype_to_ir_dtype(dtype):
    """Map torch dtype-like values to ``pypto_ir.DataType`` enum."""
    import pypto_ir

    dtype_name = str(dtype)
    if not dtype_name.startswith("torch."):
        raise ValueError(f"unsupported dtype ::: {dtype}")
    base = dtype_name.replace("torch.", "")
    # Special-case bfloat16 to map to BF16 instead of BFP16
    if base == "bfloat16":
        ir_dtype_name = "BF16"
    else:
        ir_dtype_name = (
            base.replace("float", "FP")
            .replace("int", "INT")
            .replace("_", "")
            .upper()
        )
    return getattr(pypto_ir.DataType, ir_dtype_name)