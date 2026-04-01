import inspect
import re
import typing
import ast
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_args, get_origin, get_type_hints

from ..helpers import (
    _FUNC_NAME__CALC_WORKSPACE,
    _FUNC_NAME__INFER_SHAPE,
    _FUNC_NAME__INFER_DTYPE,
    _snake_case_to_camel_case,
    _torch_dtype_to_ge_dtype,
    _unwrap_decorated_func_name,
    _unwrap_decorated_func_source,
)

# This module is WIP and subject to changes (pending requirement clarifications)

__all__: tuple[str, ...] = ()

_OP_TYPE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TUPLE_ORIGINS = (tuple, typing.Tuple)

# Naming for codegen string constants: {optional _}{group}__{VALUE}, e.g. _INFER_SHAPE_MODE__FIXED.
# Tuple-shape metadata ``kind`` and parse result ``input_modes`` / ``output_modes`` values.
_INFER_SHAPE_MODE__FIXED = "fixed"
_INFER_SHAPE_MODE__VARIADIC = "variadic"


@dataclass(frozen=True)
class _InferShapeTupleMeta:
    """Per-parameter or return tuple annotation: fixed rank or variadic ``tuple[T, ...]``."""

    kind: str
    dims: Optional[int] = None
    elem_cpp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind == _INFER_SHAPE_MODE__FIXED:
            if self.dims is None:
                raise TypeError("_InferShapeTupleMeta: fixed kind requires dims")
            if self.elem_cpp is not None:
                raise TypeError("_InferShapeTupleMeta: fixed kind must not set elem_cpp")
        elif self.kind == _INFER_SHAPE_MODE__VARIADIC:
            if self.elem_cpp is None:
                raise TypeError("_InferShapeTupleMeta: variadic kind requires elem_cpp")
            if self.dims is not None:
                raise TypeError("_InferShapeTupleMeta: variadic kind must not set dims")
        else:
            raise TypeError(f"_InferShapeTupleMeta: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class _InferShapeCodegenMeta:
    """Result of :func:`_parse_infer_shape_for_codegen` (infer_shape C++ glue)."""

    sig: inspect.Signature
    param_names: list[str]
    input_modes: list[str]
    input_dims: list[Optional[int]]
    input_elem_cpp: list[Optional[str]]
    output_modes: list[str]
    output_dims: list[Optional[int]]
    output_elem_cpp: list[Optional[str]]
    py_func_name: str
    cpp_bind_name: str


@dataclass(frozen=True)
class _InferDTypeRule:
    """One output dtype rule parsed from infer_dtype (copy input or constant)."""

    copy_input_index: Optional[int]
    const_ge_dtype: Optional[str]


@dataclass(frozen=True)
class _InferDTypeCodegenMeta:
    """Result of parsing infer_dtype for C++ glue across all outputs."""

    rules: list[_InferDTypeRule]


@dataclass(frozen=True)
class _CalcWorkspaceCodegenMeta:
    """Result of parsing calc_workspace for executor C++ glue (shape-only v1)."""

    sig: inspect.Signature
    input_modes: list[str]
    input_dims: list[Optional[int]]
    input_elem_cpp: list[Optional[str]]
    cpp_bind_name: str


def _validate_op_type_identifier(op_type: str) -> None:
    """Ensure *op_type* is safe to embed as a C++ class / macro identifier."""
    if not op_type or not _OP_TYPE_ID_RE.match(op_type):
        raise ValueError(
            "op_type must be a non-empty C++ identifier (letters, digits, underscore; "
            f"must not start with a digit), got {op_type!r}"
        )


def _is_tuple_annotation(ann: Any) -> bool:
    """True if ann is tuple[...] or typing.Tuple[...]."""
    origin = get_origin(ann)
    return origin in _TUPLE_ORIGINS


def _is_variadic_tuple_annotation(ann: Any) -> bool:
    """True if ann is tuple[T, ...] / Tuple[T, ...] (variadic, any single element type T)."""
    if not _is_tuple_annotation(ann):
        return False
    targs = get_args(ann)
    return len(targs) == 2 and targs[1] is Ellipsis


def _tuple_dim_count(ann: Any, ctx: str) -> int:
    """Return number of dimensions for a fixed tuple[...] shape annotation (not tuple[T, ...])."""
    if not _is_tuple_annotation(ann):
        raise TypeError(f"{ctx}: expected tuple[..., ...] annotation, got {ann!r}")
    if _is_variadic_tuple_annotation(ann):
        raise TypeError(f"{ctx}: use variadic metadata for tuple[..., ...], not dim count")
    args = get_args(ann)
    if not args:
        raise TypeError(f"{ctx}: tuple annotation must have at least one element")
    return len(args)


def _infer_shape_tuple_meta(ann: Any, ctx: str) -> _InferShapeTupleMeta:
    """Return fixed or variadic tuple metadata (see ``_INFER_SHAPE_MODE__*``)."""
    if not _is_tuple_annotation(ann):
        raise TypeError(f"{ctx}: expected tuple[..., ...] annotation, got {ann!r}")
    targs = get_args(ann)
    if not targs:
        raise TypeError(f"{ctx}: tuple annotation must have at least one element")
    if len(targs) == 2 and targs[1] is Ellipsis:
        elem_cpp = _to_cpp_type(targs[0])
        return _InferShapeTupleMeta(kind=_INFER_SHAPE_MODE__VARIADIC, elem_cpp=elem_cpp)
    return _InferShapeTupleMeta(kind=_INFER_SHAPE_MODE__FIXED, dims=len(targs))


def _parse_return_shape_outputs(ret_ann: Any, ctx: str) -> list[_InferShapeTupleMeta]:
    """Split *infer_shape* return annotation into one metadata entry per logical output.

    - ``tuple[int, ...]`` / ``Tuple[T, ...]`` → single variadic output.
    - ``tuple[tuple[...], tuple[...], ...]`` where every outer arg is itself a tuple
      annotation → fixed output count N, one inner shape per output.
    - Otherwise the whole annotation is a single output shape (fixed rank or variadic).
    """
    if not _is_tuple_annotation(ret_ann):
        raise TypeError(f"{ctx}: expected tuple[..., ...] return annotation, got {ret_ann!r}")
    targs = get_args(ret_ann)
    if not targs:
        raise TypeError(f"{ctx}: tuple annotation must have at least one element")
    if len(targs) == 2 and targs[1] is Ellipsis:
        return [_infer_shape_tuple_meta(ret_ann, ctx)]
    if all(_is_tuple_annotation(a) for a in targs):
        return [_infer_shape_tuple_meta(a, f"{ctx} output[{i}]") for i, a in enumerate(targs)]
    return [_infer_shape_tuple_meta(ret_ann, ctx)]


def _to_cpp_type(py_ann: Any) -> str:
    """Map a Python type annotation to a C++ type string (recursively for tuple/list)."""
    if py_ann == int:
        return "int64_t"  # Python int -> fixed-width C++ (aligns with gert::Shape dimensions)
    if py_ann == float:
        return "float"
    if py_ann == bool:
        return "bool"
    origin = get_origin(py_ann)
    if origin in _TUPLE_ORIGINS:
        targs = get_args(py_ann)
        if len(targs) == 2 and targs[1] is Ellipsis:
            elem = _to_cpp_type(targs[0])
            return f"std::vector<{elem}>"
        inner = ", ".join(_to_cpp_type(a) for a in targs)
        return f"std::tuple<{inner}>"
    if origin is list:
        targs = get_args(py_ann)
        if not targs:
            raise TypeError(f"list annotation must specify element type: {py_ann!r}")
        return f"std::vector<{_to_cpp_type(targs[0])}>"
    raise TypeError(f"Unsupported annotation for C++ mapping: {py_ann!r}")


def _generate_pybind_wrapper(
    func: Callable,
    cpp_func_name: str,
    *,
    include_preamble: bool = True,
) -> str:
    """Generate C++ pybind11 wrapper that runs the given Python function.

    If include_preamble is False, emit the function inside an anonymous namespace
    (for embedding in a TU that already includes pybind headers and
    ``namespace py = pybind11``).
    """
    py_source = _unwrap_decorated_func_source(inspect.getsource(func))
    py_func_name = _unwrap_decorated_func_name(func.__name__)
    py_sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    def ann_for(p: inspect.Parameter):
        if p.name in hints:
            return hints[p.name]
        return p.annotation

    param_cpp_types = [_to_cpp_type(ann_for(py_arg)) for py_arg in py_sig.parameters.values()]
    cpp_args_list = ", ".join(
        [f"{param_cpp_types[i]} {py_arg.name}" for i, py_arg in enumerate(py_sig.parameters.values())]
    )
    ret_ann = hints.get("return", py_sig.return_annotation)
    cpp_return_type = _to_cpp_type(ret_ann)

    pybind_args_list = ", ".join([f"py::cast({py_arg.name})" for py_arg in py_sig.parameters.values()])

    fn_block = f"""{cpp_return_type} {cpp_func_name}({cpp_args_list}) {{
    py::gil_scoped_acquire gil;

    const std::string py_source = R"(
{py_source}
)";

    py::module m = py::module_::import("__main__");
    py::dict globals = m.attr("__dict__");
    py::exec(py_source, globals, globals);
    py::object {py_func_name}_py = globals["{py_func_name}"];

    return {py_func_name}_py({pybind_args_list}).cast<{cpp_return_type}>();
}}"""

    if not include_preamble:
        return f"""namespace {{

{fn_block}

}}
"""

    vec_inc = _cpp_signature_needs_vector_include(cpp_return_type, param_cpp_types)
    vector_include = "#include <vector>\n" if vec_inc else ""

    return f"""// Auto-generated

#include <cstdint>
{vector_include}#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

{fn_block}
"""


def _parse_infer_shape_for_codegen(func: Callable) -> _InferShapeCodegenMeta:
    """Validate infer_shape annotations (fixed or variadic tuple) and return codegen metadata."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(f"infer_shape requires type annotations resolvable by get_type_hints: {exc}") from exc
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError("infer_shape must have a return annotation (tuple[..., ...])")
    ret_ann = hints.get("return", sig.return_annotation)
    out_metas = _parse_return_shape_outputs(ret_ann, "infer_shape return")
    params = list(sig.parameters.values())
    if not params:
        raise TypeError("infer_shape must accept at least one shape argument")
    input_modes: list[str] = []
    input_dims: list[Optional[int]] = []
    input_elem_cpp: list[Optional[str]] = []
    for p in params:
        ann = hints.get(p.name, p.annotation)
        im = _infer_shape_tuple_meta(ann, f"infer_shape parameter {p.name!r}")
        input_modes.append(im.kind)
        if im.kind == _INFER_SHAPE_MODE__FIXED:
            input_dims.append(im.dims)
            input_elem_cpp.append(None)
        else:
            input_dims.append(None)
            input_elem_cpp.append(im.elem_cpp)
    output_modes: list[str] = []
    output_dims: list[Optional[int]] = []
    output_elem_cpp: list[Optional[str]] = []
    for om in out_metas:
        if om.kind == _INFER_SHAPE_MODE__FIXED:
            output_modes.append(_INFER_SHAPE_MODE__FIXED)
            output_dims.append(om.dims)
            output_elem_cpp.append(None)
        else:
            output_modes.append(_INFER_SHAPE_MODE__VARIADIC)
            output_dims.append(None)
            output_elem_cpp.append(om.elem_cpp)
    return _InferShapeCodegenMeta(
        sig=sig,
        param_names=[p.name for p in params],
        input_modes=input_modes,
        input_dims=input_dims,
        input_elem_cpp=input_elem_cpp,
        output_modes=output_modes,
        output_dims=output_dims,
        output_elem_cpp=output_elem_cpp,
        py_func_name=_unwrap_decorated_func_name(func.__name__),
        cpp_bind_name=_snake_case_to_camel_case(_FUNC_NAME__INFER_SHAPE),
    )


def _cpp_signature_needs_vector_include(cpp_return_type: str, parameter_cpp_types: list[str]) -> bool:
    """True if generated C++ uses ``std::vector`` (variadic tuple mapping)."""
    if "std::vector" in cpp_return_type:
        return True
    return any("std::vector" in t for t in parameter_cpp_types)


def _parse_calc_workspace_for_codegen(func: Callable) -> _CalcWorkspaceCodegenMeta:
    """Parse calc_workspace function for executor C++ glue (shape-only v1).

    Contract: parameter annotations mirror infer_shape (fixed or variadic tuple[int, ...]),
    return annotation is int (workspace size in bytes).
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(f"calc_workspace requires type annotations resolvable by get_type_hints: {exc}") from exc
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError("calc_workspace must have a return annotation (int)")
    ret_ann = hints.get("return", sig.return_annotation)
    if ret_ann is not int:
        raise TypeError(f"calc_workspace return annotation must be int, got {ret_ann!r}")
    params = list(sig.parameters.values())
    if not params:
        raise TypeError("calc_workspace must accept at least one shape argument")
    input_modes: list[str] = []
    input_dims: list[Optional[int]] = []
    input_elem_cpp: list[Optional[str]] = []
    for p in params:
        ann = hints.get(p.name, p.annotation)
        im = _infer_shape_tuple_meta(ann, f"calc_workspace parameter {p.name!r}")
        input_modes.append(im.kind)
        if im.kind == _INFER_SHAPE_MODE__FIXED:
            input_dims.append(im.dims)
            input_elem_cpp.append(None)
        else:
            input_dims.append(None)
            input_elem_cpp.append(_to_cpp_type(get_args(ann)[0]))
    cpp_bind_name = _snake_case_to_camel_case(_FUNC_NAME__CALC_WORKSPACE)
    return _CalcWorkspaceCodegenMeta(
        sig=sig,
        input_modes=input_modes,
        input_dims=input_dims,
        input_elem_cpp=input_elem_cpp,
        cpp_bind_name=cpp_bind_name,
    )


def _generate_infer_shape_pybind_embedded(func: Callable, *, embed_in_host: bool = False) -> str:
    """C++ pybind callable for infer_shape (C++ tuple types from annotations via _to_cpp_type).

    When embed_in_host is True, omit includes/py alias (host TU already provides them);
    the wrapper is placed in an anonymous namespace.
    """
    return _generate_pybind_wrapper(
        func,
        _snake_case_to_camel_case(_FUNC_NAME__INFER_SHAPE),
        include_preamble=not embed_in_host,
    )


def _generate_calc_workspace_pybind_embedded(func: Callable) -> str:
    """C++ pybind callable for calc_workspace (embedded in an executor TU).

    The wrapper is placed in an anonymous namespace and assumes that the
    including translation unit already provides pybind11 includes and
    ``namespace py = pybind11``.
    """
    return _generate_pybind_wrapper(
        func,
        _snake_case_to_camel_case(_FUNC_NAME__CALC_WORKSPACE),
        include_preamble=False,
    )


def _infer_shape_input_block_fixed(i: int, dim_count: int) -> str:
    """One fixed-rank input: rank check and ``std::make_tuple`` from ``gert::Shape``."""
    idx = ", ".join(f"(*in{i}_shape)[{j}]" for j in range(dim_count))
    return f"""    const gert::Shape* in{i}_shape = context->GetInputShape({i});
    if (in{i}_shape->GetDimNum() != static_cast<size_t>({dim_count})) {{
        return GRAPH_FAILED;
    }}
    auto in{i}_tuple = std::make_tuple({idx});"""


def _infer_shape_input_block_variadic(i: int, elem_cpp: str) -> str:
    """One variadic-rank input: fill ``std::vector<elem_cpp>`` from ``gert::Shape`` dims."""
    if elem_cpp == "int64_t":
        push_expr = f"(*in{i}_shape)[j]"
    else:
        push_expr = f"static_cast<{elem_cpp}>((*in{i}_shape)[j])"
    return f"""    const gert::Shape* in{i}_shape = context->GetInputShape({i});
    std::vector<{elem_cpp}> in{i}_vec;
    in{i}_vec.reserve(in{i}_shape->GetDimNum());
    for (size_t j = 0; j < in{i}_shape->GetDimNum(); ++j) {{
        in{i}_vec.push_back({push_expr});
    }}"""


def _infer_shape_ge_impl_body(meta: _InferShapeCodegenMeta) -> str:
    """Return C++ statements for InferShapeGeImpl's body (no surrounding braces).

    Reads each input from ``context->GetInputShape(i)``, calls the embedded pybind wrapper
    ``meta.cpp_bind_name``, and fills ``context->GetOutputShape(k)`` for each output.
    ``meta`` must come from ``_parse_infer_shape_for_codegen``.
    """
    blocks: list[str] = []
    for i, mode in enumerate(meta.input_modes):
        if mode == _INFER_SHAPE_MODE__FIXED:
            blocks.append(_infer_shape_input_block_fixed(i, meta.input_dims[i]))
        else:
            blocks.append(_infer_shape_input_block_variadic(i, meta.input_elem_cpp[i]))
    inputs_cpp = "\n".join(blocks)
    n_in = len(meta.input_modes)
    call_parts: list[str] = []
    for i in range(n_in):
        if meta.input_modes[i] == _INFER_SHAPE_MODE__FIXED:
            call_parts.append(f"in{i}_tuple")
        else:
            call_parts.append(f"in{i}_vec")
    call_args = ", ".join(call_parts)
    bind = meta.cpp_bind_name
    n_out = len(meta.output_modes)

    def _fixed_assign_from_inner(inner_expr: str, dim_count: int, out_idx: int) -> str:
        out_idx_list = ",\n\t\t".join(
            f"static_cast<int64_t>(std::get<{j}>({inner_expr}))" for j in range(dim_count)
        )
        return f"""    gert::Shape* out_shape_{out_idx} = context->GetOutputShape({out_idx});
    *out_shape_{out_idx} = gert::Shape{{
        {out_idx_list}
    }};"""

    def _variadic_assign_from_vec(vec_expr: str, elem_cpp: str, out_idx: int) -> str:
        if elem_cpp == "int64_t":
            assign_inner = f"(*out_shape_{out_idx})[j] = {vec_expr}[j];"
        else:
            assign_inner = f"(*out_shape_{out_idx})[j] = static_cast<int64_t>({vec_expr}[j]);"
        return f"""    gert::Shape* out_shape_{out_idx} = context->GetOutputShape({out_idx});
    out_shape_{out_idx}->SetDimNum({vec_expr}.size());
    for (size_t j = 0; j < {vec_expr}.size(); ++j) {{
        {assign_inner}
    }}"""

    if n_out == 1 and meta.output_modes[0] == _INFER_SHAPE_MODE__FIXED:
        out_d = meta.output_dims[0]
        assert out_d is not None
        out_idx = ",\n\t\t".join(
            f"static_cast<int64_t>(std::get<{j}>(out_tuple))" for j in range(out_d)
        )
        return f"""{inputs_cpp}
    auto out_tuple = {bind}({call_args});
    gert::Shape* out_shape = context->GetOutputShape(0);
    *out_shape = gert::Shape{{
        {out_idx}
    }};
    return GRAPH_SUCCESS;"""

    if n_out == 1 and meta.output_modes[0] == _INFER_SHAPE_MODE__VARIADIC:
        out_elem = meta.output_elem_cpp[0]
        assert out_elem is not None
        if out_elem == "int64_t":
            assign = f"""    auto out_vec = {bind}({call_args});
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(out_vec.size());
    for (size_t j = 0; j < out_vec.size(); ++j) {{
        (*out_shape)[j] = out_vec[j];
    }}"""
        else:
            assign = f"""    auto out_vec = {bind}({call_args});
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(out_vec.size());
    for (size_t j = 0; j < out_vec.size(); ++j) {{
        (*out_shape)[j] = static_cast<int64_t>(out_vec[j]);
    }}"""
        return f"""{inputs_cpp}
{assign}
    return GRAPH_SUCCESS;"""

    # Multiple outputs: pybind returns nested std::tuple; unpack per output index.
    out_assign_lines: list[str] = []
    for k in range(n_out):
        inner = f"std::get<{k}>(out_tuple)"
        if meta.output_modes[k] == _INFER_SHAPE_MODE__FIXED:
            od = meta.output_dims[k]
            assert od is not None
            out_assign_lines.append(_fixed_assign_from_inner(inner, od, k))
        else:
            elem = meta.output_elem_cpp[k]
            assert elem is not None
            out_assign_lines.append(_variadic_assign_from_vec(inner, elem, k))
    assigns = "\n".join(out_assign_lines)
    return f"""{inputs_cpp}
    auto out_tuple = {bind}({call_args});
{assigns}
    return GRAPH_SUCCESS;"""


def _generate_infer_shape_host_tu_for_test(func: Callable) -> str:
    """Assemble infer_shape host slice for compile tests: preamble, embed pybind, InferShapeGeImpl only.

    Callers must include a mock header (e.g. ``gert_ge_minimal.hpp``) before this fragment
    so ``gert::`` / ``ge::`` symbols resolve. Omits GE register headers and OpDef.
    """
    meta = _parse_infer_shape_for_codegen(func)
    pybind_block = _generate_infer_shape_pybind_embedded(func, embed_in_host=True) + "\n\n"
    infer_shape_ge_body = _infer_shape_ge_impl_body(meta)
    any_variadic = any(m == _INFER_SHAPE_MODE__VARIADIC for m in meta.output_modes) or any(
        m == _INFER_SHAPE_MODE__VARIADIC for m in meta.input_modes
    )
    vec_inc = "#include <vector>\n" if any_variadic else ""
    return f"""// Auto-generated test TU fragment (include mock gert/ge header first)

#include <cstdint>
#include <tuple>
{vec_inc}#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

{pybind_block}namespace ge {{

static ge::graphStatus InferShapeGeImpl(gert::InferShapeContext* context) {{
{infer_shape_ge_body}
}}

}}
"""


def _generate_workspace_block_for_calc_workspace(meta: _CalcWorkspaceCodegenMeta) -> str:
    """Return C++ statements that compute workspaceSize via calc_workspace (no surrounding braces).

    For v1, parameters mirror infer_shape metadata (fixed or variadic tuple). We only support
    constructing std::tuple<int64_t, ...> or std::vector<int64_t> from logical shapes.
    """
    blocks: list[str] = []
    for i, mode in enumerate(meta.input_modes):
        if mode == _INFER_SHAPE_MODE__FIXED:
            dim_count = meta.input_dims[i]
            assert dim_count is not None
            idx = ", ".join(f"in{i}_shape[{j}]" for j in range(dim_count))
            blocks.append(
                f"""        const auto* in{i}_tensor = ctx->GetInputTensor({i});
        const auto& in{i}_storage = in{i}_tensor->GetShape();
        const gert::Shape& in{i}_shape = in{i}_storage.GetShape();
        if (in{i}_shape.GetDimNum() != static_cast<size_t>({dim_count})) {{
            return GRAPH_FAILED;
        }}
        auto in{i}_tuple = std::make_tuple({idx});"""
            )
        else:
            elem_cpp = meta.input_elem_cpp[i]
            assert elem_cpp is not None
            if elem_cpp == "int64_t":
                push_expr = f"in{i}_shape[j]"
            else:
                push_expr = f"static_cast<{elem_cpp}>(in{i}_shape[j])"
            blocks.append(
                f"""        const auto* in{i}_tensor = ctx->GetInputTensor({i});
        const auto& in{i}_storage = in{i}_tensor->GetShape();
        const gert::Shape& in{i}_shape = in{i}_storage.GetShape();
        std::vector<{elem_cpp}> in{i}_vec;
        in{i}_vec.reserve(in{i}_shape.GetDimNum());
        for (size_t j = 0; j < in{i}_shape.GetDimNum(); ++j) {{
            in{i}_vec.push_back({push_expr});
        }}"""
            )
    call_args: list[str] = []
    for i, mode in enumerate(meta.input_modes):
        if mode == _INFER_SHAPE_MODE__FIXED:
            call_args.append(f"in{i}_tuple")
        else:
            call_args.append(f"in{i}_vec")
    call_args_str = ", ".join(call_args)
    per_input = "\n".join(blocks)
    return f"""{per_input}
        auto ws = {meta.cpp_bind_name}({call_args_str});
        if (ws < 0) {{
            return GRAPH_FAILED;
        }}
        size_t workspaceSize = static_cast<size_t>(ws);"""


def _infer_dtype_rule_from_ast(
    ret_node: ast.expr,
    param_names: list[str],
) -> _InferDTypeRule:
    if isinstance(ret_node, ast.Name) and ret_node.id in param_names:
        idx = param_names.index(ret_node.id)
        return _InferDTypeRule(copy_input_index=idx, const_ge_dtype=None)
    if isinstance(ret_node, ast.Attribute) and isinstance(ret_node.value, ast.Name) and ret_node.value.id == "torch":
        import torch

        if not hasattr(torch, ret_node.attr):
            raise TypeError(
                f"{_FUNC_NAME__INFER_DTYPE} constant rule uses unknown torch dtype: torch.{ret_node.attr}"
            )
        torch_dtype = getattr(torch, ret_node.attr)
        ge_dtype = _torch_dtype_to_ge_dtype(torch_dtype)
        return _InferDTypeRule(copy_input_index=None, const_ge_dtype=ge_dtype)
    raise TypeError(
        f"{_FUNC_NAME__INFER_DTYPE} each output rule must be 'return <input_param>' or 'return torch.<dtype>'"
    )


def _parse_infer_dtype_for_codegen(func: Callable, *, output_count: int) -> _InferDTypeCodegenMeta:
    """Parse infer_dtype and return metadata containing one rule per output.

    - Single output: ``return <param>`` or ``return torch.<dtype>`` (or a one-element ``return (x,)``).
    - Multiple outputs: ``return (rule0, ..., rule_{N-1})`` with exactly *output_count* elements.
    """
    src = _unwrap_decorated_func_source(inspect.getsource(func))
    tree = ast.parse(src)
    func_def = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if func_def is None:
        raise TypeError(f"{_FUNC_NAME__INFER_DTYPE} must be a function")
    param_names = [arg.arg for arg in func_def.args.args]
    if not func_def.body or not isinstance(func_def.body[0], ast.Return):
        raise TypeError(f"{_FUNC_NAME__INFER_DTYPE} must consist of a single return statement for v1")
    ret = func_def.body[0].value

    if output_count == 1:
        if isinstance(ret, ast.Tuple) and len(ret.elts) == 1:
            return _InferDTypeCodegenMeta(rules=[_infer_dtype_rule_from_ast(ret.elts[0], param_names)])
        return _InferDTypeCodegenMeta(rules=[_infer_dtype_rule_from_ast(ret, param_names)])

    if not isinstance(ret, ast.Tuple) or len(ret.elts) != output_count:
        raise TypeError(
            f"{_FUNC_NAME__INFER_DTYPE} with {output_count} outputs must return a tuple of "
            f"{output_count} elements (one rule per output)"
        )
    return _InferDTypeCodegenMeta(rules=[_infer_dtype_rule_from_ast(el, param_names) for el in ret.elts])


def _generate_op_custom_plugin_cpp(
    op_type: str,
    *,
    framework_type: str,
) -> str:
    """Emit domi plugin registration; *framework_type* is the ``FrameworkType`` enum token (e.g. ``ONNX``)."""
    _validate_op_type_identifier(op_type)
    parse_fn = f"ParseParam{op_type}"
    return f"""// Auto-generated

#include "register/register.h"

namespace domi {{
// Parsing onnx params
Status {parse_fn}(const Message* op_src, ge::Operator& op_dest) {{
    return SUCCESS;
}}

REGISTER_CUSTOM_OP("{op_type}")
    .FrameworkType({framework_type})
    .OriginOpType("{op_type}")
    .ParseParamsByOperator({parse_fn});
}}
"""


def _generate_op_custom_def_cpp(
    infer_shape_func: Callable,
    infer_dtype_func: Callable,
    *,
    op_type: str,
    dtypes: list[Any],
) -> str:
    """GE OpDef TU under ``op_host``: OpDef plus InferShape / InferDataType with embedded pybind."""
    _validate_op_type_identifier(op_type)
    if not dtypes:
        raise ValueError("dtypes must be a non-empty list")

    infer_shape_meta = _parse_infer_shape_for_codegen(infer_shape_func)
    n_out = len(infer_shape_meta.output_modes)
    infer_dtype_meta = _parse_infer_dtype_for_codegen(infer_dtype_func, output_count=n_out)
    if len(infer_dtype_meta.rules) != n_out:
        raise ValueError("internal: infer_dtype rule count must match infer_shape output count")

    input_defs: list[str] = []
    for i, dtype in enumerate(dtypes):
        ge_dtype = _torch_dtype_to_ge_dtype(dtype)
        input_defs.append(
            f"""        this->Input("in{i}")
            .ParamType(REQUIRED)
            .DataType({{{ge_dtype}}})
            .Format({{ge::FORMAT_ND}});"""
        )

    output_defs: list[str] = []
    infer_dtype_lines: list[str] = []
    for k, infer_dtype_rule in enumerate(infer_dtype_meta.rules):
        if infer_dtype_rule.const_ge_dtype is not None:
            ge_dt = infer_dtype_rule.const_ge_dtype
            infer_dtype_lines.append(f"    context->SetOutputDataType({k}, {ge_dt});")
        else:
            assert infer_dtype_rule.copy_input_index is not None
            if infer_dtype_rule.copy_input_index >= len(dtypes):
                raise ValueError("infer_dtype requested input index out of range for dtypes")
            ge_dt = _torch_dtype_to_ge_dtype(dtypes[infer_dtype_rule.copy_input_index])
            infer_dtype_lines.append(
                f"    context->SetOutputDataType({k}, context->GetInputDataType({infer_dtype_rule.copy_input_index}));"
            )
        output_defs.append(
            f"""        this->Output("out{k}")
            .ParamType(REQUIRED)
            .DataType({{{ge_dt}}})
            .Format({{ge::FORMAT_ND}});"""
        )
    infer_dtype_ge_body = "\n".join(infer_dtype_lines) + "\n    return GRAPH_SUCCESS;"
    io_defs = "\n".join(input_defs + output_defs)

    pybind_block = _generate_infer_shape_pybind_embedded(
        infer_shape_func,
        embed_in_host=True,
    ) + "\n\n"
    infer_shape_ge_body = _infer_shape_ge_impl_body(infer_shape_meta)
    any_variadic = any(m == _INFER_SHAPE_MODE__VARIADIC for m in infer_shape_meta.output_modes) or any(
        m == _INFER_SHAPE_MODE__VARIADIC for m in infer_shape_meta.input_modes
    )
    vec_inc = "#include <vector>\n" if any_variadic else ""

    return f"""// Auto-generated

#include <cstdint>
#include <tuple>
{vec_inc}#include "register/op_def_registry.h"
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

{pybind_block}namespace ge {{

static ge::graphStatus InferShapeGeImpl(gert::InferShapeContext* context) {{
{infer_shape_ge_body}
}}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {{
{infer_dtype_ge_body}
}}

namespace ops {{
class {op_type} : public OpDef {{
public:
    explicit {op_type}(const char *name) : OpDef(name) {{
{io_defs}
        this->SetInferShape(InferShapeGeImpl);
        this->SetInferDataType(InferDataType);
        this->AICore().AddConfig("kirinx90");
    }}
}};

OP_ADD({op_type});
}} // namespace ops
}} // namespace ge
"""


def _custom_executor_class_cpp(op_type: str, workspace_block: str) -> str:
    """Return C++ for the sinkable executor class named *op_type* plus ``REG_AUTO_MAPPING_OP``."""
    _validate_op_type_identifier(op_type)
    return f"""using namespace ge;
using namespace gert;
class {op_type} : public SinkableExecuteOp {{
public:
    ge::graphStatus PrepareExecute(gert::SinkableOpExecutionContext *ctx)
    {{
        // get input & output
        printf("Entering PrepareExecute\\n");
        size_t inputNum = ctx->GetComputeNodeInputNum();
        size_t outputNum = ctx->GetComputeNodeOutputNum();

{workspace_block}
        // args layout: [out0]...[out_{{K-1}}][in0]...[in_{{M-1}}][workspace]
        size_t argsSize = sizeof(int64_t) * (outputNum + inputNum + 1);
        void* args = malloc(argsSize);
        int64_t* p = reinterpret_cast<int64_t*>(args);
        for (size_t k = 0; k < outputNum; ++k) {{
            p[k] = reinterpret_cast<int64_t>(ctx->GetOutputTensor(k)->GetAddr());
        }}
        for (size_t i = 0; i < inputNum; ++i) {{
            p[outputNum + i] = reinterpret_cast<int64_t>(ctx->GetInputTensor(i)->GetAddr());
        }}
        void *workspaceAddr = ctx->MallocWorkspace(workspaceSize);
        p[outputNum + inputNum] = reinterpret_cast<int64_t>(workspaceAddr);

        ctx->HostArgsToDevice(args, argsSize);

        std::vector<size_t> inputOffsets(inputNum);
        const size_t outBytes = sizeof(int64_t) * outputNum;
        for (size_t i = 0; i < inputNum; ++i) {{
            inputOffsets[i] = outBytes + i * sizeof(int64_t);
        }}
        (void)ctx->SpecifyToOffset(SinkableOpIo::kInput, inputOffsets.data(), inputNum);
        (void)ctx->SpecifyToOffset(SinkableOpIo::kOutput, 0, outputNum);

        return 0;
    }}

    graphStatus Execute(gert::EagerOpExecutionContext *ctx)
    {{
        return 0;
    }}

    graphStatus Compile()
    {{
        compiled_bin_path_ = "fixme/stub/path";
        return 0;
    }}

private:
    std::string compiled_bin_path_;
}};

REG_AUTO_MAPPING_OP({op_type});

"""


def _generate_custom_executor_cpp(
    op_type: str,
    *,
    calc_workspace_func: Callable,
    for_compile_test: bool = False,
) -> str:
    """Assemble the full executor TU, embedding the calc_workspace pybind wrapper and executor class.

    When *for_compile_test* is True, omit GE/runtime headers (``graph/custom_op.h``, ACL, etc.);
    tests are expected to include fixture mocks and standard / pybind headers only.
    """
    meta = _parse_calc_workspace_for_codegen(calc_workspace_func)
    workspace_block = _generate_workspace_block_for_calc_workspace(meta)
    pybind_block = _generate_calc_workspace_pybind_embedded(calc_workspace_func)
    if for_compile_test:
        preamble = """// Auto-generated

#include <vector>
#include <utility>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

"""
    else:
        preamble = """// Auto-generated

#include "graph/custom_op.h"
#include "exe_graph/runtime/sinkable_op_execution_context.h"
#include <vector>
#include <utility>
#include "acl/acl_rt.h"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

"""
    return f"""{preamble}{pybind_block}

{_custom_executor_class_cpp(op_type, workspace_block)}"""
