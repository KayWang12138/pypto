import inspect
import re
import typing
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_args, get_origin, get_type_hints

from ..helpers import (
    _FUNC_NAME__INFER_SHAPE,
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
# Tuple-shape metadata ``kind`` and parse result ``input_modes`` / ``output_mode`` values.
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
    output_mode: str
    output_dims: Optional[int]
    output_elem_cpp: Optional[str]
    py_func_name: str
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
    out_meta = _infer_shape_tuple_meta(ret_ann, "infer_shape return")
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
    if out_meta.kind == _INFER_SHAPE_MODE__FIXED:
        output_mode = _INFER_SHAPE_MODE__FIXED
        output_dims: Optional[int] = out_meta.dims
        output_elem_cpp: Optional[str] = None
    else:
        output_mode = _INFER_SHAPE_MODE__VARIADIC
        output_dims = None
        output_elem_cpp = out_meta.elem_cpp
    return _InferShapeCodegenMeta(
        sig=sig,
        param_names=[p.name for p in params],
        input_modes=input_modes,
        input_dims=input_dims,
        input_elem_cpp=input_elem_cpp,
        output_mode=output_mode,
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
    ``meta.cpp_bind_name``, and assigns ``*context->GetOutputShape(0)``.
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
    call_parts = []
    for i in range(n_in):
        if meta.input_modes[i] == _INFER_SHAPE_MODE__FIXED:
            call_parts.append(f"in{i}_tuple")
        else:
            call_parts.append(f"in{i}_vec")
    call_args = ", ".join(call_parts)
    bind = meta.cpp_bind_name

    if meta.output_mode == _INFER_SHAPE_MODE__FIXED:
        out_d = meta.output_dims
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

    out_elem = meta.output_elem_cpp
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


def _generate_infer_shape_host_tu_for_test(func: Callable) -> str:
    """Assemble infer_shape host slice for compile tests: preamble, embed pybind, InferShapeGeImpl only.

    Callers must include a mock header (e.g. ``gert_ge_minimal.hpp``) before this fragment
    so ``gert::`` / ``ge::`` symbols resolve. Omits GE register headers and OpDef.
    """
    meta = _parse_infer_shape_for_codegen(func)
    pybind_block = _generate_infer_shape_pybind_embedded(func, embed_in_host=True) + "\n\n"
    infer_shape_ge_body = _infer_shape_ge_impl_body(meta)
    any_variadic = meta.output_mode == _INFER_SHAPE_MODE__VARIADIC or any(
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
    infer_shape_func: Optional[Callable] = None,
    *,
    op_type: str,
    dtypes: list[Any],
) -> str:
    """GE OpDef TU under ``op_host``: OpDef stub plus optional InferShape via gert::Shape and embedded pybind."""
    _validate_op_type_identifier(op_type)
    if not dtypes:
        raise ValueError("dtypes must be a non-empty list")

    input_defs = []
    for i, dtype in enumerate(dtypes):
        ge_dtype = _torch_dtype_to_ge_dtype(dtype)
        input_defs.append(
            f"""        this->Input("in{i}")
            .ParamType(REQUIRED)
            .DataType({{{ge_dtype}}})
            .Format({{ge::FORMAT_ND}});"""
        )
    output_dtype = _torch_dtype_to_ge_dtype(dtypes[0])
    output_def = f"""        this->Output("out0")
            .ParamType(REQUIRED)
            .DataType({{{output_dtype}}})
            .Format({{ge::FORMAT_ND}});"""
    io_defs = "\n".join(input_defs + [output_def])

    if infer_shape_func is None:
        infer_shape_ge_body = """    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;"""
        pybind_block = ""
        vec_inc = ""
    else:
        meta = _parse_infer_shape_for_codegen(infer_shape_func)
        pybind_block = _generate_infer_shape_pybind_embedded(
            infer_shape_func,
            embed_in_host=True,
        ) + "\n\n"
        infer_shape_ge_body = _infer_shape_ge_impl_body(meta)
        any_variadic = meta.output_mode == _INFER_SHAPE_MODE__VARIADIC or any(
            m == _INFER_SHAPE_MODE__VARIADIC for m in meta.input_modes
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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


def _custom_executor_class_cpp(op_type: str) -> str:
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

        size_t workspaceSize = 1024; // TODO: replace with call to pybind wrapper calc_workspace.cpp::CalcWorkspace()
        // args layout: [output0][input0]...[inputN-1][workspace]
        size_t argsSize = sizeof(int64_t) * (1 + inputNum + 1);
        void* args = malloc(argsSize);
        int64_t* p = reinterpret_cast<int64_t*>(args);
        p[0] = reinterpret_cast<int64_t>(ctx->GetOutputTensor(0)->GetAddr());
        for (size_t i = 0; i < inputNum; ++i) {{
            p[1 + i] = reinterpret_cast<int64_t>(ctx->GetInputTensor(i)->GetAddr());
        }}
        void *workspaceAddr = ctx->MallocWorkspace(workspaceSize);
        p[1 + inputNum] = reinterpret_cast<int64_t>(workspaceAddr);

        ctx->HostArgsToDevice(args, argsSize);

        std::vector<size_t> inputOffsets(inputNum);
        for (size_t i = 0; i < inputNum; ++i) {{
            inputOffsets[i] = (i == 0) ? 8 : (inputOffsets[i - 1] + 8);
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


def _custom_executor_class_cpp_for_test(op_type: str) -> str:
    """Return executor class + REG macro for compile tests (no GE/ACL includes)."""
    return _custom_executor_class_cpp(op_type)


def _generate_custom_executor_cpp(op_type: str) -> str:
    return f"""// Auto-generated

#include "graph/custom_op.h"
#include "exe_graph/runtime/sinkable_op_execution_context.h"
#include <vector>
#include <utility>
#include "acl/acl_rt.h"

{_custom_executor_class_cpp(op_type)}"""
