import inspect
import re
import typing
from typing import Any, Callable, Optional, get_args, get_origin, get_type_hints

from .helpers import _unwrap_decorated_func_source, _unwrap_decorated_func_name, _snake_case_to_camel_case

# This module is WIP and subject to changes (pending requirement clarifications)

__all__: tuple[str, ...] = ()

_OP_TYPE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TUPLE_ORIGINS = (tuple, typing.Tuple)


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


def _tuple_dim_count(ann: Any, ctx: str) -> int:
    """Return number of dimensions for a tuple[...] shape annotation."""
    if not _is_tuple_annotation(ann):
        raise TypeError(f"{ctx}: expected tuple[..., ...] annotation, got {ann!r}")
    args = get_args(ann)
    if not args:
        raise TypeError(f"{ctx}: tuple annotation must have at least one element")
    return len(args)


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

    cpp_args_list = ", ".join(
        [f"{_to_cpp_type(ann_for(py_arg))} {py_arg.name}" for py_arg in py_sig.parameters.values()]
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

    return f"""// Auto-generated

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

{fn_block}
"""


def _parse_infer_shape_for_codegen(func: Callable):
    """Validate infer_shape annotations (tuple only) and return codegen metadata."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(f"infer_shape requires type annotations resolvable by get_type_hints: {exc}") from exc
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError("infer_shape must have a return annotation (tuple[..., ...])")
    ret_ann = hints.get("return", sig.return_annotation)
    out_dims = _tuple_dim_count(ret_ann, "infer_shape return")
    params = list(sig.parameters.values())
    if not params:
        raise TypeError("infer_shape must accept at least one shape argument")
    input_dims = []
    for p in params:
        ann = hints.get(p.name, p.annotation)
        input_dims.append(_tuple_dim_count(ann, f"infer_shape parameter {p.name!r}"))
    return {
        "sig": sig,
        "param_names": [p.name for p in params],
        "input_dims": input_dims,
        "output_dims": out_dims,
        "py_func_name": _unwrap_decorated_func_name(func.__name__),
        "cpp_bind_name": _snake_case_to_camel_case("infer_shape"),
    }


def _generate_infer_shape_pybind_embedded(func: Callable, *, embed_in_host: bool = False) -> str:
    """C++ pybind callable for infer_shape (C++ tuple types from annotations via _to_cpp_type).

    When embed_in_host is True, omit includes/py alias (host TU already provides them);
    the wrapper is placed in an anonymous namespace.
    """
    return _generate_pybind_wrapper(
        func,
        _snake_case_to_camel_case("infer_shape"),
        include_preamble=not embed_in_host,
    )


def _infer_shape_input_block(i: int, dim_count: int) -> str:
    """One input's gert::Shape read, rank check, and std::make_tuple for InferShapeGeImpl."""
    idx = ", ".join(f"(*in{i}_shape)[{j}]" for j in range(dim_count))
    return f"""    const gert::Shape* in{i}_shape = context->GetInputShape({i});
    if (in{i}_shape->GetDimNum() != static_cast<size_t>({dim_count})) {{
        return GRAPH_FAILED;
    }}
    auto in{i}_tuple = std::make_tuple({idx});"""


def _infer_shape_ge_impl_body(meta: dict) -> str:
    """Return C++ statements for InferShapeGeImpl's body (no surrounding braces).

    Reads each input rank and dims from ``context->GetInputShape(i)`` (``gert::Shape``),
    calls the embedded pybind wrapper named ``meta['cpp_bind_name']`` (e.g. ``inferShape``),
    and assigns the result to ``*context->GetOutputShape(0)`` as ``gert::Shape``.
    ``meta`` must come from ``_parse_infer_shape_for_codegen``.
    """
    inputs_cpp = "\n".join(
        _infer_shape_input_block(i, d) for i, d in enumerate(meta["input_dims"])
    )
    n_in = len(meta["input_dims"])
    call_args = ", ".join(f"in{i}_tuple" for i in range(n_in))
    out_d = meta["output_dims"]
    out_idx = ",\n\t\t".join(
        f"static_cast<int64_t>(std::get<{j}>(out_tuple))" for j in range(out_d)
    )
    return f"""{inputs_cpp}
    auto out_tuple = {meta['cpp_bind_name']}({call_args});
    gert::Shape* out_shape = context->GetOutputShape(0);
    *out_shape = gert::Shape{{
        {out_idx}
    }};
    return GRAPH_SUCCESS;"""


def _generate_infer_shape_host_tu_for_test(func: Callable) -> str:
    """Assemble infer_shape host slice for compile tests: preamble, embed pybind, InferShapeGeImpl only.

    Callers must include a mock header (e.g. ``gert_ge_minimal.hpp``) before this fragment
    so ``gert::`` / ``ge::`` symbols resolve. Omits GE register headers and OpDef.
    """
    meta = _parse_infer_shape_for_codegen(func)
    pybind_block = _generate_infer_shape_pybind_embedded(func, embed_in_host=True) + "\n\n"
    infer_shape_ge_body = _infer_shape_ge_impl_body(meta)
    return f"""// Auto-generated test TU fragment (include mock gert/ge header first)

#include <cstdint>
#include <tuple>
#include <pybind11/pybind11.h>
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


def _generate_op_custom_def_cpp(infer_shape_func: Optional[Callable] = None, *, op_type: str) -> str:
    """GE OpDef TU under ``op_host``: OpDef stub plus optional InferShape via gert::Shape and embedded pybind."""
    _validate_op_type_identifier(op_type)
    if infer_shape_func is None:
        infer_shape_ge_body = """    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;"""
        pybind_block = ""
    else:
        meta = _parse_infer_shape_for_codegen(infer_shape_func)
        pybind_block = _generate_infer_shape_pybind_embedded(
            infer_shape_func,
            embed_in_host=True,
        ) + "\n\n"
        infer_shape_ge_body = _infer_shape_ge_impl_body(meta)

    return f"""// Auto-generated

#include <cstdint>
#include <tuple>
#include "register/op_def_registry.h"
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
    (void)context->GetInputDataType(0);
    return GRAPH_SUCCESS;
}}

namespace ops {{
class {op_type} : public OpDef {{
public:
    explicit {op_type}(const char *name) : OpDef(name) {{
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT16}})
            .Format({{ge::FORMAT_ND}});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT16}})
            .Format({{ge::FORMAT_ND}});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT16}})
            .Format({{ge::FORMAT_ND}});
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
