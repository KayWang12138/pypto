# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Optional compile-and-run tests for export C++ codegen.

Uses a single :func:`_compile_and_run` with :class:`CppCompileOptions`:

- ``embed_python=True``: infer_shape TU (pybind11 + Python embed).
- default (``embed_python=False``): TU fragments that do not use pybind / embedded Python.
- custom-executor compile tests use ``embed_python=True`` because generated ``calc_workspace`` glue embeds pybind and runs Python.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

import pytest
import torch


@dataclass(frozen=True)
class CppCompileOptions:
    """Options for compiling a generated C++ TU in tests."""

    embed_python: bool = False
    """If True, add pybind11 + Python embed link flags (infer_shape TU)."""
    extra_includes: tuple[str, ...] = ()
    """Extra ``-I`` directories (after fixtures include)."""
    extra_ldflags: tuple[str, ...] = ()
    """Extra linker flags (e.g. ``-lfoo``); appended before Python ldflags when embedding."""

from pypto.export.cpp import codegen as cpp_mod

_EXPORT_TEST_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _EXPORT_TEST_DIR / "fixtures"


def _load_samples_module(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    path = _EXPORT_TEST_DIR / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


infer_shape_samples = _load_samples_module("infer_shape_samples_ut_compile", "infer_shape_samples.py")
calc_workspace_samples = _load_samples_module("calc_workspace_samples_ut_compile", "calc_workspace_samples.py")


def _find_cxx() -> str | None:
    for key in (os.environ.get("CXX"), "g++", "clang++"):
        if key and shutil.which(key.split()[0]):
            return key
    return None


def _pybind_include() -> str | None:
    try:
        import pybind11

        return str(pybind11.get_include())
    except Exception:
        return None


def _python_config_cmd() -> str | None:
    exe = Path(sys.executable)
    name = exe.name
    candidate = exe.parent / f"{name}-config"
    if candidate.is_file():
        return str(candidate)
    for alt in ("python3-config", "python-config"):
        p = shutil.which(alt)
        if p:
            return p
    return None


def _python_embed_link_flags() -> list[str] | None:
    cfg = _python_config_cmd()
    if not cfg:
        return None
    try:
        out = subprocess.check_output(
            [cfg, "--ldflags", "--embed"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            out = subprocess.check_output([cfg, "--ldflags"], stderr=subprocess.DEVNULL, text=True)
            return out.split()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


def _python_cflags() -> list[str]:
    cfg = _python_config_cmd()
    if not cfg:
        return []
    try:
        out = subprocess.check_output([cfg, "--cflags"], stderr=subprocess.DEVNULL, text=True)
        return out.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _conda_link_flags() -> list[str]:
    """Return extra linker flags for conda Python lib directory, if present."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return []
    lib_dir = Path(conda_prefix) / "lib"
    if not lib_dir.is_dir():
        return []
    return [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]


def _compile_and_run(cpp_source: str, *, options: CppCompileOptions | None = None) -> None:
    opts = options if options is not None else CppCompileOptions()
    cxx = _find_cxx()
    if not cxx:
        pytest.skip("No C++ compiler")

    if opts.embed_python:
        pybind_inc = _pybind_include()
        ldflags = _python_embed_link_flags()
        if not pybind_inc or not ldflags:
            pytest.skip(
                "embed_python=True requires pybind11 include path and python-config --ldflags"
            )

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        src = tdir / "export_codegen_test.cpp"
        exe = tdir / "export_codegen_test_bin"
        src.write_text(cpp_source, encoding="utf-8")

        cmd: list[str] = [
            cxx,
            "-std=c++17",
            "-O0",
            f"-I{_FIXTURES_DIR}",
        ]
        for inc in opts.extra_includes:
            cmd.append(f"-I{inc}")

        if opts.embed_python:
            assert pybind_inc is not None and ldflags is not None  # guarded by skip above
            cmd.append(f"-I{pybind_inc}")
            cmd.extend(_python_cflags())

        cmd.append(str(src))

        if opts.embed_python:
            cmd.extend(_conda_link_flags())
            cmd.extend(opts.extra_ldflags)
            cmd.extend(ldflags)
        else:
            cmd.extend(opts.extra_ldflags)

        cmd.extend(["-o", str(exe)])

        label = "embed_python" if opts.embed_python else "native"
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"Compile failed ({label}):\n"
                + " ".join(cmd)
                + "\n"
                + (e.stderr or "")
                + "\n"
                + (e.stdout or "")
            )

        run_env = dict(os.environ)
        if opts.embed_python:
            conda_prefix = run_env.get("CONDA_PREFIX")
            if conda_prefix:
                conda_lib = str(Path(conda_prefix) / "lib")
                old_ld = run_env.get("LD_LIBRARY_PATH", "")
                run_env["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else conda_lib

        proc = subprocess.run([str(exe)], capture_output=True, text=True, env=run_env)
        if proc.returncode != 0:
            pytest.fail(
                f"Binary exited {proc.returncode} ({label})\n"
                f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
            )


def _custom_executor_test_main_cpp(
    n_inputs: int, n_outputs: int, *, op_type: str
) -> str:
    """Return C++ ``main()`` that validates ``PrepareExecute`` host-args layout (compile-only test).

    Exit codes are test-specific (not GE/graphStatus). When a check fails, ``main`` returns:

    Bands assume at most **10 outputs** and **10 inputs** so codes stay disjoint. Indexed failures use a
    base ending in ``0`` so the **ones digit** is the index (e.g. ``10 + k`` → ``10..19`` for ``k = 0..9``).

    - ``0`` — success
    - ``1`` — ``PrepareExecute`` returned non-zero
    - ``2`` — ``last_args().size()`` != ``n_outputs + n_inputs + 1`` (expected ``[outs…][ins…][workspace]``)
    - ``10 + k`` (``k < 10``) — output ``k`` pointer in ``last_args`` does not match ``GetOutputTensor(k)->GetAddr()``
    - ``20 + i`` (``i < 10``) — input ``i`` pointer does not match ``GetInputTensor(i)->GetAddr()``
    - ``40`` — workspace slot does not match ``workspace_ptr()``
    - ``50`` — ``input_offsets().size()`` != ``n_inputs``
    - ``60`` — ``input_offsets[0]`` != ``8 * n_outputs`` (first input starts after all output pointers)
    - ``70 + i`` (``i >= 1``, ``i < 10``) — ``input_offsets[i] != input_offsets[i-1] + 8`` (ones digit is ``i``)
    - ``80`` — ``last_output_spec_count()`` != ``n_outputs``
    """
    out_checks = "\n".join(
        f"  if (a[{k}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>("
        f"ctx.GetOutputTensor({k})->GetAddr()))) return {10 + k};"
        for k in range(n_outputs)
    )
    input_addr_checks = "\n".join(
        f"  if (a[{n_outputs + i}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>("
        f"ctx.GetInputTensor({i})->GetAddr()))) return {20 + i};"
        for i in range(n_inputs)
    )
    first_input_off = 8 * n_outputs
    offset_checks = "\n".join(
        (
            f"  if (off[0] != {first_input_off}u) return 60;"
            if i == 0
            else f"  if (off[{i}] != off[{i - 1}] + 8u) return {70 + i};"
        )
        for i in range(n_inputs)
    )
    total_slots = n_outputs + n_inputs + 1
    ws_idx = n_outputs + n_inputs
    return f"""int main() {{
  pybind11::scoped_interpreter guard{{}};
  {op_type} op;
  gert::MockSinkableOpExecutionContext ctx({n_inputs}, {n_outputs});
  ge::graphStatus st = op.PrepareExecute(&ctx);
  if (st != 0) return 1;
  const auto& a = ctx.last_args();
  if (a.size() != {total_slots}u) return 2;
{out_checks}
{input_addr_checks}
  if (a[{ws_idx}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(ctx.workspace_ptr()))) return 40;
  const auto& off = ctx.input_offsets();
  if (off.size() != static_cast<size_t>({n_inputs})) return 50;
{offset_checks}
  if (ctx.last_output_spec_count() != {n_outputs}u) return 80;
  return 0;
}}"""


def _full_custom_executor_test_tu(n_inputs: int, *, op_type: str, n_outputs: int = 1) -> str:
    main = _custom_executor_test_main_cpp(n_inputs, n_outputs, op_type=op_type)
    # Use the full executor TU generated by codegen (includes class and REG macro).
    # calc_workspace returns a positive byte estimate from mock shape and element size
    # (GeDataTypeElementSize → pybind int); MallocWorkSpace in fixtures ignores size.
    # One tensor input → one (shape, dtype_size) pair; glue reads GetInputTensor(0),
    # so the mock context must have n_inputs >= 1 (see test parametrization).
    def _ws(shape0: tuple[int, int], dtype_size: int) -> int:
        return int(shape0[0] * shape0[1] * dtype_size)

    exe_tu = cpp_mod._generate_custom_executor_cpp(
        op_type, calc_workspace_func=_ws, for_compile_test=True
    )
    return f"""#include "sinkable_executor_minimal.hpp"
#include <pybind11/embed.h>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <string>

{exe_tu}

{main}
"""


def _custom_executor_workspace_test_main_cpp(
    op_type: str, *, n_inputs: int, n_outputs: int = 1
) -> str:
    """Return a shorter C++ ``main()`` for workspace compile tests (fewer assertions than full layout test).

    Exit codes (same pointer bands as :func:`_custom_executor_test_main_cpp` for outputs/inputs; assumes
    at most **10** outputs and **10** inputs):

    - ``0`` — success
    - ``1`` — ``PrepareExecute`` returned non-zero
    - ``2`` — ``last_args().size()`` != ``n_outputs + n_inputs + 1``
    - ``10 + k`` — output ``k`` pointer mismatch (ones digit ``k`` for ``k < 10``)
    - ``20 + i`` — input ``i`` pointer mismatch (ones digit ``i`` for ``i < 10``)
    - ``90`` — workspace pointer mismatch (distinct from ``40`` in the full test so the two ``main()`` kinds are identifiable)
    """
    # For workspace-focused tests, we rely on the sinkable executor minimal
    # fixtures to expose a workspace pointer and basic invariants; we only
    # assert that PrepareExecute succeeds and does not crash for a shape that
    # exercises calc_workspace. *n_inputs* is the tensor count (PrepareExecute),
    # not the Python calc_workspace parameter count (2 * n_inputs for shape+dtype_size pairs).
    out_checks = "\n".join(
        f"  if (a[{k}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>("
        f"ctx.GetOutputTensor({k})->GetAddr()))) return {10 + k};"
        for k in range(n_outputs)
    )
    input_checks = "\n".join(
        f"  if (a[{n_outputs + i}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>("
        f"ctx.GetInputTensor({i})->GetAddr()))) return {20 + i};"
        for i in range(n_inputs)
    )
    total_slots = n_outputs + n_inputs + 1
    ws_idx = n_outputs + n_inputs
    return f"""int main() {{
  pybind11::scoped_interpreter guard{{}};
  {op_type} op;
  gert::MockSinkableOpExecutionContext ctx({n_inputs}, {n_outputs});
  ge::graphStatus st = op.PrepareExecute(&ctx);
  if (st != 0) return 1;
  const auto& a = ctx.last_args();
  if (a.size() != {total_slots}u) return 2;
{out_checks}
{input_checks}
  if (a[{ws_idx}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(ctx.workspace_ptr()))) return 90;
  return 0;
}}"""


def _full_custom_executor_workspace_tu(
    *,
    op_type: str,
    calc_workspace_func: Callable,
    n_inputs: int,
) -> str:
    # Use a calc_workspace implementation that depends on shape to exercise
    # the tuple/int64_t conversion path.
    exe_tu = cpp_mod._generate_custom_executor_cpp(
        op_type,
        calc_workspace_func=calc_workspace_func,
        for_compile_test=True,
    )
    main = _custom_executor_workspace_test_main_cpp(op_type, n_inputs=n_inputs, n_outputs=1)
    return f"""#include "sinkable_executor_minimal.hpp"
#include <pybind11/embed.h>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <string>

{exe_tu}

{main}
"""


def _normalize_infer_shape_outputs(out: tuple) -> list[tuple]:
    if not out:
        return []
    if isinstance(out[0], int):
        return [tuple(out)]
    return [tuple(x) for x in out]


def _build_main(
    fn: Callable[..., tuple],
    input_shapes: Sequence[Tuple[int, ...]],
) -> str:
    out = fn(*input_shapes)
    outputs = _normalize_infer_shape_outputs(out)
    lines = [
        "int main() {",
        "  pybind11::scoped_interpreter guard{};",
        "  gert::InferShapeContext ctx;",
    ]
    for i, shp in enumerate(input_shapes):
        br = ", ".join(str(int(x)) for x in shp)
        lines.append(f"  ctx.SetInput({i}, {{{br}}});")
    lines.extend(
        [
            "  ge::graphStatus st = ge::InferShapeGeImpl(&ctx);",
            "  if (st != ge::GRAPH_SUCCESS) return 1;",
        ]
    )
    ret_code = 2
    for oi, dims in enumerate(outputs):
        lines.append(f"  const auto& o{oi} = ctx.OutputShape({oi}u);")
        lines.append(f"  if (o{oi}.GetDimNum() != {len(dims)}u) return {ret_code};")
        ret_code += 1
        for j, v in enumerate(dims):
            lines.append(f"  if (o{oi}[{j}] != {int(v)}ll) return {ret_code};")
            ret_code += 1
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)


def _full_infer_shape_translation_unit(
    fn: Callable[..., tuple],
    input_shapes: Sequence[Tuple[int, ...]],
) -> str:
    tu = cpp_mod._generate_infer_shape_host_tu_for_test(fn)
    main = _build_main(fn, input_shapes)
    return f"""#include "gert_ge_minimal.hpp"
#include <pybind11/embed.h>
{tu}

{main}
"""


def _opdef_infer_pair_for_arity(n: int) -> tuple[Callable, Callable]:
    mapping: dict[int, tuple[Callable, Callable]] = {
        1: (infer_shape_samples.infer_shape_one_2d, infer_shape_samples.infer_dtype_one),
        2: (infer_shape_samples.infer_shape_two_by_two, infer_shape_samples.infer_dtype_two),
        3: (infer_shape_samples.infer_shape_three_4d, infer_shape_samples.infer_dtype_three),
    }
    return mapping[n]


def _full_op_custom_def_translation_unit(*, op_type: str, dtypes: Sequence[str]) -> str:
    n = len(dtypes)
    infer_shape_fn, infer_dtype_fn = _opdef_infer_pair_for_arity(n)
    host = cpp_mod._generate_op_custom_def_cpp(
        infer_shape_fn, infer_dtype_fn, op_type=op_type, dtypes=list(dtypes)
    )
    if n == 1:
        set_inputs = ["  ctx.SetInput(0, {2, 3});"]
    elif n == 2:
        set_inputs = ["  ctx.SetInput(0, {2, 3});", "  ctx.SetInput(1, {2, 3});"]
    else:
        set_inputs = [
            "  ctx.SetInput(0, {1, 1, 2, 3});",
            "  ctx.SetInput(1, {1, 1, 2, 3});",
            "  ctx.SetInput(2, {1, 1, 2, 3});",
        ]
    set_inputs_block = "\n".join(set_inputs)
    return f"""#include "gert_ge_minimal.hpp"
#include "register/op_def_registry.h"
#include <pybind11/embed.h>

{host}

int main() {{
  pybind11::scoped_interpreter guard{{}};
  ops::{op_type} op("x");
  (void)op;
  gert::InferShapeContext ctx;
{set_inputs_block}
  ge::graphStatus st = ge::InferShapeGeImpl(&ctx);
  if (st != ge::GRAPH_SUCCESS) return 1;
  gert::InferDataTypeContext dt_ctx;
  st = ge::InferDataType(&dt_ctx);
  if (st != ge::GRAPH_SUCCESS) return 2;
  return 0;
}}
"""


@pytest.mark.cpp_codegen
@pytest.mark.parametrize(
    ("fn", "inputs"),
    [
        (infer_shape_samples.infer_shape_two_by_two, ((32, 64), (10, 20))),
        (infer_shape_samples.infer_shape_4d_broadcast, ((1, 2, 3, 4), (1, 2, 3, 4))),
        (infer_shape_samples.infer_shape_sum_last, ((2, 3, 5), (2, 3, 7))),
        (infer_shape_samples.infer_shape_nd_identity, ((1, 2, 3, 4),)),
        (infer_shape_samples.infer_shape_two_outputs, ((5, 6), (7, 8))),
    ],
)
def test_compile_and_run_infer_shape_host(fn, inputs):
    src = _full_infer_shape_translation_unit(fn, inputs)
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize("n_inputs", [1, 3])
@pytest.mark.parametrize("n_outputs", [1, 2])
@pytest.mark.parametrize("op_type", ["Add", "MyKernel"])
def test_compile_custom_executor_prepare_execute(
    n_inputs: int, n_outputs: int, op_type: str
):
    src = _full_custom_executor_test_tu(
        n_inputs, op_type=op_type, n_outputs=n_outputs
    )
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize("op_type", ["Add", "MyKernel"])
@pytest.mark.parametrize(
    ("calc_ws_fn", "n_inputs"),
    [
        (calc_workspace_samples.calc_workspace_fixed, 2),
        (calc_workspace_samples.calc_workspace_variadic, 1),
    ],
)
def test_compile_custom_executor_calc_workspace(op_type: str, calc_ws_fn: Callable, n_inputs: int):
    src = _full_custom_executor_workspace_tu(
        op_type=op_type, calc_workspace_func=calc_ws_fn, n_inputs=n_inputs
    )
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize("op_type", ["Add", "MyKernel"])
@pytest.mark.parametrize(
    "dtypes",
    [
        ("torch.float16",),
        ("torch.float16", "torch.float32"),
        ("torch.float16", "torch.float32", "torch.bfloat16"),
    ],
)
def test_compile_and_run_full_op_custom_def_cpp(op_type: str, dtypes: Sequence[str]):
    src = _full_op_custom_def_translation_unit(op_type=op_type, dtypes=dtypes)
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))
