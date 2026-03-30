# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Optional compile-and-run tests for export C++ codegen.

Uses a single :func:`_compile_and_run` with :class:`CppCompileOptions`:

- ``embed_python=True``: infer_shape TU (pybind11 + Python embed).
- default (``embed_python=False``): custom executor TU (compiler only, no Python link).
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

_SAMPLES_PATH = Path(__file__).resolve().parent / "infer_shape_samples.py"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_samples():
    spec = importlib.util.spec_from_file_location("infer_shape_samples_ut_compile", _SAMPLES_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


samples = _load_samples()


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


def _custom_executor_test_main_cpp(n_inputs: int, *, op_type) -> str:
    input_addr_checks = "\n".join(
        f"  if (a[{1 + i}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>("
        f"ctx.GetInputTensor({i})->GetAddr()))) return {10 + i};"
        for i in range(n_inputs)
    )
    offset_checks = "\n".join(
        (
            "  if (off[0] != 8u) return 40;"
            if i == 0
            else f"  if (off[{i}] != off[{i - 1}] + 8u) return {41 + i};"
        )
        for i in range(n_inputs)
    )
    return f"""int main() {{
  {op_type} op;
  gert::MockSinkableOpExecutionContext ctx({n_inputs}, 1);
  ge::graphStatus st = op.PrepareExecute(&ctx);
  if (st != 0) return 1;
  const auto& a = ctx.last_args();
  if (a.size() != {1 + n_inputs + 1}u) return 2;
  if (a[0] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(ctx.GetOutputTensor(0)->GetAddr()))) return 3;
{input_addr_checks}
  if (a[{1 + n_inputs}] != static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(ctx.workspace_ptr()))) return 20;
  const auto& off = ctx.input_offsets();
  if (off.size() != static_cast<size_t>({n_inputs})) return 30;
{offset_checks}
  if (ctx.last_output_spec_count() != 1u) return 50;
  return 0;
}}"""


def _full_custom_executor_test_tu(n_inputs: int, *, op_type) -> str:
    fragment = cpp_mod._custom_executor_class_cpp_for_test(op_type)
    main = _custom_executor_test_main_cpp(n_inputs, op_type=op_type)
    return f"""#include "sinkable_executor_minimal.hpp"
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <string>

{fragment}

{main}
"""


def _build_main(
    fn: Callable[..., Tuple[int, ...]],
    input_shapes: Sequence[Tuple[int, ...]],
) -> str:
    out = fn(*input_shapes)
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
            "  const auto& o = ctx.OutputShape();",
            f"  if (o.GetDimNum() != {len(out)}u) return 2;",
        ]
    )
    for j, v in enumerate(out):
        lines.append(f"  if (o[{j}] != {int(v)}ll) return {3 + j};")
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)


def _full_infer_shape_translation_unit(
    fn: Callable[..., Tuple[int, ...]],
    input_shapes: Sequence[Tuple[int, ...]],
) -> str:
    tu = cpp_mod._generate_infer_shape_host_tu_for_test(fn)
    main = _build_main(fn, input_shapes)
    return f"""#include "gert_ge_minimal.hpp"
#include <pybind11/embed.h>
{tu}

{main}
"""


def _full_op_custom_def_translation_unit(*, op_type: str, dtypes: Sequence[str]) -> str:
    host = cpp_mod._generate_op_custom_def_cpp(op_type=op_type, dtypes=list(dtypes))
    return f"""#include "gert_ge_minimal.hpp"
#include "register/op_def_registry.h"

{host}

int main() {{
  ge::ops::{op_type} op("x");
  (void)op;
  gert::InferShapeContext ctx;
  ctx.SetInput(0, {{2, 3}});
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
        (samples.infer_shape_two_by_two, ((32, 64), (10, 20))),
        (samples.infer_shape_4d_broadcast, ((1, 2, 3, 4), (1, 2, 3, 4))),
        (samples.infer_shape_sum_last, ((2, 3, 5), (2, 3, 7))),
        (samples.infer_shape_nd_identity, ((1, 2, 3, 4),)),
    ],
)
def test_compile_and_run_infer_shape_host(fn, inputs):
    src = _full_infer_shape_translation_unit(fn, inputs)
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize("n_inputs", [0, 1, 3])
@pytest.mark.parametrize("op_type", ["Add", "MyKernel"])
def test_compile_custom_executor_prepare_execute(n_inputs: int, op_type: str):
    src = _full_custom_executor_test_tu(n_inputs, op_type=op_type)
    _compile_and_run(src, options=CppCompileOptions())


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
