# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0
"""Helpers for turning an extracted cpp_sources tree into a buildable project.

These functions are the library-side of ``examples/export/load_and_compile.py``:
they do pure file-tree transforms (copy / patch / template emission) that
prepare the generated cpp for compilation. The ``cmake`` invocation itself
stays in the example (no subprocess dependencies enter the library).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from .layout import load_cpp_layout


__all__ = (
    "patch_pybind_wrapper_cpp_files_under",
    "collect_required_cpp_files",
    "generate_bindings_cpp",
    "generate_cmakelists",
    "default_ascend_home",
)


# ---------------------------------------------------------------------------
# pybind embed -> gil_scoped_acquire patching
# ---------------------------------------------------------------------------


def _cpp_uses_pybind_embed_style(text: str) -> bool:
    return (
        "#include <pybind11/embed.h>" in text
        or "py::scoped_interpreter guard{};" in text
    )


def _patch_generated_cpp_if_needed(cpp_path: Path) -> None:
    """Rewrite ``py::scoped_interpreter`` to ``py::gil_scoped_acquire`` in-place.

    Generated files use ``#include <pybind11/embed.h>`` + ``py::scoped_interpreter guard{};``
    — fine for standalone binaries, wrong for a Python extension module where the interpreter
    is already running. Swap to ``<pybind11/eval.h>`` + ``py::gil_scoped_acquire``.
    """
    text = cpp_path.read_text(encoding="utf-8")
    changed = False

    if "#include <pybind11/embed.h>" in text:
        text = text.replace(
            "#include <pybind11/embed.h>",
            "#include <pybind11/pybind11.h>\n#include <pybind11/eval.h>",
        )
        changed = True

    if "py::scoped_interpreter guard{};" in text:
        text = text.replace(
            "py::scoped_interpreter guard{};",
            "py::gil_scoped_acquire gil;",
        )
        changed = True

    if changed:
        cpp_path.write_text(text, encoding="utf-8")
        print(f"[PATCH] Updated {cpp_path}")


def patch_pybind_wrapper_cpp_files_under(root: Path) -> None:
    """Walk *root* and rewrite every cpp that uses ``scoped_interpreter`` to use
    ``gil_scoped_acquire``. Safe to re-run (no-op on already-patched files)."""
    for cpp_path in sorted(Path(root).rglob("*.cpp")):
        try:
            text = cpp_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _cpp_uses_pybind_embed_style(text):
            _patch_generated_cpp_if_needed(cpp_path)


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------


def _collect_cpp_paths_from_layout(layout: dict, base: Path) -> dict:
    """Map layout key -> resolved source Path for the 3 canonical cpp TUs."""
    keys = ("op_custom_def", "executor_tu", "onnx_plugin")
    out = {}
    for k in keys:
        rel = layout.get(k)
        if not rel:
            raise KeyError(f"cpp_layout.json missing required key {k!r}")
        out[k] = base / rel
    return out


def collect_required_cpp_files(extracted_dir: Path, final_src_dir: Path) -> dict:
    """Copy op_def / executor / onnx_plugin cpps into *final_src_dir*, return resolved names.

    Applies the pybind embed -> acquire patch on copied files.
    """
    final_src_dir = Path(final_src_dir)
    final_src_dir.mkdir(parents=True, exist_ok=True)

    layout, base = load_cpp_layout(Path(extracted_dir))
    cpp_paths = _collect_cpp_paths_from_layout(layout, base)

    copied: dict = {}
    for key, src in cpp_paths.items():
        dst = final_src_dir / src.name
        shutil.copy2(src, dst)
        print(f"[COPY] {src} -> {dst}")
        copied[key] = dst

    patch_pybind_wrapper_cpp_files_under(final_src_dir)
    return copied


# ---------------------------------------------------------------------------
# bindings.cpp + CMakeLists.txt emission
# ---------------------------------------------------------------------------


def generate_bindings_cpp(dst: Path, module_name: str) -> None:
    """Emit a stub pybind module so the generated cpp tree has a
    ``PYBIND11_MODULE`` entry point and can be linked as a Python extension.

    The generated TUs' top-level functions (``inferShape`` etc.) live in
    anonymous namespaces so they cannot be forward-declared or re-exposed
    from here. The .so's value is end-to-end compile + link validation
    against Ascend libs; the Python-visible surface is intentionally empty.
    """
    content = f'''#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE({module_name}, m) {{
    m.doc() = "Generated PyPTO operator runtime module (stub bindings)";
}}
'''
    Path(dst).write_text(content, encoding="utf-8")
    print(f"[GEN] {dst}")


def generate_cmakelists(
    dst: Path,
    module_name: str,
    *,
    cpp_source_names: List[str],
    ascend_home: Path,
) -> None:
    """Emit CMakeLists.txt wiring all generated cpp sources + Ascend includes/libs."""
    sources_block = "\n".join(f"    src/{name}" for name in ["bindings.cpp", *cpp_source_names])

    ascend_lib_dir = Path(ascend_home) / "lib64"
    ascend_inc_dir = Path(ascend_home) / "include"

    content = f'''cmake_minimum_required(VERSION 3.15)
project({module_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(ASCEND_HOME "{ascend_home}")
set(ASCEND_INCLUDE_DIR "{ascend_inc_dir}")
set(ASCEND_LIB_DIR "{ascend_lib_dir}")

pybind11_add_module({module_name}
{sources_block}
)

target_include_directories({module_name} PRIVATE ${{ASCEND_INCLUDE_DIR}})
target_link_directories({module_name} PRIVATE ${{ASCEND_LIB_DIR}})
target_link_libraries({module_name} PRIVATE
    graph
    register
    exe_graph
    ascendcl
)
set_target_properties({module_name} PROPERTIES
    INSTALL_RPATH "${{ASCEND_LIB_DIR}}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
'''
    Path(dst).write_text(content, encoding="utf-8")
    print(f"[GEN] {dst}")


# ---------------------------------------------------------------------------
# Ascend install discovery
# ---------------------------------------------------------------------------


def default_ascend_home() -> Path:
    """Return the arch-specific Ascend install subtree containing ``include/`` + ``lib64/``.

    Honors ``$ASCEND_HOME_PATH``; falls back to ``/usr/local/Ascend/cann``. If the
    value is already arch-specific it is returned unchanged.
    """
    home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
    home_path = Path(home)
    for arch in ("aarch64-linux", "x86_64-linux"):
        sub = home_path / arch
        if (sub / "include").is_dir():
            return sub
    return home_path
