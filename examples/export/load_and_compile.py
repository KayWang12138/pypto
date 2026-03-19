#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
import pypto
from torchair.ge._ge_graph import GeGraph

DOMAIN = "ai.onnx.contrib"
OP_TYPE__ADD = "AddPyptoCustomOp"
DOMAIN_OP_TYPE__ADD = f"{DOMAIN}::{OP_TYPE__ADD}"

// TODO: derive OP_TYPE from layout json
// FIXME: support ASC lib in makefile for "register/op_def_registry.h" and etc.

def run_cmd(cmd, cwd=None, env=None):
    print(f"\n[RUN] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def load_model_and_extract_node(path: Path):
    model_format = path.suffix.lstrip(".")
    print(f"\nLoading {model_format} model from: {path}")

    if model_format == "onnx":
        m = onnx.load(path)
        node = pypto.export.extract_node_from_onnx(
            onnx_model=m,
            domain=DOMAIN,
            op_type=OP_TYPE__ADD,
        )
    elif model_format == "air":
        with open(path, "rb") as f:
            ge = GeGraph(serialized_model_def=f.read())
            node = pypto.export.extract_node_from_ge_graph(
                ge_graph=ge,
                op_type=DOMAIN_OP_TYPE__ADD,
            )
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return model_format, node


def extract_cpp_sources(node, work_dir: Path):
    """
    Extract cpp sources from node into work_dir/src_extracted.
    Returns the extracted source directory path.
    """
    src_extract_dir = work_dir / "src_extracted"
    src_extract_dir.mkdir(parents=True, exist_ok=True)

    cpp_sources_zip_meta = pypto.export.extract_cpp_sources(
        node=node,
        out_dir=str(src_extract_dir),
        strict=True,
    )
    print(f"[INFO] cpp_sources_zip_meta = {cpp_sources_zip_meta}")

    return src_extract_dir


def find_cpp_files(search_dir: Path):
    cpp_files = list(search_dir.rglob("*.cpp"))
    if not cpp_files:
        raise FileNotFoundError(f"No .cpp files found under {search_dir}")
    return cpp_files


def patch_generated_cpp_if_needed(cpp_path: Path):
    """
    Replace py::scoped_interpreter guard{} with py::gil_scoped_acquire gil;
    and switch headers from embed.h to pybind11/pybind11.h + pybind11/eval.h
    when needed.
    """
    text = cpp_path.read_text(encoding="utf-8")

    changed = False

    if "#include <pybind11/embed.h>" in text:
        text = text.replace(
            "#include <pybind11/embed.h>",
            "#include <pybind11/pybind11.h>\n#include <pybind11/eval.h>"
        )
        changed = True

    if "py::scoped_interpreter guard{};" in text:
        text = text.replace(
            "py::scoped_interpreter guard{};",
            "py::gil_scoped_acquire gil;"
        )
        changed = True

    if changed:
        cpp_path.write_text(text, encoding="utf-8")
        print(f"[PATCH] Updated {cpp_path}")


def _cpp_uses_pybind_embed_style(text: str) -> bool:
    """True if *text* looks like generated code that needs :func:`patch_generated_cpp_if_needed`."""
    return (
        "#include <pybind11/embed.h>" in text
        or "py::scoped_interpreter guard{};" in text
    )


def patch_pybind_wrapper_cpp_files_under(root: Path) -> None:
    """Run :func:`patch_generated_cpp_if_needed` on every ``*.cpp`` under *root* that embeds pybind that way."""
    for cpp_path in sorted(root.rglob("*.cpp")):
        try:
            text = cpp_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _cpp_uses_pybind_embed_style(text):
            patch_generated_cpp_if_needed(cpp_path)


def _op_def_cpp_relpath_from_layout(layout: dict) -> str:
    """Pick the manifest path for ``op_host/<stem>_def.cpp`` (not under ``op_host/src``)."""
    candidates: list[str] = []
    for v in layout.values():
        if not isinstance(v, str) or not v.endswith(".cpp"):
            continue
        parts = Path(v).parts
        if len(parts) >= 2 and parts[0] == "op_host" and parts[1] != "src":
            candidates.append(v)
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one cpp path in layout under op_host/ but not op_host/src/ "
            f"(OpDef TU), got {candidates!r}"
        )
    return candidates[0]


def collect_required_cpp_files(extracted_dir: Path, final_src_dir: Path):
    """
    Resolve cpp paths from ``cpp_layout.json`` (written with cpp_sources zip) and copy into
    *final_src_dir* with the basenames expected by ``CMakeLists.txt`` / ``bindings.cpp``.
    """
    final_src_dir.mkdir(parents=True, exist_ok=True)

    layout, base = pypto.export.load_cpp_layout(extracted_dir)

    copied_paths: list[Path] = []

    def_rel = _op_def_cpp_relpath_from_layout(layout)
    def_src = base / def_rel
    infer_cpp_name = def_src.name
    dst_def = final_src_dir / infer_cpp_name
    shutil.copy2(def_src, dst_def)
    copied_paths.append(dst_def)
    print(f"[COPY] {def_src} -> {dst_def}")

    patch_pybind_wrapper_cpp_files_under(final_src_dir)

    # Canonical export always embeds inferShape in the GE OpDef TU under op_host/.
    use_ge_host_infer = True
    return copied_paths, use_ge_host_infer, infer_cpp_name


def generate_bindings_cpp(dst: Path, module_name: str, *, infer_from_ge_host: bool):
    if infer_from_ge_host:
        infer_decl = """std::tuple<int64_t, int64_t> inferShape(std::tuple<int64_t, int64_t> x0_shape,
                                 std::tuple<int64_t, int64_t> x1_shape);"""
        infer_def = '    m.def("infer_shape", &inferShape, "Infer output shape");'
    else:
        infer_decl = """std::tuple<int64_t, int64_t> infer_shape(std::tuple<int64_t, int64_t> x0_shape,
                                 std::tuple<int64_t, int64_t> x1_shape);"""
        infer_def = '    m.def("infer_shape", &infer_shape, "Infer output shape");'

    content = f'''#include <cstdint>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations
{infer_decl}

PYBIND11_MODULE({module_name}, m) {{
    m.doc() = "Generated PyPTO operator runtime module";

{infer_def}
}}
'''
    dst.write_text(content, encoding="utf-8")
    print(f"[GEN] {dst}")


def generate_cmakelists(dst: Path, module_name: str, *, infer_cpp_name: str):
    infer_src = f"    src/{infer_cpp_name}\n"
    content = f'''cmake_minimum_required(VERSION 3.15)
project({module_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module({module_name}
    src/bindings.cpp
{infer_src})
'''
    dst.write_text(content, encoding="utf-8")
    print(f"[GEN] {dst}")


def build_shared_object(project_dir: Path, module_name: str):
    build_dir = project_dir / "build"

    pybind11_cmakedir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        text=True,
    ).strip()

    run_cmd([
        "cmake",
        "-S", str(project_dir),
        "-B", str(build_dir),
        f"-Dpybind11_DIR={pybind11_cmakedir}",
    ])

    run_cmd([
        "cmake",
        "--build", str(build_dir),
        "-j",
    ])

    so_candidates = list(build_dir.glob(f"{module_name}*.so"))
    if not so_candidates:
        raise FileNotFoundError(f"Build finished but no .so found in {build_dir}")

    print("\n[OK] Built shared object(s):")
    for so in so_candidates:
        print(f"  {so}")

    return so_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .onnx or .air model")
    parser.add_argument(
        "--module-name",
        type=str,
        default="op_kernel_lib",
        help="Python module name / output extension name",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output project directory; default is next to input model",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before regenerating",
    )
    args = parser.parse_args()

    model_path = Path(args.path).resolve()
    module_name = args.module_name

    if args.out_dir is None:
        project_dir = model_path.parent / f"build_{module_name}"
    else:
        project_dir = Path(args.out_dir).resolve()

    if args.clean and project_dir.exists():
        print(f"[CLEAN] Removing {project_dir}")
        shutil.rmtree(project_dir)

    src_dir = project_dir / "src"
    extracted_dir = project_dir / "src_extracted"

    project_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    model_format, node = load_model_and_extract_node(model_path)

    print("\n----------------")
    kernel_format = pypto.export.extract_kernel_format(node)
    pypto_meta = pypto.export.extract_pypto_meta(node)
    print(f"kernel_format: {kernel_format}")
    print(f"pypto_meta:    {pypto_meta}")
    print("----------------\n")

    extract_cpp_sources(node, project_dir)
    _, infer_from_ge_host, infer_cpp_name = collect_required_cpp_files(extracted_dir, src_dir)

    generate_bindings_cpp(
        src_dir / "bindings.cpp", module_name, infer_from_ge_host=infer_from_ge_host
    )
    generate_cmakelists(
        project_dir / "CMakeLists.txt", module_name, infer_cpp_name=infer_cpp_name
    )

    so_files = build_shared_object(project_dir, module_name)

    print("\n[DONE]")
    print(f"Project directory: {project_dir}")
    print("Built module(s):")
    for so in so_files:
        print(f"  {so}")

    print("\nTest with:")
    print(f'python3 - <<\'PY\'')
    print("import sys")
    print(f"sys.path.append(r'{project_dir / 'build'}')")
    print(f"import {module_name}")
    print(f"print('ok')")
    print("PY")


if __name__ == "__main__":
    main()