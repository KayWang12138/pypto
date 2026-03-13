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


def collect_required_cpp_files(extracted_dir: Path, final_src_dir: Path):
    """
    Find extracted cpp files and copy them into final_src_dir with expected names.
    Assumes files are already named:
      - infer_shape.cpp
      - calc_workspace.cpp
      - op_kernel_info.cpp
      - op_compile.cpp
      - op_execute.cpp
    """
    final_src_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "infer_shape.cpp": None,
        "calc_workspace.cpp": None,
        "op_kernel_info.cpp": None,
        "op_compile.cpp": None,
        "op_execute.cpp": None,
    }

    for cpp in extracted_dir.rglob("*.cpp"):
        name = cpp.name
        if name in required and required[name] is None:
            required[name] = cpp

    missing = [name for name, path in required.items() if path is None]
    if missing:
        available = [str(p) for p in extracted_dir.rglob("*.cpp")]
        raise FileNotFoundError(
            f"Missing required cpp files: {missing}\n"
            f"Available cpp files:\n" + "\n".join(available)
        )

    copied_paths = []
    for name, src_path in required.items():
        dst_path = final_src_dir / name
        shutil.copy2(src_path, dst_path)
        patch_generated_cpp_if_needed(dst_path)
        copied_paths.append(dst_path)
        print(f"[COPY] {src_path} -> {dst_path}")

    return copied_paths


def generate_bindings_cpp(dst: Path, module_name: str):
    content = f'''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations
std::tuple<int, int> infer_shape(std::tuple<int, int> x0_shape,
                                 std::tuple<int, int> x1_shape);

int calc_workspace(std::tuple<int, int> x0_shape,
                   std::tuple<int, int> x1_shape);

std::tuple<int, int> op_kernel_info(int inputs_num, int outputs_num);
int op_compile(int flags);
int op_execute(int flags);

PYBIND11_MODULE({module_name}, m) {{
    m.doc() = "Generated PyPTO operator runtime module";

    m.def("infer_shape", &infer_shape, "Infer output shape");
    m.def("calc_workspace", &calc_workspace, "Calculate workspace size");
    m.def("op_kernel_info", &op_kernel_info, "Get kernel info");
    m.def("op_compile", &op_compile, "Compile op");
    m.def("op_execute", &op_execute, "Execute op");
}}
'''
    dst.write_text(content, encoding="utf-8")
    print(f"[GEN] {dst}")


def generate_cmakelists(dst: Path, module_name: str):
    content = f'''cmake_minimum_required(VERSION 3.15)
project({module_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module({module_name}
    src/bindings.cpp
    src/infer_shape.cpp
    src/calc_workspace.cpp
    src/op_kernel_info.cpp
    src/op_compile.cpp
    src/op_execute.cpp
)
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
    collect_required_cpp_files(extracted_dir, src_dir)

    generate_bindings_cpp(src_dir / "bindings.cpp", module_name)
    generate_cmakelists(project_dir / "CMakeLists.txt", module_name)

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