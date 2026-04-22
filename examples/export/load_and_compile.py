#!/usr/bin/env python3
"""Load a pypto-exported ONNX/AIR model, extract generated cpp sources, and
build a shared object linked against the local Ascend package.

Most of the heavy lifting (patching the generated cpp, emitting CMakeLists,
collecting the canonical TU set, Ascend-home autodetection) lives in
``pypto.export.cpp`` so other tools can reuse it without shelling out to
this script. This example's own job is the CLI + the one non-library step:
driving ``cmake`` to produce the .so.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pypto


def run_cmd(cmd, cwd=None, env=None):
    print(f"\n[RUN] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


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
        "--op-type",
        type=str,
        required=True,
        help="Custom op type to extract (e.g. AddPyptoCustomOp)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=pypto.export.DEFAULT_ONNX_DOMAIN,
        help=f"ONNX custom-op domain (default: {pypto.export.DEFAULT_ONNX_DOMAIN})",
    )
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
        "--ascend-home",
        type=str,
        default=None,
        help="Override Ascend install arch-subdir (with include/ and lib64/). "
             "Defaults to $ASCEND_HOME_PATH/<arch>.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before regenerating",
    )
    args = parser.parse_args()

    model_path = Path(args.path).resolve()
    module_name = args.module_name

    ascend_home = (
        Path(args.ascend_home).resolve()
        if args.ascend_home
        else pypto.export.cpp.default_ascend_home()
    )
    if not (ascend_home / "include").is_dir():
        raise FileNotFoundError(
            f"Expected Ascend include dir under {ascend_home}/include — "
            "pass --ascend-home pointing at the arch-specific subtree "
            "(e.g. $ASCEND_HOME_PATH/aarch64-linux)."
        )
    print(f"[INFO] Ascend home: {ascend_home}")

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
    extracted_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from: {model_path}")
    model, model_format = pypto.export.load_model(model_path)
    node = pypto.export.extract_node(model, op_type=args.op_type, domain=args.domain)

    print("\n----------------")
    kernel_format = pypto.export.extract_kernel_format(node)
    pypto_meta = pypto.export.extract_pypto_meta(node)
    print(f"model_format:  {model_format}")
    print(f"kernel_format: {kernel_format}")
    print(f"pypto_meta:    {pypto_meta}")
    print("----------------\n")

    cpp_sources_zip_meta = pypto.export.extract_cpp_sources(
        node=node,
        out_dir=str(extracted_dir),
        strict=True,
    )
    print(f"[INFO] cpp_sources_zip_meta = {cpp_sources_zip_meta}")

    cpp_paths = pypto.export.cpp.collect_required_cpp_files(extracted_dir, src_dir)

    pypto.export.cpp.generate_bindings_cpp(
        src_dir / "bindings.cpp",
        module_name,
    )
    pypto.export.cpp.generate_cmakelists(
        project_dir / "CMakeLists.txt",
        module_name,
        cpp_source_names=[p.name for p in cpp_paths.values()],
        ascend_home=ascend_home,
    )

    so_files = build_shared_object(project_dir, module_name)

    print("\n[DONE]")
    print(f"Project directory: {project_dir}")
    print("Built module(s):")
    for so in so_files:
        print(f"  {so}")


if __name__ == "__main__":
    main()
