#!/usr/bin/env python3
# coding: utf-8

"""Pure-Python regression tests for frontend.jit artifact path isolation."""

import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[6]
JIT_PATH = REPO_ROOT / "python/pypto_block" / "frontend" / "jit.py"

_STUB_MODULE_NAMES = (
    "torch",
    "pypto_block",
    "pypto_block.backend",
    "pypto_block.frontend",
    "pypto_block.frontend.kernel",
    "pypto_block.pypto_core",
    "pypto_block.pypto_core.codegen",
    "pypto_block.pypto_core.ir",
    "pypto_block.pypto_core.passes",
)


class _DummyProgram:
    def __init__(self, name: str, ir_text: str):
        self.name = name
        self._ir_text = ir_text

    def __str__(self) -> str:
        return self._ir_text


def _install_pass_context_stub():
    pypto_block_mod = sys.modules.get("pypto_block")
    if pypto_block_mod is None:
        pypto_block_mod = types.ModuleType("pypto_block")
        sys.modules["pypto_block"] = pypto_block_mod

    pypto_core_mod = sys.modules.get("pypto_block.pypto_core")
    if pypto_core_mod is None:
        pypto_core_mod = types.ModuleType("pypto_block.pypto_core")
        sys.modules["pypto_block.pypto_core"] = pypto_core_mod

    passes_mod = types.ModuleType("pypto_block.pypto_core.passes")

    class VerificationMode:
        BEFORE_AND_AFTER = "before_and_after"

    class VerificationInstrument:
        def __init__(self, *_args, **_kwargs):
            pass

    class PassContext:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    passes_mod.PassContext = PassContext
    passes_mod.VerificationInstrument = VerificationInstrument
    passes_mod.VerificationMode = VerificationMode

    pypto_core_mod.passes = passes_mod
    pypto_block_mod.pypto_core = pypto_core_mod
    sys.modules["pypto_block.pypto_core.passes"] = passes_mod


_install_pass_context_stub()


def _install_stub_modules():
    saved = {name: sys.modules.get(name) for name in _STUB_MODULE_NAMES}

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = type("Tensor", (), {})
    for attr in (
        "float16",
        "bfloat16",
        "float32",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint64",
        "bool",
    ):
        setattr(torch_mod, attr, object())
    torch_mod.npu = types.SimpleNamespace(
        current_stream=lambda: types.SimpleNamespace(_as_parameter_=None)
    )

    data_type = types.SimpleNamespace(
        FP16="fp16",
        BF16="bf16",
        FP32="fp32",
        INT8="int8",
        INT16="int16",
        INT32="int32",
        INT64="int64",
        UINT8="uint8",
        UINT16="uint16",
        UINT32="uint32",
        UINT64="uint64",
        BOOL="bool",
        INDEX="index",
    )

    backend_api = types.SimpleNamespace(
        reset_for_testing=lambda: None,
        set_backend_type=lambda *_args, **_kwargs: None,
    )

    pypto_block_mod = types.ModuleType("pypto_block")
    pypto_block_mod.DataType = data_type
    pypto_block_mod.backend = backend_api

    backend_mod = types.ModuleType("pypto_block.backend")

    class BackendType:
        CCE = "cce"
        PTO = "pto"

    backend_mod.BackendType = BackendType

    pypto_core_mod = types.ModuleType("pypto_block.pypto_core")
    codegen_mod = types.ModuleType("pypto_block.pypto_core.codegen")

    class DummyCCECodegen:
        def generate_single(self, prog, arch):
            return "__global__ AICORE void fake_kernel(__gm__ float* x) {}"

    class DummyPTOCodegen:
        pass

    codegen_mod.CCECodegen = DummyCCECodegen
    codegen_mod.PTOCodegen = DummyPTOCodegen

    ir_mod = types.ModuleType("pypto_block.pypto_core.ir")
    for name in ("ConstInt", "PtrType", "ScalarType", "TensorType", "Var"):
        setattr(ir_mod, name, type(name, (), {}))

    frontend_mod = types.ModuleType("pypto_block.frontend")
    kernel_mod = types.ModuleType("pypto_block.frontend.kernel")

    class KernelDef:
        pass

    kernel_mod.KernelDef = KernelDef
    frontend_mod.kernel = kernel_mod

    pypto_core_mod.codegen = codegen_mod
    pypto_core_mod.ir = ir_mod

    sys.modules["torch"] = torch_mod
    sys.modules["pypto_block"] = pypto_block_mod
    sys.modules["pypto_block.backend"] = backend_mod
    sys.modules["pypto_block.frontend"] = frontend_mod
    sys.modules["pypto_block.frontend.kernel"] = kernel_mod
    sys.modules["pypto_block.pypto_core"] = pypto_core_mod
    sys.modules["pypto_block.pypto_core.codegen"] = codegen_mod
    sys.modules["pypto_block.pypto_core.ir"] = ir_mod

    return saved


def _redirect_build_root(monkeypatch, jit_module):
    artifact_root = REPO_ROOT / "build" / f"jit-artifact-paths-{uuid.uuid4().hex}"
    real_join = jit_module.os.path.join

    def join_wrapper(*parts):
        if len(parts) >= 2 and parts[0] == "." and parts[1] == "build":
            return real_join(os.fspath(artifact_root), *parts[2:])
        return real_join(*parts)

    monkeypatch.setattr(jit_module.os.path, "join", join_wrapper)
    return artifact_root


@pytest.fixture()
def jit_module(monkeypatch):
    saved = _install_stub_modules()
    module_name = "_jit_artifact_path_test"
    sys.modules.pop(module_name, None)
    saved_env = {name: os.environ.get(name) for name in ("PYPTO_JIT_ARCH", "npu_arch")}

    try:
        spec = importlib.util.spec_from_file_location(module_name, JIT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        monkeypatch.setattr(module, "_extract_param_specs", lambda _prog: [])
        monkeypatch.setattr(module.Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
        monkeypatch.setattr(
            module.Path,
            "write_text",
            lambda self, content, encoding="utf-8": len(content),
        )
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/fake/ascend/toolkit")
        monkeypatch.setenv("ASCEND_HOME_PATH", "/fake/ascend/home")
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda *args, **kwargs: types.SimpleNamespace(
                returncode=0, stdout=b"", stderr=b""
            ),
        )
        yield module
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_compile_artifact_path_uses_test_kernel_arch_and_mode(jit_module, monkeypatch):
    """Artifact paths are scoped by test name, kernel name, arch, and codegen mode."""
    _redirect_build_root(monkeypatch, jit_module)

    first = _DummyProgram("matmul_add_matmul_add", "program { store(workspace) }")
    second = _DummyProgram("matmul_add_matmul_add", "program { move(mm1_res) }")

    first_lib = jit_module.compile(first, arch="a5", codegen_mode="cce").lib_path
    second_lib = jit_module.compile(second, arch="a5", codegen_mode="cce").lib_path

    assert first_lib == second_lib
    assert Path(first_lib).parent.name == "jit_artifact_paths__matmul_add_matmul_add__a5__cce"


def test_compile_artifacts_are_scoped_by_arch(jit_module, monkeypatch):
    """The same program compiled for different arches must not reuse one artifact path."""
    _redirect_build_root(monkeypatch, jit_module)

    prog = _DummyProgram("matmul_add_matmul_add", "program { body }")

    a5_lib = jit_module.compile(prog, arch="a5", codegen_mode="cce").lib_path
    a3_lib = jit_module.compile(prog, arch="a3", codegen_mode="cce").lib_path

    assert a5_lib != a3_lib


def test_compile_artifact_path_stays_stable_for_same_program(jit_module, monkeypatch):
    """Repeated compiles of the same test/kernel/arch/mode use one directory."""
    _redirect_build_root(monkeypatch, jit_module)

    prog = _DummyProgram("matmul_add_matmul_add", "program { body }")

    first_lib = jit_module.compile(prog, arch="a5", codegen_mode="cce").lib_path
    second_lib = jit_module.compile(prog, arch="a5", codegen_mode="cce").lib_path

    assert first_lib == second_lib
