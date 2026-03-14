from __future__ import annotations

from dataclasses import dataclass

import torch

from pypto import pypto_impl

from .backend import GpuVkBackend
from .config import CompileConfig


def _analyze_matmul_inputs(lhs: torch.Tensor, rhs: torch.Tensor):
    if lhs.ndim not in {1, 2, 3} or rhs.ndim not in {1, 2, 3}:
        return None
    lhs_batch = lhs.shape[0] if lhs.ndim == 3 else 1
    rhs_batch = rhs.shape[0] if rhs.ndim == 3 else 1
    rhs_k = rhs.shape[0] if rhs.ndim == 1 else rhs.shape[-2]
    if lhs.shape[-1] != rhs_k:
        return None
    if lhs_batch != rhs_batch and lhs_batch != 1 and rhs_batch != 1:
        return None
    return lhs_batch, rhs_batch


def _supports_real_vulkan(op_name: str, inputs: tuple[torch.Tensor, ...], artifact) -> bool:
    if not bool(artifact.is_real_spirv):
        return False
    if op_name == "matmul":
        if len(inputs) != 2:
            return False
        lhs, rhs = inputs
        analyzed = _analyze_matmul_inputs(lhs, rhs)
        return (
            lhs.device.type == "cpu"
            and rhs.device.type == "cpu"
            and lhs.dtype == torch.float32
            and rhs.dtype == torch.float32
            and analyzed is not None
        )
    if op_name not in {"add", "mul", "relu", "where"}:
        return False
    return all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in inputs)


def _normalize_options(options: dict | CompileConfig | None) -> CompileConfig:
    if options is None:
        return CompileConfig()
    if isinstance(options, CompileConfig):
        return CompileConfig(
            optimize=options.optimize,
            debug_info=options.debug_info,
            glslang_validator_path=options.glslang_validator_path,
            dump_artifacts=options.dump_artifacts,
            dump_dir=options.dump_dir,
        )
    if isinstance(options, dict):
        return CompileConfig(
            optimize=options.get("optimize", True),
            debug_info=options.get("debug_info", False),
            glslang_validator_path=options.get("glslang_validator_path"),
            dump_artifacts=options.get("dump_artifacts", False),
            dump_dir=options.get("dump_dir"),
        )
    raise TypeError(f"Unsupported options type: {type(options).__name__}.")


def build_tensor_graph(op_name: str, *inputs: torch.Tensor):
    return GpuVkBackend().build_tensor_graph(op_name, *inputs)


def lower_to_dispatch(graph):
    return pypto_impl.GpuVkLowerToDispatch(graph)


def lower_to_shader(graph):
    return pypto_impl.GpuVkLowerToShader(graph)


def compile_to_spirv(shader_ir, dispatch_graph, dtype: int = 0, options: dict | CompileConfig | None = None):
    config = _normalize_options(options)
    meta = pypto_impl.GpuVkBuildShaderMeta(dispatch_graph, dtype)
    glsl = pypto_impl.GpuVkEmitGlsl(shader_ir, meta)
    spirv = pypto_impl.GpuVkCompileSpirv(
        glsl,
        config.optimize,
        config.debug_info,
        config.glslang_validator_path or "",
        config.dump_artifacts,
        config.dump_dir or "",
    )
    return meta, glsl, spirv


@dataclass(slots=True)
class CompiledVkOp:
    op_name: str
    graph: object
    dispatch_graph: object
    shader_ir: object
    artifact: object
    options: CompileConfig
    supports_real_vulkan: bool = False

    def __call__(self, *inputs, runtime=None):
        from .runtime import VkRuntime

        runtime = runtime or VkRuntime()
        return runtime.execute(self, *inputs)

    def dump_glsl(self) -> str:
        return self.artifact.glsl

    def dump_spirv(self) -> bytes:
        return b"".join(int(word).to_bytes(4, "little", signed=False) for word in self.artifact.spirv)


def compile_op(op_name: str, *inputs: torch.Tensor, options: dict | CompileConfig | None = None):
    config = _normalize_options(options)
    backend = GpuVkBackend(config)
    graph = backend.build_tensor_graph(op_name, *inputs)
    dispatch_graph = backend.lower_to_dispatch(graph)
    shader_ir = backend.lower_to_shader(graph)
    artifact = backend.compile_artifact(graph)
    return CompiledVkOp(
        op_name,
        graph,
        dispatch_graph,
        shader_ir,
        artifact,
        config,
        _supports_real_vulkan(op_name, inputs, artifact),
    )
