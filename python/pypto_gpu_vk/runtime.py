from __future__ import annotations

import torch

import pypto  # noqa: F401
from pypto import pypto_impl

from .config import RuntimeConfig


def _normalize_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got {type(tensor).__name__}.")
    return tensor.contiguous() if not tensor.is_contiguous() else tensor


class VkRuntime:
    def __init__(
        self,
        enable_validation: bool = False,
        enable_cache: bool = True,
        fallback_to_torch: bool = True,
        allow_cpu_fallback_for_dev: bool | None = None,
    ):
        self.config = RuntimeConfig(
            enable_validation=enable_validation,
            enable_cache=enable_cache,
            fallback_to_torch=fallback_to_torch,
            allow_cpu_fallback_for_dev=fallback_to_torch if allow_cpu_fallback_for_dev is None else allow_cpu_fallback_for_dev,
        )
        self._launcher = pypto_impl.GpuVkLauncher()
        self._runner = pypto_impl.GpuVkRunner()
        self._status = self._launcher.initialize(enable_validation)
        self._runner_status = self._runner.initialize(enable_validation)
        self._last_execute_status = None
        if self._status == pypto_impl.GpuVkStatus.SUCCESS and self._runner_status != pypto_impl.GpuVkStatus.SUCCESS:
            self._status = self._runner_status
        self._runner.set_enable_cache(enable_cache)

    @property
    def status(self):
        return self._status

    def available(self) -> bool:
        return self._launcher.available() and self._runner.available()

    def pipeline_cache_hit_count(self) -> int:
        return self._runner.pipeline_cache_hit_count()

    def pipeline_cache_entry_count(self) -> int:
        return self._runner.pipeline_cache_entry_count()

    @property
    def last_execute_status(self):
        return self._last_execute_status

    def run_elementwise_binary(self, op_name: str, input0, input1):
        input0 = _normalize_tensor(input0)
        input1 = _normalize_tensor(input1)
        status = self._launcher.run_elementwise_binary(op_name, input0, input1)
        if status == pypto_impl.GpuVkStatus.SUCCESS and self.available():
            return None
        return self._cpu_fallback(op_name, input0, input1)

    def run_elementwise_unary(self, op_name: str, input0):
        input0 = _normalize_tensor(input0)
        status = self._launcher.run_elementwise_unary(op_name, input0)
        if status == pypto_impl.GpuVkStatus.SUCCESS and self.available():
            return None
        return self._cpu_fallback(op_name, input0)

    def execute(self, compiled_op, *inputs):
        tensors = [_normalize_tensor(tensor) for tensor in inputs]
        if not compiled_op.supports_real_vulkan:
            self._last_execute_status = pypto_impl.GpuVkStatus.NOT_SUPPORTED
            if not self.config.fallback_to_torch and not self.config.allow_cpu_fallback_for_dev:
                raise RuntimeError(f"Vulkan execution is not supported for op {compiled_op.op_name}.")
            return self._cpu_fallback(compiled_op.op_name, *tensors)

        status, output = self._runner.run_artifact(compiled_op.artifact, tensors)
        self._last_execute_status = status
        if status == pypto_impl.GpuVkStatus.SUCCESS and output is not None:
            return output
        if not self.config.fallback_to_torch and not self.config.allow_cpu_fallback_for_dev:
            raise RuntimeError(f"Vulkan execution failed with status {status}.")
        return self._cpu_fallback(compiled_op.op_name, *tensors)

    def synchronize(self) -> None:
        return None

    def destroy(self) -> None:
        self._runner.destroy()
        self._launcher.destroy()

    def _cpu_fallback(self, op_name: str, *inputs):
        if not self.config.fallback_to_torch and not self.config.allow_cpu_fallback_for_dev:
            raise RuntimeError("Vulkan execution is unavailable and fallback_to_torch is disabled.")
        if op_name == "add":
            return inputs[0] + inputs[1]
        if op_name == "mul":
            return inputs[0] * inputs[1]
        if op_name == "relu":
            return torch.relu(inputs[0])
        if op_name == "where":
            return torch.where(inputs[0].bool(), inputs[1], inputs[2])
        raise NotImplementedError(f"Unsupported op for torch fallback: {op_name}")
