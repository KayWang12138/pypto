import torch

import pypto  # noqa: F401
from pypto import pypto_impl


_INTEROP = pypto_impl.GpuVkTorchInterop()


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got {type(tensor).__name__}.")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def torch_to_vk_cpu_staging(tensor: torch.Tensor):
    return _INTEROP.import_from_torch_cpu(_normalize_tensor(tensor))


def from_torch(tensor: torch.Tensor, storage_mode: str = "cpu_staging"):
    tensor = _normalize_tensor(tensor)
    if storage_mode == "cpu_staging":
        return _INTEROP.import_from_torch_cpu(tensor)
    if storage_mode == "external_memory":
        return _INTEROP.try_import_external_memory(tensor)
    if storage_mode == "tensor":
        return tensor
    raise ValueError(f"Unsupported storage_mode: {storage_mode}")


def to_torch(vk_tensor):
    return _INTEROP.export_to_torch_cpu(vk_tensor)


def alloc_output_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(_normalize_tensor(tensor))
