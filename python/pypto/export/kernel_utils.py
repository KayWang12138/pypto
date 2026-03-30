import json
import os
from pathlib import Path

__all__: tuple[str, ...] = ()

_ROOT_KERNEL_BINARIES_DIR = "output"
_ROOT_KERNEL_IR_DIR = "build_output"  # TODO update after ir converter is finalized


def _json_has_rawname_for_kernel(obj, kernel_name: str) -> bool:
    """True if any ``rawname`` field in a nested JSON structure matches ``TENSOR_{kernel_name}``."""
    target = f"TENSOR_{kernel_name}"
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rawname" and isinstance(v, str) and v == target:
                return True
            if _json_has_rawname_for_kernel(v, kernel_name):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if _json_has_rawname_for_kernel(v, kernel_name):
                return True
    return False


def _find_kernel_binary_path(kernel_name):
    """Return the newest compiled kernel directory path for the given kernel name.

    A kernel directory is considered a match if its ``program.json`` contains
    at least one ``rawname == TENSOR_{kernel_name}`` entry anywhere in the JSON.
    """
    root = Path(_ROOT_KERNEL_BINARIES_DIR)
    if not root.is_dir():
        raise OSError(
            f"Kernel binaries root directory '{_ROOT_KERNEL_BINARIES_DIR}' "
            "does not exist or is not a directory"
        )

    kernels = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)

    for kernel_dir in kernels:
        program_json = kernel_dir / "program.json"
        if not program_json.is_file():
            continue
        try:
            with program_json.open("r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception:
            # Skip malformed JSON but keep scanning other candidates.
            continue
        if _json_has_rawname_for_kernel(json_data, kernel_name):
            return str(kernel_dir)

    raise OSError(
        f"No binaries were found for kernel {kernel_name!r} "
        f"under '{_ROOT_KERNEL_BINARIES_DIR}'"
    )


def _find_kernel_pto_path(kernel_name):
    """Return the newest ``output.pto`` path for the given kernel name.

    The IR directory is chosen as the latest subdirectory of ``build_output`` whose
    name starts with ``kernel_name`` and that contains an ``output.pto`` file.
    """
    root = Path(_ROOT_KERNEL_IR_DIR)
    if not root.is_dir():
        raise OSError(
            f"Kernel IR root directory '{_ROOT_KERNEL_IR_DIR}' "
            "does not exist or is not a directory"
        )

    kernels = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)

    for kernel_dir in kernels:
        if not kernel_dir.name.startswith(kernel_name):
            continue
        pto_path = kernel_dir / "output.pto"
        if not pto_path.is_file():
            raise OSError(f"No IR was found in '{kernel_dir}' (expected 'output.pto')")
        return str(pto_path)

    raise OSError(
        f"No IRs were found for kernel {kernel_name!r} "
        f"under '{_ROOT_KERNEL_IR_DIR}'"
    )
