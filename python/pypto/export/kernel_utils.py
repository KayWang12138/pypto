import os

__all__: tuple[str, ...] = ()

_ROOT_KERNEL_BINARIES_DIR = "output"
_ROOT_KERNEL_IR_DIR = "build_output"  # TODO update after ir converter is finalized


def _find_kernel_binary_path(kernel_name):
    """Locate the most recent kernel binary directory for the given kernel name."""
    kernels = os.listdir(_ROOT_KERNEL_BINARIES_DIR)
    kernels.sort(reverse=True)

    for kernel_subdir in kernels:
        kernel_path = os.path.join(_ROOT_KERNEL_BINARIES_DIR, kernel_subdir)
        for root, dirs, files in os.walk(kernel_path):
            if not "kernel" in root:
                dirs[:] = [d for d in dirs if d.startswith("kernel")]
            for file in files:
                if kernel_name in file:
                    return kernel_path

    raise OSError(f"No binaries were found for kernel {kernel_name}")  # OSError or ValueError ?


def _find_kernel_pto_path(kernel_name):
    """Locate the most recent output.pto file for the given kernel name."""
    kernels = os.listdir(_ROOT_KERNEL_IR_DIR)
    kernels.sort(reverse=True)

    for kernel_subdir in kernels:
        if kernel_subdir.startswith(kernel_name):
            pto_path = os.path.join(_ROOT_KERNEL_IR_DIR, kernel_subdir, "output.pto")
            if not os.path.exists(pto_path):
                raise OSError(f"No IR was found in {kernel_subdir} directory")
            return pto_path

    raise OSError(f"No IRs were found for kernel {kernel_name}")
