import os

ROOT_KERNEL_BINARIES_DIR = "output"
ROOT_KERNEL_IR_DIR = "/home/p84341448/n84434144_nikita/pypto_ir/build_output/" # TODO update after pypto ir repo is merged to cann/pypto

def _find_kernel_binary_path(kernel_name):
    kernels = os.listdir(ROOT_KERNEL_BINARIES_DIR)
    kernels.sort(reverse=True)

    for kernel_subdir in kernels:
        kernel_path = os.path.join(ROOT_KERNEL_BINARIES_DIR, kernel_subdir)
        for root, dirs, files in os.walk(kernel_path):
            if not "kernel" in root:
                dirs[:] = [d for d in dirs if d.startswith("kernel")]
            for file in files:
                if kernel_name in file:
                    return kernel_path

    raise OSError(f"No binaries were found for kernel {kernel_name}") # OSError or ValueError ?

def _find_kernel_pto_path(kernel_name):
    kernels = os.listdir(ROOT_KERNEL_IR_DIR)
    kernels.sort(reverse=True)

    for kernel_subdir in kernels:
        if kernel_subdir.startswith(kernel_name):
            pto_path = os.path.join(ROOT_KERNEL_IR_DIR, kernel_subdir, "output.pto")
            if not os.path.exists(pto_path):
                raise OSError(f"No IR was found in {kernel_subdir} directory")
            return pto_path

    raise OSError(f"No IRs were found for kernel {kernel_name}")
