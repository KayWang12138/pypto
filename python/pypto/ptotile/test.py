from .runtime import *
from .compiler import *


@kernel
def TestKernelFunction():
    print("This is a test kernel function.")
    return

@jit
def TestJITFunction():
    print("This is a test JIT function.")
    TestKernelFunction().launch()
    return


if __name__ == "__main__":
    TestJITFunction()