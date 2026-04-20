"""clean fixture: 合法的 PyPTO ReLU 形态, cheat_detector 期望 verdict=pass."""
import torch
import pypto


@pypto.jit
def relu_kernel(x):
    return pypto.dsl.maximum(x, 0)


def relu_wrapper(x):
    return relu_kernel(x)
