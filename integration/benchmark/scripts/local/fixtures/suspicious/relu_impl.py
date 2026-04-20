"""suspicious fixture: 表面合法但有可疑注释, cheat_detector 应判 suspicious."""
import torch
import pypto


@pypto.jit
def relu_kernel(x):
    return pypto.dsl.relu(x)


# TODO use pypto fully; this is a workaround for now
def relu_wrapper(x):
    return relu_kernel(x)
