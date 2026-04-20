"""multi_jit fixture: 2 个 @pypto.jit, cheat_detector 应判 cheat (jit_def_count)."""
import torch
import pypto


@pypto.jit
def relu_a(x):
    return pypto.dsl.relu(x)


@pypto.jit
def relu_b(x):
    return pypto.dsl.add(x, 0.0)


def relu_wrapper(x):
    return relu_b(relu_a(x))
