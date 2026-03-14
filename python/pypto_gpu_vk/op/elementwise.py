from pypto_gpu_vk.compile import compile_op
from pypto_gpu_vk.runtime import VkRuntime


def add(input0, input1, runtime: VkRuntime | None = None):
    runtime = runtime or VkRuntime()
    return runtime.execute(compile_op("add", input0, input1), input0, input1)


def mul(input0, input1, runtime: VkRuntime | None = None):
    runtime = runtime or VkRuntime()
    return runtime.execute(compile_op("mul", input0, input1), input0, input1)


def relu(input0, runtime: VkRuntime | None = None):
    runtime = runtime or VkRuntime()
    return runtime.execute(compile_op("relu", input0), input0)
