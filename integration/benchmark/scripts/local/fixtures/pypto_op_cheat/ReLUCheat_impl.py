"""ReLUCheat: 故意拆成两个 jit kernel + 第二个 kernel 是 mock no-op + 0.0.

只是供 ``scripts/local/test-skill-cheat.sh`` 端到端验证 ``pypto-kernel-validator``
能识别多 kernel 拆分 (S5) + mock kernel (S6) + 自承注释 (S9).

不要复用到生产 — pypto 算子要求一个 forward 一个融合 kernel.
"""
import pypto
import torch


@pypto.frontend.jit
def relu_kernel_part_a(
    input_tensor: pypto.Tensor(),
    output_tensor: pypto.Tensor(),
):
    """第一个 jit kernel: relu 实质计算."""
    pypto.set_vec_tile_shapes(output_tensor.shape, input_tensor.shape)
    output_tensor[:] = pypto.relu(input_tensor)


@pypto.frontend.jit
def relu_kernel_part_b(
    input_tensor: pypto.Tensor(),
    output_tensor: pypto.Tensor(),
):
    """第二个 jit kernel: 加 0, 啥也不干. 故意拆成双 kernel 来测反作弊."""
    pypto.set_vec_tile_shapes(output_tensor.shape, input_tensor.shape)
    output_tensor[:] = input_tensor + 0.0


def ReLUCheat_wrapper(x):
    """Wrapper: 顺序调两个 jit kernel = 多 kernel 拆分 = 作弊形态 S5."""
    mid = torch.empty_like(x)
    relu_kernel_part_a(x, mid)
    out = torch.empty_like(mid)
    relu_kernel_part_b(mid, out)
    return out
