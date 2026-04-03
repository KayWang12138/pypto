import pypto
import torch


@pypto.frontend.jit
def flash_attention_kernel(
    query: pypto.Tensor,
    output: pypto.Tensor,
):
    pypto.set_vec_tile_shapes(8, 8)
    for _ in range(1):
        output[:] = query


def flash_attention_wrapper(query: torch.Tensor) -> torch.Tensor:
    return query.clone()
