# pylint: disable=missing-docstring
import pypto

N, M = 1023, 1654
VIEW_SHAPE = (16, 16)
TILE_SHAPE = (8, 8)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@pypto.frontend.jit()
def vector_operation_rsqrt(
    a: pypto.Tensor((N, M), pypto.DT_FP32),
) -> pypto.Tensor((N, M), pypto.DT_FP32):
    b = pypto.tensor((N, M), pypto.DT_FP32)
    for b_idx in range(ceil_div(N, TILE_SHAPE[0])):
        for s_idx in range(ceil_div(M, TILE_SHAPE[1])):
            tile_a = pypto.view(
                a,
                VIEW_SHAPE,
                [b_idx * TILE_SHAPE[0], s_idx * TILE_SHAPE[1]],
                [N - b_idx * TILE_SHAPE[0], M - s_idx * TILE_SHAPE[1]],
            )
            pypto.set_vec_tile_shapes(TILE_SHAPE[0], TILE_SHAPE[1])
            tile_a = pypto.rsqrt(tile_a)
            pypto.assemble(tile_a, [b_idx * TILE_SHAPE[0], s_idx * TILE_SHAPE[1]], b)
    return b


print(pypto.dump())
