# Sample infer_shape functions for codegen tests (must live in a real module for inspect.getsource).


def infer_shape_two_by_two(
    x0_shape: tuple[int, int],
    x1_shape: tuple[int, int],
) -> tuple[int, int]:
    return x0_shape


def infer_shape_4d_broadcast(
    a_shape: tuple[int, int, int, int],
    b_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return a_shape


def infer_shape_sum_last(
    x0_shape: tuple[int, int, int],
    x1_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        x0_shape[0],
        x0_shape[1],
        x0_shape[2] + x1_shape[2],
    )
