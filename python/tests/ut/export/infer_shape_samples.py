# Sample infer_shape / infer_dtype functions for codegen tests (must live in a real module for inspect.getsource).

import torch


def infer_shape_one_2d(x0_shape: tuple[int, int]) -> tuple[int, int]:
    return x0_shape


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


def infer_shape_nd_identity(x_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Dynamic rank: output shape equals input shape."""
    return x_shape


def infer_shape_three_4d(
    a_shape: tuple[int, int, int, int],
    b_shape: tuple[int, int, int, int],
    c_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return a_shape


def infer_shape_two_outputs(
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a_shape, b_shape)


def infer_dtype_one(a_dtype: torch.dtype) -> torch.dtype:
    return a_dtype


def infer_dtype_two(a_dtype: torch.dtype, b_dtype: torch.dtype) -> torch.dtype:
    return a_dtype


def infer_dtype_three(
    a_dtype: torch.dtype, b_dtype: torch.dtype, c_dtype: torch.dtype
) -> torch.dtype:
    return a_dtype


def infer_dtype_two_outputs(
    a_dtype: torch.dtype, b_dtype: torch.dtype
) -> tuple[torch.dtype, torch.dtype]:
    return (a_dtype, b_dtype)
